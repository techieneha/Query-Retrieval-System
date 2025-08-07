from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from typing import List, Any
import os
import logging
import asyncio
from dotenv import load_dotenv
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm
from time import perf_counter

load_dotenv()
TEAM_TOKEN = os.getenv("TEAM_TOKEN", "hackrx_token")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup/shutdown"""
    logger.info("🔥 Warming up models...")
    try:
        # Warm up LLM and embeddings
        warmup_docs = await load_pdf("https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D")
        embed_chunks(warmup_docs[:1], "warmup")
        logger.info("✅ Models warmed up")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

app = FastAPI(
    title="Policy Query Engine (Optimized)",
    description="Optimized RAG system using local Mistral and embeddings",
    lifespan=lifespan
)

class RunRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# Control concurrency of LLM calls
LLM_SEMAPHORE = asyncio.Semaphore(4)  # Adjust based on CPU cores

async def process_question(q: str, vectorstore: Any) -> str:
    """Process each question with proper error handling"""
    async with LLM_SEMAPHORE:
        try:
            top_docs = await retrieve_with_rerank(q, vectorstore)
            return await answer_with_llm(q, top_docs)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "Answer unavailable."

@app.post("/api/v1/hackrx/run")
async def run_query(request: Request, req: RunRequest, authorization: str = Header(...)):
    request.state.start_time = perf_counter()
    
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        logger.info(f"🚀 Processing {len(req.questions)} questions...")
        
        documents = await load_pdf(req.documents)
        vectorstore = embed_chunks(documents, req.documents)
        
        answers = await asyncio.gather(*[
            process_question(q, vectorstore)
            for q in req.questions
        ])

        total_time = perf_counter() - request.state.start_time
        logger.info(f"✅ Completed in {total_time:.2f}s")
        return {"answers": answers}

    except Exception as e:
        logger.exception("❌ Processing failed")
        raise HTTPException(500, "Query processing failed")

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    try:
        response = await call_next(request)
        if hasattr(request.state, 'start_time'):
            response.headers["X-Processing-Time"] = f"{perf_counter() - request.state.start_time:.3f}s"
        response.headers["Keep-Alive"] = "timeout=600"
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "1.0.0"}