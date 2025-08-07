from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os
import logging
import asyncio
from dotenv import load_dotenv
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank  # Fixed import path
from rag_pipeline.llm_reasoner import answer_with_llm
from fastapi import Request
from fastapi.responses import JSONResponse
from time import perf_counter

load_dotenv()
TEAM_TOKEN = os.getenv("TEAM_TOKEN", "hackrx_token")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Policy Query Engine (Local)",
    description="RAG system using local Mistral and embeddings"
)

class RunRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

async def process_question(q, vectorstore):
    """Process each question with proper error handling"""
    try:
        top_docs = await retrieve_with_rerank(q, vectorstore)
        return await answer_with_llm(q, top_docs)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return "Answer unavailable."

@app.post("/api/v1/hackrx/run")
async def run_query(req: RunRequest, authorization: str = Header(...)):
    start_time = perf_counter()

    print(f"🔐 Incoming token: {authorization}")
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        logger.info("🚀 Starting query processing...")
        t0 = perf_counter()

        documents = await load_pdf(req.documents)
        t1 = perf_counter()
        logger.info(f"📄 PDF loaded in {t1 - t0:.2f}s")

        vectorstore = embed_chunks(documents, req.documents)
        t2 = perf_counter()
        logger.info(f"📦 Embeddings ready in {t2 - t1:.2f}s")

        answers = await asyncio.gather(*[
            process_question(q, vectorstore)
            for q in req.questions
        ])
        t3 = perf_counter()
        logger.info(f"✅ All questions processed in {t3 - t2:.2f}s")

        logger.info(f"⏱️ Total time: {t3 - start_time:.2f}s")
        return {"answers": answers}

    except Exception as e:
        logger.exception("❌ Processing failed")
        raise HTTPException(500, "Query processing failed")

    

@app.middleware("http")
async def keepalive(request, call_next):
    response = await call_next(request)
    response.headers["Keep-Alive"] = "timeout=600" 
    return response

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "OK", "version": "1.0.0"},
        status_code=200
    )
    