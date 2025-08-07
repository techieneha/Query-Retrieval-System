import warnings
from huggingface_hub.utils import disable_progress_bars
import os
from fastapi import FastAPI, HTTPException, Header,Request
from pydantic import BaseModel
from typing import List
import asyncio
import logging
import time
import hashlib
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm
import sys

disable_progress_bars()  # Disables HuggingFace's progress bars
warnings.filterwarnings("ignore", category=FutureWarning)  # Silences deprecation warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log", mode='w'),  # 'w' to overwrite each run
        logging.StreamHandler(sys.stdout)  # Explicitly log to stdout
    ],
    force=True  # Override any existing handlers
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure your logger catches INFO level

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]
    doc_id: str = None

@app.post("/api/v1/hackrx/run")
async def run_query(req: RunRequest, authorization: str = Header(...)):
    start_time = time.time()
    logger.info(f"🔵 API Hit - Starting processing for {len(req.questions)} questions")
    logger.debug(f"Request details - Documents: {req.documents[:50]}... Questions: {req.questions}")

    try:
        # Document ID handling
        doc_id = req.doc_id or hashlib.md5(req.documents.encode()).hexdigest()
        logger.info(f"Document ID: {doc_id} {'(generated)' if not req.doc_id else '(provided)'}")

        # Document loading
        logger.info("🔄 Loading and processing document...")
        docs = await load_pdf(req.documents)
        logger.info(f"📄 Loaded {len(docs)} document pages")

        # Embedding
        logger.info("🧠 Generating embeddings...")
        await embed_chunks(docs, doc_id)
        logger.info("✅ Embeddings generated and stored")

        # Question processing
        logger.info(f"❓ Processing {len(req.questions)} questions...")
        answers = []
        for i, q in enumerate(req.questions):
            try:
                logger.debug(f"Processing Q{i+1}: {q[:50]}...")
                retrieval_start = time.time()
                relevant_chunks = await retrieve_with_rerank(q, doc_id)
                retrieval_time = time.time() - retrieval_start
                logger.debug(f"Retrieved {len(relevant_chunks)} chunks in {retrieval_time:.2f}s")

                llm_start = time.time()
                answer = await answer_with_llm(q, relevant_chunks)
                llm_time = time.time() - llm_start
                logger.debug(f"Generated answer in {llm_time:.2f}s")

                answers.append(answer)
                logger.info(f"✔️ Q{i+1} processed")
            except Exception as qe:
                logger.error(f"❌ Failed processing Q{i+1}: {str(qe)}")
                answers.append("Error processing question")

        # Final response
        processing_time = time.time() - start_time
        logger.info(f"🏁 Completed in {processing_time:.2f} seconds")
        logger.debug(f"Sample answer: {answers[0][:100]}..." if answers else "No answers generated")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"🔥 Critical error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "document_id": doc_id if 'doc_id' in locals() else None,
                "processing_stage": "See logs for details"
            }
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"🔵 Incoming request: {request.method} {request.url}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"✅ Completed: {request.method} {request.url} | Status: {response.status_code} | {process_time:.2f}ms")
    
    return response    