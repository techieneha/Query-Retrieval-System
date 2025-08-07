import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
import asyncio
import logging
import time
import hashlib
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()  

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]
    doc_id: str = None  # Make optional with default None

@app.post("/api/v1/hackrx/run")
async def run_query(req: RunRequest, authorization: str = Header(...)):
    start_time = time.time()
    
    try:
        # Generate doc_id if not provided
        doc_id = req.doc_id or hashlib.md5(req.documents.encode()).hexdigest()
        
        # Load and process document
        docs = await load_pdf(req.documents)
        await embed_chunks(docs, doc_id)  # Pass the doc_id
        
        # Process questions
        answers = await asyncio.gather(*[
            answer_with_llm(q, await retrieve_with_rerank(q, doc_id))  # Pass doc_id here
            for q in req.questions
        ])
        
        return {
            "answers": answers,
            "doc_id": doc_id,  # Return the doc_id for reference
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))