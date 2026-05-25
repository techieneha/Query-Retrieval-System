"""
api/main.py
Complete PolicyAI FastAPI application.
Existing endpoints: /api/v1/upload  /api/v1/query  /admin/dashboard
New endpoints:      /api/v1/chat/*  (via chat_router)
"""
import hashlib, os, time, json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from rag_pipeline.retriever  import PolicyRetriever
from rag_pipeline.llm_reasoner import LLMReasoner
from rag_pipeline.cache      import QueryCache
from rag_pipeline.analytics  import Analytics
from claim_agent.agents.chat import router as chat_router

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="PolicyAI",
    description="Insurance AI: RAG Q&A + Conversational Claims Assistant",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(chat_router)

# ── Singletons ────────────────────────────────────────────────────
retriever = PolicyRetriever()
reasoner  = LLMReasoner()
cache     = QueryCache()
analytics = Analytics()


# ── Models ────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    file_id:   str
    questions: list[str]


# ── POST /api/v1/upload ───────────────────────────────────────────
@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    data = await file.read()
    try:
        meta = retriever.ingest(data, file.filename)
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(500, f"Ingestion failed: {e}")

    analytics.log_upload(meta["file_id"], file.filename)
    logger.info(f"Uploaded {file.filename} → file_id={meta['file_id']}")
    return {
        **meta,
        "status":   "indexed",
        "message":  f"'{file.filename}' indexed successfully. Use file_id to start chatting.",
    }


# ── GET /api/v1/files ─────────────────────────────────────────────
@app.get("/api/v1/files")
def list_files():
    return {"files": retriever.list_files()}


# ── POST /api/v1/query ────────────────────────────────────────────
@app.post("/api/v1/query")
def query_document(req: QueryRequest):
    results = []
    for question in req.questions:
        key    = hashlib.md5(f"{req.file_id}:{question}".encode()).hexdigest()
        cached = cache.get(key)
        if cached:
            results.append({**cached, "cached": True})
            continue

        t0     = time.time()
        chunks = retriever.retrieve(question, req.file_id, top_k=4)
        answer, confidence, sources = reasoner.reason(question, chunks)
        latency = round((time.time()-t0)*1000)

        result = {
            "question":   question,
            "answer":     answer,
            "confidence": confidence,
            "sources":    sources,
            "latency_ms": latency,
            "cached":     False,
        }
        cache.set(key, result)
        analytics.log_query(req.file_id, question, confidence, latency)
        results.append(result)

    return {"file_id": req.file_id, "results": results}


# ── GET /admin/dashboard ──────────────────────────────────────────
@app.get("/admin/dashboard")
def dashboard():
    stats = analytics.get_stats()
    return JSONResponse(stats)


# ── GET /health ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "features": ["rag","chat","claims"]}