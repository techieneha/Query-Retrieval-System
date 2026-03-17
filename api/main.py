import warnings
from huggingface_hub.utils import disable_progress_bars
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import logging
import time
import uuid
import sys
import json
import asyncio


from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm, stream_llm_answer
from rag_pipeline.cache import get_cached_result, cache_result, get_cache_stats
from rag_pipeline.analytics import analytics_tracker

disable_progress_bars()
warnings.filterwarnings("ignore", category=FutureWarning)


UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance RAG API - Production")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_storage = {}


class QueryRequest(BaseModel):
    file_id: str
    questions: List[str]

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str

class QueryResponse(BaseModel):
    results: List[dict]
    processing_time: float
    document: str
    cache_stats: dict




@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process PDF"""
    logger.info(f"📤 Upload: {file.filename}")
    
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process PDF
        docs = await load_pdf(file_path)
        logger.info(f"✅ Loaded {len(docs)} pages")
        
        # Generate embeddings
        doc_id = f"doc_{file_id}"
        await embed_chunks(docs, doc_id)
        logger.info(f"✅ Embeddings ready for {doc_id}")
        
        # Store metadata
        file_storage[file_id] = {
            "file_path": file_path,
            "original_name": file.filename,
            "upload_time": time.time(),
            "file_size": len(content),
            "doc_id": doc_id,
            "processed": True,
            "num_pages": len(docs)
        }
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            message=f"Processed {len(docs)} pages successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process queries with caching and analytics"""
    start_time = time.time()
    logger.info(f"🔵 Query for file: {request.file_id}")
    
    try:
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[request.file_id]
        doc_id = file_info["doc_id"]
        
        results = []
        cache_hits = 0
        
        for question in request.questions:
            query_start = time.time()
            logger.info(f"❓ Processing: {question}")
            
            # Try cache first
            cached = await get_cached_result(doc_id, question)
            
            if cached:
                cached['cached'] = True
                results.append(cached)
                cache_hits += 1
                
                # Still track analytics for cached queries
                analytics_tracker.log_query({
                    'query': question,
                    'doc_id': doc_id,
                    'latency': time.time() - query_start,
                    'confidence': cached['confidence'],
                    'cached': True,
                    'quality': cached['quality']
                })
                continue
            
            # Cache miss - process normally
            retrieval_result = await retrieve_with_rerank(question, doc_id)
            answer_result = await answer_with_llm(question, retrieval_result)
            
            query_latency = time.time() - query_start
            
            result = {
                'question': question,
                'answer': answer_result['answer'],
                'confidence': answer_result['confidence'],
                'sources': answer_result['sources'],
                'quality': answer_result['quality'],
                'cached': False,
                'latency': round(query_latency, 3)
            }
            
            # Cache for future
            await cache_result(doc_id, question, result, ttl=3600)
            
            # Track analytics
            analytics_tracker.log_query({
                'query': question,
                'doc_id': doc_id,
                'latency': query_latency,
                'confidence': answer_result['confidence'],
                'cached': False,
                'quality': answer_result['quality']
            })
            
            results.append(result)
            logger.info(f"✅ Confidence: {answer_result['confidence']:.2%}")
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            results=results,
            processing_time=round(processing_time, 2),
            document=file_info["original_name"],
            cache_stats={
                'hits': cache_hits,
                'misses': len(request.questions) - cache_hits,
                'hit_rate': round(cache_hits / len(request.questions) * 100, 1) if request.questions else 0
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query-stream")
async def stream_query(request: QueryRequest):
    """Stream answers token-by-token"""
    
    async def generate_stream():
        try:
            if request.file_id not in file_storage:
                yield f"data: {json.dumps({'error': 'File not found'})}\n\n"
                return
            
            file_info = file_storage[request.file_id]
            doc_id = file_info["doc_id"]
            
            for i, question in enumerate(request.questions):
                yield f"data: {json.dumps({'status': f'Processing question {i+1}...', 'index': i})}\n\n"
                
                # Check cache first
                cached = await get_cached_result(doc_id, question)
                if cached:
                    yield f"data: {json.dumps({'answer': cached['answer'], 'cached': True, 'index': i})}\n\n"
                    continue
                
                # Retrieve
                retrieval_result = await retrieve_with_rerank(question, doc_id)
                yield f"data: {json.dumps({'status': 'Generating answer...', 'sources': len(retrieval_result['chunks']), 'index': i})}\n\n"
                
                # Stream LLM response
                full_answer = ""
                async for token in stream_llm_answer(question, retrieval_result):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'index': i})}\n\n"
                    await asyncio.sleep(0.01)
                
                # Send completion with metadata
                yield f"data: {json.dumps({'complete': True, 'index': i, 'confidence': retrieval_result['confidence']})}\n\n"
                
                # Cache the result
                await cache_result(doc_id, question, {
                    'answer': full_answer,
                    'confidence': retrieval_result['confidence'],
                    'quality': 'good'
                }, ttl=3600)
        
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )




@app.get("/api/v1/analytics/stats")
async def get_analytics(hours: int = 24):
    """Get system analytics"""
    return analytics_tracker.get_stats(hours)


@app.get("/api/v1/analytics/popular")
async def get_popular_queries(limit: int = 10):
    """Get most frequently asked questions"""
    return {
        'popular_queries': analytics_tracker.get_popular_queries(limit)
    }


@app.get("/api/v1/cache/stats")
async def cache_stats():
    """Get cache performance"""
    return get_cache_stats()




@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard():
    """Analytics dashboard"""
    stats = analytics_tracker.get_stats(24)
    popular = analytics_tracker.get_popular_queries(5)
    cache_info = get_cache_stats()
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Analytics Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, system-ui; padding: 40px; background: #0f172a; color: white; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ margin-bottom: 40px; }}
            h1 {{ font-size: 32px; margin-bottom: 8px; }}
            .subtitle {{ color: #94a3b8; }}
            .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid #334155; }}
            .metric-value {{ font-size: 36px; font-weight: bold; margin: 12px 0 8px; }}
            .metric-label {{ color: #94a3b8; font-size: 14px; }}
            .blue {{ color: #3b82f6; }}
            .green {{ color: #10b981; }}
            .purple {{ color: #a855f7; }}
            .yellow {{ color: #f59e0b; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
            th {{ background: #0f172a; font-weight: 600; color: #94a3b8; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
            .badge-excellent {{ background: #10b98120; color: #10b981; }}
            .badge-good {{ background: #3b82f620; color: #3b82f6; }}
            .badge-fair {{ background: #f59e0b20; color: #f59e0b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 RAG System Analytics</h1>
                <p class="subtitle">Real-time performance metrics • Auto-refresh every 30s</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <div class="metric-label">Total Queries</div>
                    <div class="metric-value blue">{stats.get('total_queries', 0)}</div>
                    <div class="metric-label">{stats.get('queries_per_hour', 0)}/hour</div>
                </div>
                
                <div class="card">
                    <div class="metric-label">Avg Latency</div>
                    <div class="metric-value green">{stats.get('latency', {}).get('mean', 0)}s</div>
                    <div class="metric-label">P95: {stats.get('latency', {}).get('p95', 0)}s</div>
                </div>
                
                <div class="card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value purple">{stats.get('confidence', {}).get('mean', 0)}</div>
                    <div class="metric-label">Min: {stats.get('confidence', {}).get('min', 0)}</div>
                </div>
                
                <div class="card">
                    <div class="metric-label">Cache Hit Rate</div>
                    <div class="metric-value yellow">{cache_info.get('hit_rate', 0)}%</div>
                    <div class="metric-label">{cache_info.get('total_keys', 0)} keys</div>
                </div>
            </div>
            
            <div class="card" style="margin-bottom: 30px;">
                <h2 style="margin-bottom: 20px;">Popular Queries</h2>
                <table>
                    <tr>
                        <th>Query</th>
                        <th>Count</th>
                    </tr>
                    {''.join(f"<tr><td>{q['query']}</td><td>{q['count']}</td></tr>" for q in popular) if popular else '<tr><td colspan="2" style="text-align:center; color: #64748b;">No queries yet</td></tr>'}
                </table>
            </div>
            
            <div class="grid" style="grid-template-columns: 1fr 1fr;">
                <div class="card">
                    <h3 style="margin-bottom: 16px;">Latency Breakdown</h3>
                    <p>P50: <strong>{stats.get('latency', {}).get('median', 0)}s</strong></p>
                    <p>P95: <strong>{stats.get('latency', {}).get('p95', 0)}s</strong></p>
                    <p>P99: <strong>{stats.get('latency', {}).get('p99', 0)}s</strong></p>
                    <p>Max: <strong>{stats.get('latency', {}).get('max', 0)}s</strong></p>
                </div>
                
                <div class="card">
                    <h3 style="margin-bottom: 16px;">Quality Distribution</h3>
                    {''.join(f'<p>{k.title()}: <strong>{v}</strong> queries</p>' for k, v in stats.get('quality_distribution', {}).items())}
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    cache_info = get_cache_stats()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uploaded_files": len(file_storage),
        "cache_enabled": cache_info.get("enabled", False),
        "total_queries_tracked": len(analytics_tracker.queries)
    }


@app.get("/")
async def root():
    return {
        "message": "Insurance RAG API - Production Ready",
        "version": "2.0.0",
        "features": [
            "Confidence scoring",
            "Source citations",
            "Redis caching",
            "Real-time analytics",
            "Streaming responses"
        ],
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "stream": "/api/v1/query-stream",
            "analytics": "/api/v1/analytics/stats",
            "dashboard": "/admin/dashboard"
        }
    }