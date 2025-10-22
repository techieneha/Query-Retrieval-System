import warnings
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging
import time
import uuid
import sys
import json
import atexit



warnings.filterwarnings("ignore")


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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

app = FastAPI(
    title="Insurance RAG API",
    description="RAG-powered insurance policy assistant",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500", 
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STORAGE_FILE = "file_storage.json"

def load_storage():
    """Load file storage from disk"""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading storage: {e}")
    return {}

def save_storage():
    """Save file storage to disk"""
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump(file_storage, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving storage: {e}")


file_storage = load_storage()


atexit.register(save_storage)


class QueryRequest(BaseModel):
    file_id: str
    questions: List[str]

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str

class QueryResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document: str


try:
    from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
    from rag_pipeline.llm_reasoner import answer_with_llm
    logger.info("✅ Successfully imported RAG components from rag_pipeline")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    
    
    from langchain.schema import Document
    
    async def load_pdf(file_path):
        logger.info(f"📄 Loading PDF: {file_path}")
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            logger.info(f"✅ Loaded {len(docs)} pages from PDF")
            return docs
        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
            raise Exception(f"Failed to load PDF: {e}")
    
    async def embed_chunks(documents, doc_id):
        logger.info(f"📚 Processing {len(documents)} documents for {doc_id}")
        
        if not hasattr(embed_chunks, 'local_storage'):
            embed_chunks.local_storage = {}
        embed_chunks.local_storage[doc_id] = documents
    
    async def retrieve_with_rerank(query, doc_id, k=3):
        logger.info(f"🔍 Query: {query} for {doc_id}")
        if hasattr(embed_chunks, 'local_storage') and doc_id in embed_chunks.local_storage:
            documents = embed_chunks.local_storage[doc_id]
            
            query_terms = query.lower().split()
            relevant_chunks = []
            for doc in documents:
                content = doc.page_content.lower()
                if any(term in content for term in query_terms):
                    relevant_chunks.append(doc.page_content)
                    if len(relevant_chunks) >= k:
                        break
            return relevant_chunks if relevant_chunks else ["No specific information found in the document."]
        return ["Document content not available."]
    
    async def answer_with_llm(query, context_clauses):
        if not context_clauses:
            return "No relevant information found in the document."
        
        combined_context = " ".join(context_clauses[:2])
        return f"Based on the policy document: {combined_context[:200]}..."

@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF file and process embeddings"""
    logger.info(f"📤 Upload request received for file: {file.filename}")
    
    try:
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        logger.info(f"💾 Saving file to: {file_path}")
        
        
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        
        logger.info(f"📄 Processing PDF: {file.filename}")
        docs = await load_pdf(file_path)
        logger.info(f"✅ Loaded {len(docs)} document pages")
        
        
        if docs and len(docs) > 0:
            sample_content = docs[0].page_content[:500].replace('\n', ' ')
            logger.info(f"📖 Document preview: {sample_content}...")
        
        
        doc_id = f"doc_{file_id}"
        await embed_chunks(docs, doc_id)
        logger.info(f"✅ Processing completed for doc_id: {doc_id}")
        
        
        file_storage[file_id] = {
            "file_path": file_path,
            "original_name": file.filename,
            "upload_time": time.time(),
            "file_size": len(content),
            "doc_id": doc_id,
            "processed": True
        }
        
        save_storage()
        
        logger.info(f"✅ File processed successfully: {file.filename}")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            message="File uploaded and processed successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Upload/Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process insurance policy questions"""
    start_time = time.time()
    logger.info(f"🔵 Query request for file: {request.file_id}")
    
    try:
        
        if request.file_id not in file_storage:
            logger.error(f"❌ File ID not found: {request.file_id}")
            raise HTTPException(status_code=404, detail="File not found. Please upload again.")
        
        file_info = file_storage[request.file_id]
        
        if not file_info.get("processed", False):
            raise HTTPException(status_code=400, detail="File is still processing. Please wait.")
        
        doc_id = file_info["doc_id"]
        logger.info(f"📄 Processing query for document: {file_info['original_name']}")
        
        
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"❓ Processing question {i+1}: {question}")
            
            # Retrieve relevant chunks
            relevant_chunks = await retrieve_with_rerank(question, doc_id)
            logger.info(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Generate answer
            answer = await answer_with_llm(question, relevant_chunks)
            answers.append(answer)
            logger.info(f"✅ Generated answer for question {i+1}")
        
        processing_time = time.time() - start_time
        logger.info(f"🎯 Completed processing in {processing_time:.2f} seconds")
        
        return QueryResponse(
            answers=answers,
            processing_time=processing_time,
            document=file_info["original_name"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/v1/files/{file_id}")
async def get_file_status(file_id: str):
    """Check if a file exists and is processed"""
    if file_id in file_storage:
        file_info = file_storage[file_id]
        return {
            "exists": True,
            "processed": file_info.get("processed", False),
            "filename": file_info["original_name"],
            "upload_time": file_info["upload_time"]
        }
    return {"exists": False}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "uploaded_files": len(file_storage),
        "api": "Insurance RAG API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "message": "Insurance RAG API is running!",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "health": "/health",
            "file_status": "/api/v1/files/{file_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Insurance RAG API Server...")
    print("📍 Backend URL: http://127.0.0.1:8000")
    print("📍 Frontend URL: http://127.0.0.1:5500")
    print("📝 Available endpoints:")
    print("   POST /api/v1/upload - Upload PDF file")
    print("   POST /api/v1/query - Ask questions about policy")
    print("   GET /api/v1/files/{file_id} - Check file status")
    print("   GET /health - Health check")
    print("🔧 CORS enabled for frontend: http://127.0.0.1:5500")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )