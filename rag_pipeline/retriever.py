from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import httpx
import os
import logging
import numpy as np
from typing import List, Any
import simsimd

logger = logging.getLogger(__name__)

EMBED_CACHE = {}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "batch_size": 128,
        "convert_to_numpy": True,
        "normalize_embeddings": True
    }
)

async def load_pdf(url: str) -> List[Any]:
    """Optimized PDF loader with streaming"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        try:
            return loader.load()
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"PDF loading failed: {str(e)}")
        raise

def embed_chunks(documents: List[Any], url: str) -> FAISS:
    """Efficient embedding with caching"""
    if url in EMBED_CACHE:
        return EMBED_CACHE[url]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    faiss_store = FAISS.from_documents(
        splits,
        EMBEDDING_MODEL
    )
    EMBED_CACHE[url] = faiss_store
    return faiss_store

async def retrieve_with_rerank(query: str, vectorstore: FAISS, k: int = 3) -> List[Any]:
    """Hybrid retrieval with fallback"""
    try:
        docs = await vectorstore.asimilarity_search(query, k=k*3)
        if len(docs) <= 1:
            return docs[:k]
            
        query_embedding = np.array(await vectorstore.embeddings.aembed_query(query), dtype=np.float32)
        doc_embeddings = np.array([d.metadata.get("embedding", []) for d in docs if hasattr(d.metadata, "get")], dtype=np.float32)
        
        if doc_embeddings.size == 0:
            return docs[:k]

        scores = simsimd.cosine(query_embedding, doc_embeddings.T)
        return [docs[i] for i in np.argsort(scores)[-k:][::-1]]
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return await vectorstore.asimilarity_search(query, k=k)