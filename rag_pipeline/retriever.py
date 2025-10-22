
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import httpx
import logging
from typing import List, Any
import asyncio
from dotenv import load_dotenv
import time


load_dotenv()


logger = logging.getLogger(__name__)

try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "policy-docs"
    
    
    FREE_TIER_REGIONS = {
        "aws": "us-east-1",  
        "gcp": "us-central1" 
    }
    
  
    if index_name not in pc.list_indexes().names():
        
        pc.create_index(
            name=index_name,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  
                region=FREE_TIER_REGIONS["aws"]  
            )
        )
        logger.info(f"Creating new index in free-tier region {FREE_TIER_REGIONS['aws']}")
        
        
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(10)
            logger.info("Waiting for index to be ready...")
    
    pinecone_index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Pinecone initialization failed: {str(e)}")
    raise
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

async def load_pdf(url: str) -> List[Any]:
    """Optimized PDF loader"""
    try:
        if url.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
            try:
                loader = PyMuPDFLoader(tmp_path)
                return loader.load()[:50]  
            finally:
                os.unlink(tmp_path)
        else:
            loader = PyMuPDFLoader(url)
            return loader.load()[:50]
    except Exception as e:
        logger.error(f"PDF loading failed: {str(e)}")
        raise

async def embed_chunks(documents: List[Any], doc_id: str) -> None:
    """Store embeddings in Pinecone with document filtering"""
    if not documents:
        raise ValueError("No documents provided for embedding")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". "]
    )
    splits = splitter.split_documents(documents)
    
   
    texts = [doc.page_content for doc in splits]
    embeddings = EMBEDDING_MODEL.embed_documents(texts)
    
   
    vectors = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        vector_id = f"{doc_id}-{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,  
            "metadata": {
                "text": text,
                "doc_id": doc_id
            }
        })
    
    
    try:
        pinecone_index.upsert(vectors=vectors)
        logger.info(f"Sample embedded text: {texts[0][:100]}...")  
        logger.info(f"Sample embedding length: {len(embeddings[0])}") 
        logger.info(f"📚 Uploaded {len(vectors)} vectors for doc {doc_id}")
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {str(e)}")
        raise

async def retrieve_with_rerank(query: str, doc_id: str, k: int = 3) -> List[str]:
    """Retrieve from Pinecone using document-specific filtering"""
    try:
    
        query_embedding = EMBEDDING_MODEL.embed_query(query)
        logger.info(f"Query: '{query}'")
        logger.info(f"Query embedding length: {len(query_embedding)}")
        
      
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=k*2,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=True
        )
        
        logger.info(f"Found {len(results.matches)} matches")
        for i, match in enumerate(results.matches[:3]):
            logger.info(f"Match {i+1}: Score={match.score:.3f}, Text={match.metadata['text'][:100]}...")
        
        return [match.metadata["text"] for match in results.matches[:k]]
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return []
