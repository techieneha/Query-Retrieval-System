import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from typing import List, Any, Dict
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)


try:
    from huggingface_hub import login
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HUGGINGFACE_TOKEN:
        login(token=HUGGINGFACE_TOKEN)
        logger.info("✅ HuggingFace authentication successful")
except Exception as e:
    logger.warning(f"⚠️ HuggingFace login failed: {e}")


try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "policy-docs"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(10)
    
    pinecone_index = pc.Index(index_name)
    logger.info("✅ Pinecone connected")
except Exception as e:
    logger.error(f"Pinecone init failed: {str(e)}")
    raise


try:
    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("✅ BGE embeddings loaded")
except Exception as e:
    logger.error(f"Embeddings failed: {e}")
    raise


async def load_pdf(url: str) -> List[Any]:
    """Load PDF from local path"""
    try:
        if os.path.exists(url):
            logger.info(f"📄 Loading: {url}")
            loader = PyMuPDFLoader(url)
            docs = loader.load()[:50]
            logger.info(f"✅ Loaded {len(docs)} pages")
            return docs
        else:
            raise ValueError(f"File not found: {url}")
    except Exception as e:
        logger.error(f"PDF load failed: {str(e)}")
        raise


async def embed_chunks(documents: List[Any], doc_id: str) -> None:
    """Embed and store in Pinecone"""
    if not documents:
        logger.warning("No documents to embed")
        return
    
    logger.info(f"🔄 Processing {len(documents)} documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". "]
    )
    splits = splitter.split_documents(documents)
    logger.info(f"📝 Created {len(splits)} chunks")
    
    try:
        texts = [doc.page_content for doc in splits]
        embeddings = EMBEDDING_MODEL.embed_documents(texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            
            page = splits[i].metadata.get('page', 0)
            
            vectors.append({
                "id": f"{doc_id}-{i}",
                "values": embedding,
                "metadata": {
                    "text": text,
                    "doc_id": doc_id,
                    "page": page,
                    "chunk_index": i
                }
            })
        
        if vectors:
            pinecone_index.upsert(vectors=vectors)
            logger.info(f"📚 Uploaded {len(vectors)} vectors for {doc_id}")
            
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise


async def retrieve_with_rerank(query: str, doc_id: str, k: int = 3) -> Dict:
    """
    Enhanced retrieval with confidence scores
    Returns: {
        'chunks': List[str],
        'scores': List[float],
        'metadata': List[dict],
        'confidence': float
    }
    """
    try:
        query_embedding = EMBEDDING_MODEL.embed_query(query)
        logger.info(f"🔍 Query: '{query}'")
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=k*2,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=True
        )
        
        logger.info(f"✅ Found {len(results.matches)} matches")
        
        if not results.matches:
            return {
                'chunks': [],
                'scores': [],
                'metadata': [],
                'confidence': 0.0
            }
        
        chunks = []
        scores = []
        metadata = []
        
        for match in results.matches[:k]:
            chunks.append(match.metadata["text"])
            scores.append(float(match.score))
            
            metadata.append({
                "page": match.metadata.get("page", "Unknown"),
                "score": float(match.score),
                "chunk_id": match.id
            })
        
      
        avg_score = sum(scores) / len(scores) if scores else 0
        top_score = scores[0] if scores else 0
        score_variance = max(scores) - min(scores) if len(scores) > 1 else 0
        
       
        confidence = min(1.0, top_score * (1 + score_variance * 0.5))
        
        logger.info(f"📊 Confidence: {confidence:.2%}, Top score: {top_score:.3f}")
        
        return {
            'chunks': chunks,
            'scores': scores,
            'metadata': metadata,
            'confidence': round(confidence, 3)
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return {
            'chunks': [],
            'scores': [],
            'metadata': [],
            'confidence': 0.0
        }