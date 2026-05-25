"""
rag_pipeline/retriever.py
Handles PDF ingestion, chunking, embedding, Pinecone upsert and retrieval.
"""
import os, json, hashlib, time
from pathlib import Path
from typing import Optional
from loguru import logger
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import redis

FILE_STORAGE = os.getenv("FILE_STORAGE", "file_storage.json")
UPLOAD_DIR   = os.getenv("UPLOAD_DIR",   "uploaded_docs")
CHUNK_SIZE   = 600
CHUNK_OVERLAP= 100


def _load_storage() -> dict:
    if os.path.exists(FILE_STORAGE):
        with open(FILE_STORAGE) as f:
            return json.load(f)
    return {}

def _save_storage(data: dict):
    with open(FILE_STORAGE, "w") as f:
        json.dump(data, f, indent=2)


class PolicyRetriever:
    def __init__(self):
        self._model:  Optional[SentenceTransformer] = None
        self._index   = None
        self._redis:  Optional[redis.Redis]         = None

    # ── lazy loaders ────────────────────────────────────────────
    @property
    def model(self) -> SentenceTransformer:
        if not self._model:
            logger.info("Loading BGE model…")
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    @property
    def index(self):
        if not self._index:
            pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            name  = os.getenv("PINECONE_INDEX", "policyai-index")
            names = [i.name for i in pc.list_indexes()]
            if name not in names:
                pc.create_index(
                    name=name, dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                time.sleep(2)
            self._index = pc.Index(name)
        return self._index

    @property
    def cache(self) -> Optional[redis.Redis]:
        if self._redis is None:
            try:
                r = redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                                port=int(os.getenv("REDIS_PORT",6379)),
                                decode_responses=True)
                r.ping(); self._redis = r
            except Exception:
                self._redis = False          # mark as unavailable
        return self._redis if self._redis else None

    # ── public API ───────────────────────────────────────────────
    def ingest(self, pdf_bytes: bytes, filename: str) -> dict:
        """Parse PDF → chunk → embed → upsert. Returns metadata dict."""
        file_id = hashlib.md5(pdf_bytes).hexdigest()[:12]

        # Save file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        dest = Path(UPLOAD_DIR) / f"{file_id}_{filename}"
        dest.write_bytes(pdf_bytes)

        # Extract text
        reader = PdfReader(dest)
        pages  = [p.extract_text() or "" for p in reader.pages]
        full   = "\n".join(pages)

        # Chunk
        chunks = self._chunk(full)
        logger.info(f"[Ingest] {filename} → {len(chunks)} chunks")

        # Embed + upsert
        ns  = os.getenv("PINECONE_NAMESPACE","policies")
        ids, vecs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            vec = self.model.encode(chunk, normalize_embeddings=True).tolist()
            ids.append(f"{file_id}_{i}")
            vecs.append(vec)
            metas.append({"file_id": file_id, "text": chunk, "chunk_idx": i})

        # Upsert in batches of 100
        for start in range(0, len(ids), 100):
            self.index.upsert(
                vectors=list(zip(ids[start:start+100],
                                 vecs[start:start+100],
                                 metas[start:start+100])),
                namespace=ns,
            )

        # Persist metadata
        storage = _load_storage()
        storage[file_id] = {
            "file_id":  file_id,
            "filename": filename,
            "pages":    len(pages),
            "chunks":   len(chunks),
            "path":     str(dest),
        }
        _save_storage(storage)
        return storage[file_id]

    def retrieve(self, query: str, file_id: str, top_k: int = 4) -> list[dict]:
        """Semantic search → ranked chunks with score."""
        cache_key = f"rag:{hashlib.md5(f'{file_id}:{query}'.encode()).hexdigest()}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)

        vec  = self.model.encode(query, normalize_embeddings=True).tolist()
        ns   = os.getenv("PINECONE_NAMESPACE","policies")
        resp = self.index.query(
            vector=vec, top_k=top_k,
            filter={"file_id": {"$eq": file_id}},
            include_metadata=True,
            namespace=ns,
        )
        results = [
            {"text": m.metadata.get("text",""), "score": m.score, "metadata": m.metadata}
            for m in resp.matches if m.metadata.get("text")
        ]
        if self.cache and results:
            self.cache.setex(cache_key, 3600, json.dumps(results))
        return results

    def get_file_info(self, file_id: str) -> Optional[dict]:
        return _load_storage().get(file_id)

    def list_files(self) -> list[dict]:
        return list(_load_storage().values())

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _chunk(text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end].strip())
            start = end - CHUNK_OVERLAP
        return [c for c in chunks if len(c) > 40]