"""rag_pipeline/analytics.py — lightweight in-Redis metrics."""
import os, json, time
from datetime import datetime
from typing import Optional
import redis
from loguru import logger

class Analytics:
    def __init__(self):
        self._r: Optional[redis.Redis] = None

    @property
    def r(self):
        if self._r is None:
            try:
                r = redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                                port=int(os.getenv("REDIS_PORT",6379)),
                                decode_responses=True)
                r.ping(); self._r = r
            except Exception:
                self._r = False
        return self._r if self._r else None

    def log_upload(self, file_id: str, filename: str):
        if not self.r: return
        self.r.incr("stats:uploads")
        self.r.lpush("recent:uploads", json.dumps({
            "file_id": file_id, "filename": filename,
            "ts": datetime.now().isoformat()
        }))
        self.r.ltrim("recent:uploads", 0, 49)

    def log_query(self, file_id: str, question: str, confidence: float,
                  latency_ms: int = 0, cached: bool = False):
        if not self.r: return
        self.r.incr("stats:queries")
        if cached: self.r.incr("stats:cache_hits")
        self.r.lpush("recent:queries", json.dumps({
            "file_id": file_id, "question": question[:80],
            "confidence": confidence, "latency_ms": latency_ms,
            "cached": cached, "ts": datetime.now().isoformat()
        }))
        self.r.ltrim("recent:queries", 0, 99)
        # rolling avg confidence
        self.r.lpush("confidences", confidence)
        self.r.ltrim("confidences", 0, 99)

    def log_claim(self, claim_id: str, policy_number: str):
        if not self.r: return
        self.r.incr("stats:claims")
        self.r.lpush("recent:claims", json.dumps({
            "claim_id": claim_id, "policy_number": policy_number,
            "ts": datetime.now().isoformat()
        }))
        self.r.ltrim("recent:claims", 0, 49)

    def get_stats(self) -> dict:
        if not self.r:
            return {"error": "Redis unavailable"}
        queries    = int(self.r.get("stats:queries") or 0)
        uploads    = int(self.r.get("stats:uploads") or 0)
        claims     = int(self.r.get("stats:claims")  or 0)
        cache_hits = int(self.r.get("stats:cache_hits") or 0)
        confidences= [float(c) for c in (self.r.lrange("confidences",0,99) or [])]
        avg_conf   = round(sum(confidences)/len(confidences), 3) if confidences else 0
        return {
            "total_queries":  queries,
            "total_uploads":  uploads,
            "total_claims":   claims,
            "cache_hit_rate": round(cache_hits / max(queries,1), 3),
            "avg_confidence": avg_conf,
            "recent_queries": [json.loads(q) for q in (self.r.lrange("recent:queries",0,9) or [])],
            "recent_claims":  [json.loads(c) for c in (self.r.lrange("recent:claims",0,9)  or [])],
        }