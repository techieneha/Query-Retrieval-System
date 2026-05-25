"""rag_pipeline/cache.py — Redis query cache."""
import os, json
from typing import Optional
import redis
from loguru import logger

class QueryCache:
    def __init__(self):
        self._r: Optional[redis.Redis] = None

    @property
    def r(self) -> Optional[redis.Redis]:
        if self._r is None:
            try:
                r = redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                                port=int(os.getenv("REDIS_PORT",6379)),
                                decode_responses=True)
                r.ping(); self._r = r
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
                self._r = False
        return self._r if self._r else None

    def get(self, key: str) -> Optional[dict]:
        if not self.r: return None
        raw = self.r.get(f"qcache:{key}")
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict, ttl: int = 3600):
        if not self.r: return
        self.r.setex(f"qcache:{key}", ttl, json.dumps(value))

    def delete(self, key: str):
        if self.r: self.r.delete(f"qcache:{key}")

    def flush(self):
        if self.r:
            for k in self.r.scan_iter("qcache:*"):
                self.r.delete(k)