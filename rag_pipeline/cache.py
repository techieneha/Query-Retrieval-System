import json
import hashlib
import os
from typing import Optional
import logging

try:
    import redis  # type: ignore
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", None),
        db=0,
        decode_responses=True,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info("✅ Redis connected")
    CACHE_ENABLED = True
except Exception as e:
    logger.warning(f"⚠️ Redis unavailable: {e}. Running without cache.")
    CACHE_ENABLED = False
    redis_client = None


def generate_cache_key(doc_id: str, query: str) -> str:
    """Generate deterministic cache key"""
    normalized = query.lower().strip()
    content = f"{doc_id}:{normalized}"
    return f"rag:query:{hashlib.md5(content.encode()).hexdigest()}"


async def get_cached_result(doc_id: str, query: str) -> Optional[dict]:
    """Retrieve cached query result"""
    if not CACHE_ENABLED:
        return None
    
    try:
        key = generate_cache_key(doc_id, query)
        cached = redis_client.get(key)
        
        if cached:
            logger.info(f"💚 Cache HIT: {query[:50]}...")
            return json.loads(cached)
        else:
            logger.info(f"💔 Cache MISS: {query[:50]}...")
            return None
            
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None


async def cache_result(doc_id: str, query: str, result: dict, ttl: int = 3600):
    """Cache query result with TTL"""
    if not CACHE_ENABLED:
        return
    
    try:
        key = generate_cache_key(doc_id, query)
        redis_client.setex(key, ttl, json.dumps(result))
        logger.info(f"💾 Cached: {query[:50]}... (TTL: {ttl}s)")
    except Exception as e:
        logger.error(f"Cache write error: {e}")


def get_cache_stats() -> dict:
    """Get cache performance metrics"""
    if not CACHE_ENABLED:
        return {"enabled": False}
    
    try:
        info = redis_client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
        
        return {
            "enabled": True,
            "total_keys": redis_client.dbsize(),
            "hit_rate": round(hit_rate, 2),
            "hits": hits,
            "misses": misses,
            "memory_used": info.get("used_memory_human", "N/A")
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}