import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("advisor.cache")

class SemanticCache:
    """In-memory cache for exact string query matching and parsed intent properties."""
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        # Structure: { query_key: {"data": value, "timestamp": float} }
        self._store: Dict[str, Dict[str, Any]] = {}

    def _clean_expired(self) -> None:
        """Prune items that have outlived their Time-To-Live duration."""
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items() 
            if now - v["timestamp"] > self.ttl
        ]
        for k in expired_keys:
            del self._store[k]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch cached response for exact string query match if available and valid."""
        self._clean_expired()
        
        normalized_query = query.strip().lower()
        if normalized_query in self._store:
            logger.info(f"Cache HIT for query: '{query}'")
            # Update hit timestamp for LRU-like preservation
            self._store[normalized_query]["timestamp"] = time.time()
            return self._store[normalized_query]["data"]
            
        logger.debug(f"Cache MISS for query: '{query}'")
        return None

    def set(self, query: str, data: Dict[str, Any]) -> None:
        """Store query results in cache, pruning oldest entries if exceeding bounds."""
        self._clean_expired()
        
        normalized_query = query.strip().lower()
        
        # Evict oldest entry if size boundaries crossed
        if len(self._store) >= self.max_size:
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k]["timestamp"])
            logger.debug(f"Cache overflow. Evicting oldest entry: '{oldest_key}'")
            del self._store[oldest_key]
            
        self._store[normalized_query] = {
            "data": data,
            "timestamp": time.time()
        }
        logger.info(f"Cached search results for query: '{query}'")

# Global instantiator
semantic_cache = SemanticCache()
