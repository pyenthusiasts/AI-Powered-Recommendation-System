"""Redis-based caching layer for recommendations."""

import hashlib
import json
from datetime import timedelta
from typing import Any, Callable, TypeVar

import structlog

from recommendation_system.config import get_settings

logger = structlog.get_logger()

T = TypeVar("T")


class CacheManager:
    """Manages caching operations with Redis or in-memory fallback."""

    def __init__(self, redis_url: str | None = None, default_ttl: int = 3600):
        """Initialize cache manager.

        Args:
            redis_url: Redis connection URL. If None, uses in-memory cache.
            default_ttl: Default TTL in seconds.
        """
        self.default_ttl = default_ttl
        self._redis_client = None
        self._memory_cache: dict[str, tuple[Any, float]] = {}
        self._connected = False

        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                # Test connection
                self._redis_client.ping()
                self._connected = True
                logger.info("Redis cache connected", url=redis_url.split("@")[-1])
            except Exception as e:
                logger.warning("Redis connection failed, using memory cache", error=str(e))
                self._redis_client = None
        else:
            logger.info("No Redis URL provided, using in-memory cache")

    @property
    def is_redis_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._redis_client is not None

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        hash_value = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"rec:{prefix}:{hash_value}"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if self._redis_client:
            try:
                value = self._redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug("Cache get failed", key=key, error=str(e))
        else:
            import time
            if key in self._memory_cache:
                value, expiry = self._memory_cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self._memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        if self._redis_client:
            try:
                self._redis_client.setex(key, ttl, json.dumps(value, default=str))
                return True
            except Exception as e:
                logger.debug("Cache set failed", key=key, error=str(e))
                return False
        else:
            import time
            self._memory_cache[key] = (value, time.time() + ttl)
            return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if self._redis_client:
            try:
                self._redis_client.delete(key)
                return True
            except Exception:
                return False
        else:
            self._memory_cache.pop(key, None)
            return True

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if self._redis_client:
            try:
                keys = self._redis_client.keys(pattern)
                if keys:
                    return self._redis_client.delete(*keys)
            except Exception:
                pass
            return 0
        else:
            import fnmatch
            deleted = 0
            for key in list(self._memory_cache.keys()):
                if fnmatch.fnmatch(key, pattern):
                    del self._memory_cache[key]
                    deleted += 1
            return deleted

    def clear(self) -> bool:
        """Clear all recommendation cache."""
        return self.delete_pattern("rec:*") > 0 or True

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: int | None = None,
    ) -> T:
        """Get value from cache or compute and store it."""
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    def cache_recommendations(
        self,
        user_id: str | None,
        item_id: str | None,
        strategy: str,
        num: int,
        recommendations: list[dict],
        ttl: int | None = None,
    ) -> None:
        """Cache recommendation results."""
        key = self._generate_key(
            "recs",
            user_id=user_id,
            item_id=item_id,
            strategy=strategy,
            num=num,
        )
        self.set(key, recommendations, ttl or self.default_ttl)

    def get_cached_recommendations(
        self,
        user_id: str | None,
        item_id: str | None,
        strategy: str,
        num: int,
    ) -> list[dict] | None:
        """Get cached recommendations."""
        key = self._generate_key(
            "recs",
            user_id=user_id,
            item_id=item_id,
            strategy=strategy,
            num=num,
        )
        return self.get(key)

    def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user."""
        return self.delete_pattern(f"rec:*user_id*{user_id}*")

    def invalidate_item_cache(self, item_id: str) -> int:
        """Invalidate all cache entries for an item."""
        return self.delete_pattern(f"rec:*item_id*{item_id}*")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._redis_client:
            try:
                info = self._redis_client.info("stats")
                return {
                    "type": "redis",
                    "connected": True,
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                    "keys": self._redis_client.dbsize(),
                }
            except Exception:
                return {"type": "redis", "connected": False}
        else:
            return {
                "type": "memory",
                "connected": True,
                "keys": len(self._memory_cache),
            }


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        _cache_manager = CacheManager(
            redis_url=settings.redis_url if settings.cache_enabled else None,
            default_ttl=settings.cache_ttl,
        )
    return _cache_manager


def reset_cache_manager():
    """Reset the cache manager (for testing)."""
    global _cache_manager
    _cache_manager = None
