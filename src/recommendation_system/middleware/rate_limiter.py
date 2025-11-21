"""Rate limiting middleware for API protection."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import structlog
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from recommendation_system.config import get_settings

logger = structlog.get_logger()


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


@dataclass
class RateLimitState:
    """State for a single rate limit bucket."""

    tokens: float
    last_update: float
    request_count: int = 0


class RateLimiter:
    """Token bucket rate limiter with Redis or in-memory backend."""

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        burst_limit: int = 20,
        redis_url: str | None = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_window: Max requests per time window.
            window_seconds: Time window in seconds.
            burst_limit: Max burst requests allowed.
            redis_url: Redis URL for distributed rate limiting.
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit
        self.rate = requests_per_window / window_seconds

        self._redis_client = None
        self._memory_buckets: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(
                tokens=float(burst_limit),
                last_update=time.time(),
            )
        )

        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self._redis_client.ping()
                logger.info("Rate limiter using Redis backend")
            except Exception as e:
                logger.warning("Redis connection failed for rate limiter", error=str(e))
                self._redis_client = None

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try API key first, then IP
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"rate:key:{api_key[:16]}"

        # Get real IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"rate:ip:{ip}"

    def _check_redis(self, key: str) -> tuple[bool, int, dict]:
        """Check rate limit using Redis."""
        now = time.time()
        pipe = self._redis_client.pipeline()

        # Get current state
        pipe.hgetall(key)
        pipe.execute()

        state = self._redis_client.hgetall(key)

        if not state:
            # Initialize new bucket
            self._redis_client.hset(key, mapping={
                "tokens": str(self.burst_limit - 1),
                "last_update": str(now),
                "count": "1",
            })
            self._redis_client.expire(key, self.window_seconds * 2)
            return True, self.burst_limit - 1, {"remaining": self.burst_limit - 1}

        tokens = float(state.get("tokens", self.burst_limit))
        last_update = float(state.get("last_update", now))
        count = int(state.get("count", 0))

        # Refill tokens
        time_passed = now - last_update
        tokens = min(self.burst_limit, tokens + time_passed * self.rate)

        if tokens >= 1:
            # Allow request
            tokens -= 1
            self._redis_client.hset(key, mapping={
                "tokens": str(tokens),
                "last_update": str(now),
                "count": str(count + 1),
            })
            self._redis_client.expire(key, self.window_seconds * 2)
            return True, int(tokens), {"remaining": int(tokens)}
        else:
            # Rate limited
            retry_after = int((1 - tokens) / self.rate)
            return False, retry_after, {"retry_after": retry_after}

    def _check_memory(self, key: str) -> tuple[bool, int, dict]:
        """Check rate limit using in-memory storage."""
        now = time.time()
        state = self._memory_buckets[key]

        # Refill tokens
        time_passed = now - state.last_update
        state.tokens = min(self.burst_limit, state.tokens + time_passed * self.rate)
        state.last_update = now

        if state.tokens >= 1:
            state.tokens -= 1
            state.request_count += 1
            return True, int(state.tokens), {"remaining": int(state.tokens)}
        else:
            retry_after = int((1 - state.tokens) / self.rate)
            return False, retry_after, {"retry_after": retry_after}

    def check(self, request: Request) -> tuple[bool, int, dict]:
        """Check if request is allowed.

        Returns:
            Tuple of (allowed, remaining_or_retry, metadata)
        """
        key = self._get_client_key(request)

        if self._redis_client:
            try:
                return self._check_redis(key)
            except Exception as e:
                logger.warning("Redis rate limit check failed", error=str(e))
                return self._check_memory(key)
        else:
            return self._check_memory(key)

    def reset(self, request: Request) -> None:
        """Reset rate limit for a client."""
        key = self._get_client_key(request)
        if self._redis_client:
            self._redis_client.delete(key)
        else:
            self._memory_buckets.pop(key, None)

    def cleanup_memory(self, max_age: int = 3600) -> int:
        """Clean up expired in-memory buckets."""
        now = time.time()
        expired = [
            k for k, v in self._memory_buckets.items()
            if now - v.last_update > max_age
        ]
        for k in expired:
            del self._memory_buckets[k]
        return len(expired)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        allowed, value, metadata = self.rate_limiter.check(request)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                path=request.url.path,
                client=self.rate_limiter._get_client_key(request),
            )
            raise RateLimitExceeded(retry_after=value)

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
        response.headers["X-RateLimit-Window"] = str(self.rate_limiter.window_seconds)

        return response


def rate_limit_middleware(app, settings=None):
    """Factory function to create rate limit middleware."""
    settings = settings or get_settings()

    if not settings.rate_limit_enabled:
        return app

    rate_limiter = RateLimiter(
        requests_per_window=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
        burst_limit=settings.rate_limit_burst,
        redis_url=settings.redis_url,
    )

    return RateLimitMiddleware(app, rate_limiter)
