"""MCP Middleware for reasoning-mcp server.

This module provides middleware components for request/response interception,
logging, telemetry, rate limiting, response caching, and error handling using
FastMCP v2's middleware system.

FastMCP v2.14+ Features:
- ResponseCacheMiddleware for caching expensive operations
- CacheMetrics for tracking cache performance
- RateLimitMiddleware for rate limiting and abuse prevention
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from mcp.types import CallToolResult, TextContent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import mcp.types as mt
    from fastmcp import FastMCP
    from fastmcp.prompts.prompt import Prompt
    from fastmcp.resources.resource import Resource
    from fastmcp.resources.template import ResourceTemplate
    from fastmcp.server.middleware import MiddlewareResult
    from fastmcp.tools.tool import Tool, ToolResult
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.types import CallToolRequest

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics collected for each request."""

    tool_calls: int = 0
    resource_reads: int = 0
    prompt_calls: int = 0
    total_duration_ms: float = 0.0
    errors: int = 0


@dataclass
class MiddlewareState:
    """Shared state for middleware components."""

    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    request_start_times: dict[str, float] = field(default_factory=dict)


class ReasoningMiddleware(Middleware):
    """Middleware for the reasoning-mcp server.

    Provides:
    - Request/response logging with timing
    - Metrics collection for telemetry
    - Error tracking and recovery
    - Rate limiting support (optional)

    Example:
        >>> middleware = ReasoningMiddleware(
        ...     enable_logging=True,
        ...     enable_metrics=True,
        ... )
        >>> middleware.register(mcp)
    """

    def __init__(
        self,
        *,
        enable_logging: bool = True,
        enable_metrics: bool = True,
        log_level: int = logging.DEBUG,
    ) -> None:
        """Initialize middleware.

        Args:
            enable_logging: Whether to log requests/responses
            enable_metrics: Whether to collect metrics
            log_level: Logging level for request/response logs
        """
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.log_level = log_level
        self.state = MiddlewareState()

    def register(self, mcp: FastMCP) -> None:
        """Register middleware with the FastMCP server."""
        mcp.add_middleware(self)

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Intercept tool calls for logging and metrics."""
        tool_name = context.message.name
        request_id = f"tool_{tool_name}_{time.time()}"

        if self.enable_logging:
            logger.log(
                self.log_level,
                f"Tool call started: {tool_name}",
                extra={"tool_name": tool_name, "request_id": request_id},
            )

        if self.enable_metrics:
            self.state.request_start_times[request_id] = time.perf_counter()
            self.state.metrics.tool_calls += 1

        try:
            result = await call_next(context)

            if self.enable_metrics:
                start = self.state.request_start_times.pop(request_id, None)
                if start:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self.state.metrics.total_duration_ms += duration_ms

            if self.enable_logging:
                logger.log(
                    self.log_level,
                    f"Tool call completed: {tool_name}",
                    extra={"tool_name": tool_name, "request_id": request_id},
                )

            return result

        except Exception as e:
            if self.enable_metrics:
                self.state.metrics.errors += 1

            logger.error(
                f"Tool call failed: {tool_name} - {e}",
                extra={"tool_name": tool_name, "error": str(e)},
            )
            raise

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        """Intercept list_tools for logging."""
        if self.enable_logging:
            logger.log(self.log_level, "Listing tools")

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Listed {len(result)} tools")

        return result

    async def on_read_resource(
        self,
        context: MiddlewareContext[mt.ReadResourceRequestParams],
        call_next: CallNext[mt.ReadResourceRequestParams, Sequence[ReadResourceContents]],
    ) -> Sequence[ReadResourceContents]:
        """Intercept resource reads for logging and metrics."""
        uri = str(context.message.uri)

        if self.enable_logging:
            logger.log(self.log_level, f"Reading resource: {uri}")

        if self.enable_metrics:
            self.state.metrics.resource_reads += 1

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Resource read completed: {uri}")

        return result

    async def on_list_resources(
        self,
        context: MiddlewareContext[mt.ListResourcesRequest],
        call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        """Intercept list_resources for logging."""
        if self.enable_logging:
            logger.log(self.log_level, "Listing resources")

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Listed {len(result)} resources")

        return result

    async def on_list_resource_templates(
        self,
        context: MiddlewareContext[mt.ListResourceTemplatesRequest],
        call_next: CallNext[mt.ListResourceTemplatesRequest, Sequence[ResourceTemplate]],
    ) -> Sequence[ResourceTemplate]:
        """Intercept list_resource_templates for logging."""
        if self.enable_logging:
            logger.log(self.log_level, "Listing resource templates")

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Listed {len(result)} resource templates")

        return result

    async def on_list_prompts(
        self,
        context: MiddlewareContext[mt.ListPromptsRequest],
        call_next: CallNext[mt.ListPromptsRequest, Sequence[Prompt]],
    ) -> Sequence[Prompt]:
        """Intercept list_prompts for logging."""
        if self.enable_logging:
            logger.log(self.log_level, "Listing prompts")

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Listed {len(result)} prompts")

        return result

    async def on_get_prompt(
        self,
        context: MiddlewareContext[mt.GetPromptRequestParams],
        call_next: CallNext[mt.GetPromptRequestParams, mt.GetPromptResult],
    ) -> mt.GetPromptResult:
        """Intercept get_prompt for logging."""
        prompt_name = context.message.name

        if self.enable_logging:
            logger.log(self.log_level, f"Getting prompt: {prompt_name}")

        if self.enable_metrics:
            self.state.metrics.prompt_calls += 1

        result = await call_next(context)

        if self.enable_logging:
            logger.log(self.log_level, f"Prompt retrieved: {prompt_name}")

        return result

    def get_metrics(self) -> RequestMetrics:
        """Get current metrics.

        Returns:
            Current RequestMetrics snapshot
        """
        return self.state.metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.state.metrics = RequestMetrics()


# ============================================================================
# Response Cache Middleware (FastMCP v2.14+)
# ============================================================================


@dataclass
class CacheEntry:
    """Single cache entry with TTL support.

    Attributes:
        value: The cached value
        created_at: When the entry was created
        ttl_seconds: Time-to-live in seconds
        hit_count: Number of times this entry was accessed
    """

    value: Any
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)

    @property
    def age_seconds(self) -> float:
        """Get the age of this entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        expirations: Number of entries expired on access
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses


class ResponseCacheMiddleware:
    """Response caching middleware for expensive operations.

    Provides in-memory caching with TTL support, LRU-style eviction,
    and performance metrics tracking.

    Example:
        >>> cache = ResponseCacheMiddleware(max_entries=1000, default_ttl=300)
        >>> cache.set("routing:abc123", route_result, ttl=60)
        >>> cached = cache.get("routing:abc123")
        >>> if cached is not None:
        ...     return cached  # Cache hit
        >>> metrics = cache.get_metrics()
        >>> print(f"Hit rate: {metrics.hit_rate:.2%}")
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        default_ttl: int = 300,
    ) -> None:
        """Initialize the cache middleware.

        Args:
            max_entries: Maximum number of cache entries (default 1000)
            default_ttl: Default TTL in seconds (default 300 = 5 minutes)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()

    def make_key(self, namespace: str, **kwargs: Any) -> str:
        """Create a cache key from namespace and arguments.

        Args:
            namespace: Cache namespace (e.g., "routing", "recommend")
            **kwargs: Key-value pairs to include in the key

        Returns:
            32-character hex digest cache key
        """
        # Sort kwargs for consistent key generation
        sorted_items = sorted(kwargs.items())
        key_data = f"{namespace}:{sorted_items}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def get(self, key: str) -> Any | None:
        """Get a cached value if it exists and is not expired.

        Args:
            key: The cache key

        Returns:
            The cached value, or None if not found or expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._metrics.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._metrics.misses += 1
                self._metrics.expirations += 1
                logger.debug(f"Cache expired: {key[:16]}...")
                return None

            entry.hit_count += 1
            self._metrics.hits += 1
            logger.debug(f"Cache hit: {key[:16]}... (hits: {entry.hit_count})")
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Cache a value with optional TTL.

        Args:
            key: The cache key
            value: The value to cache
            ttl: TTL in seconds (uses default if not specified)
        """
        async with self._lock:
            # Evict entries if at capacity
            if len(self._cache) >= self._max_entries:
                self._evict_oldest_unlocked()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=datetime.now(),
                ttl_seconds=ttl if ttl is not None else self._default_ttl,
            )
            logger.debug(f"Cache set: {key[:16]}... (ttl: {ttl or self._default_ttl}s)")

    async def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: The cache key to invalidate

        Returns:
            True if the entry was found and removed, False otherwise
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache invalidated: {key[:16]}...")
                return True
            return False

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace.

        Note: This is O(n) and should be used sparingly.

        Args:
            namespace: The namespace prefix to invalidate

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            # Keys are hashed, so we can't efficiently filter by namespace
            # This is a limitation of the current design
            # In practice, this would require storing namespace metadata
            logger.warning(f"invalidate_namespace({namespace}) not efficient with hashed keys")
            return 0

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")
            return count

    def _evict_oldest_unlocked(self) -> int:
        """Evict oldest entries when at capacity.

        Note: This method must be called with self._lock held.

        Removes the oldest 10% of entries (at least 1).

        Returns:
            Number of entries evicted
        """
        if not self._cache:
            return 0

        # Sort entries by creation time, oldest first
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )

        # Remove oldest 10% (at least 1)
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]
            self._metrics.evictions += 1

        logger.debug(f"Cache evicted {to_remove} oldest entries")
        return to_remove

    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics.

        Returns:
            CacheMetrics with hit/miss/eviction counts
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics to zero."""
        self._metrics = CacheMetrics()

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    @property
    def max_size(self) -> int:
        """Maximum number of entries allowed."""
        return self._max_entries


def create_logging_middleware(
    mcp: FastMCP,
    *,
    log_level: int = logging.DEBUG,
) -> ReasoningMiddleware:
    """Create and register a logging-only middleware.

    Args:
        mcp: The FastMCP server instance
        log_level: Logging level for request/response logs

    Returns:
        Configured ReasoningMiddleware instance
    """
    middleware = ReasoningMiddleware(
        enable_logging=True,
        enable_metrics=False,
        log_level=log_level,
    )
    middleware.register(mcp)
    return middleware


def create_full_middleware(
    mcp: FastMCP,
    *,
    log_level: int = logging.DEBUG,
) -> ReasoningMiddleware:
    """Create and register full middleware with logging and metrics.

    Args:
        mcp: The FastMCP server instance
        log_level: Logging level for request/response logs

    Returns:
        Configured ReasoningMiddleware instance
    """
    middleware = ReasoningMiddleware(
        enable_logging=True,
        enable_metrics=True,
        log_level=log_level,
    )
    middleware.register(mcp)
    return middleware


# ============================================================================
# Rate Limiting Middleware (MCPaaS Feature)
# ============================================================================


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Implements the token bucket algorithm for rate limiting with burst support.

    Attributes:
        capacity: Maximum number of tokens (burst size)
        tokens: Current number of tokens available
        refill_rate: Tokens added per second
        last_refill: Last time tokens were refilled
    """

    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)

    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume (default 1.0)

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_retry_after(self) -> float:
        """Get the time in seconds until tokens will be available.

        Returns:
            Seconds until at least 1 token is available
        """
        if self.tokens >= 1.0:
            return 0.0
        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.refill_rate


@dataclass
class RateLimitInfo:
    """Rate limit information for a client.

    Tracks rate limiting state across multiple time windows.

    Attributes:
        minute_bucket: Token bucket for per-minute limit
        hour_bucket: Token bucket for per-hour limit
        request_count: Total request count for metrics
        first_seen: First time this client was seen
        last_seen: Last time this client made a request
    """

    minute_bucket: TokenBucket
    hour_bucket: TokenBucket
    request_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitMetrics:
    """Metrics for rate limiting.

    Attributes:
        requests_allowed: Number of requests that passed rate limiting
        requests_rejected: Number of requests rejected due to rate limiting
        unique_clients: Number of unique clients tracked
    """

    requests_allowed: int = 0
    requests_rejected: int = 0
    unique_clients: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of requests processed."""
        return self.requests_allowed + self.requests_rejected

    @property
    def rejection_rate(self) -> float:
        """Rate of rejected requests (0.0 to 1.0)."""
        total = self.total_requests
        return self.requests_rejected / total if total > 0 else 0.0


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
        message: Error message
    """

    def __init__(self, retry_after: float, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        self.message = message
        super().__init__(self.message)


class RateLimitMiddleware:
    """Rate limiting middleware using token bucket algorithm.

    Provides per-client rate limiting with configurable limits for different
    time windows (minute/hour). Supports burst traffic and bypass for certain keys.

    Example:
        >>> from reasoning_mcp.config import Settings
        >>> settings = Settings(
        ...     enable_rate_limiting=True,
        ...     rate_limit_requests_per_minute=60,
        ...     rate_limit_requests_per_hour=1000,
        ... )
        >>> rate_limiter = RateLimitMiddleware(settings)
        >>> rate_limiter.register(mcp)
    """

    def __init__(self, settings: Any) -> None:
        """Initialize rate limiting middleware.

        Args:
            settings: Settings instance with rate limiting configuration
        """
        self._settings = settings
        self._limits: dict[str, RateLimitInfo] = {}
        self._metrics = RateLimitMetrics()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()

    def _get_client_id(self, request: CallToolRequest) -> str:
        """Extract client identifier from request.

        Tries to identify the client using the configured header (API key),
        falling back to a generic identifier if not available.

        Args:
            request: The MCP request

        Returns:
            Client identifier string
        """
        # In MCP, we don't have direct access to HTTP headers from the request
        # For now, we'll use a simple approach based on request metadata
        # In a real implementation, this would be extracted from the transport layer

        # Try to get client ID from request params if available
        # For now, use a placeholder - this would need to be implemented
        # based on how FastMCP exposes client info
        return "default-client"

    def _is_bypass_key(self, client_id: str) -> bool:
        """Check if client has bypass permission.

        Args:
            client_id: Client identifier

        Returns:
            True if client bypasses rate limiting
        """
        return client_id in self._settings.rate_limit_bypass_keys

    def _get_or_create_limit_info_unlocked(self, client_id: str) -> RateLimitInfo:
        """Get or create rate limit info for a client.

        Note: This method must be called with self._lock held.

        Args:
            client_id: Client identifier

        Returns:
            RateLimitInfo for the client
        """
        if client_id not in self._limits:
            # Create token buckets for minute and hour windows
            minute_bucket = TokenBucket(
                capacity=float(self._settings.rate_limit_burst_size),
                tokens=float(self._settings.rate_limit_burst_size),
                refill_rate=self._settings.rate_limit_requests_per_minute / 60.0,
            )
            hour_bucket = TokenBucket(
                capacity=float(self._settings.rate_limit_requests_per_hour),
                tokens=float(self._settings.rate_limit_requests_per_hour),
                refill_rate=self._settings.rate_limit_requests_per_hour / 3600.0,
            )
            self._limits[client_id] = RateLimitInfo(
                minute_bucket=minute_bucket,
                hour_bucket=hour_bucket,
            )
            self._metrics.unique_clients = len(self._limits)

        return self._limits[client_id]

    async def _check_rate_limit(self, client_id: str) -> tuple[bool, float]:
        """Check if request should be rate limited.

        Args:
            client_id: Client identifier

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # Bypass check
        if self._is_bypass_key(client_id):
            self._metrics.requests_allowed += 1
            return (True, 0.0)

        async with self._lock:
            # Get or create limit info
            limit_info = self._get_or_create_limit_info_unlocked(client_id)
            limit_info.last_seen = datetime.now()
            limit_info.request_count += 1

            # Check minute bucket
            if not limit_info.minute_bucket.consume():
                retry_after = limit_info.minute_bucket.get_retry_after()
                self._metrics.requests_rejected += 1
                logger.warning(
                    f"Rate limit exceeded for client {client_id[:16]}... "
                    f"(per-minute limit, retry after {retry_after:.1f}s)"
                )
                return (False, retry_after)

            # Check hour bucket
            if not limit_info.hour_bucket.consume():
                retry_after = limit_info.hour_bucket.get_retry_after()
                self._metrics.requests_rejected += 1
                logger.warning(
                    f"Rate limit exceeded for client {client_id[:16]}... "
                    f"(per-hour limit, retry after {retry_after:.1f}s)"
                )
                return (False, retry_after)

            self._metrics.requests_allowed += 1
            return (True, 0.0)

    async def _cleanup_old_entries(self) -> None:
        """Background task to cleanup old rate limit entries."""
        while self._running:
            try:
                await asyncio.sleep(self._settings.rate_limit_cleanup_interval)

                async with self._lock:
                    # Remove entries that haven't been seen recently
                    cutoff = datetime.now() - timedelta(
                        seconds=self._settings.rate_limit_cleanup_interval * 2
                    )
                    to_remove = [
                        client_id
                        for client_id, info in self._limits.items()
                        if info.last_seen < cutoff
                    ]

                    for client_id in to_remove:
                        del self._limits[client_id]

                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} old rate limit entries")
                        self._metrics.unique_clients = len(self._limits)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")

    def register(self, mcp: FastMCP) -> None:
        """Register rate limiting middleware with the FastMCP server.

        Args:
            mcp: The FastMCP server instance
        """

        @mcp.on_tool_call  # type: ignore[untyped-decorator]
        async def rate_limit_tool_call(
            request: CallToolRequest,
            call_next: Callable[[], Any],
        ) -> MiddlewareResult[CallToolResult]:
            """Apply rate limiting to tool calls."""
            client_id = self._get_client_id(request)

            # Check rate limit (metrics are tracked inside this method)
            allowed, retry_after = await self._check_rate_limit(client_id)

            if not allowed:
                # Return 429 Too Many Requests error
                # Note: MCP doesn't have HTTP status codes, so we return an error result
                error_result = CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Rate limit exceeded. "
                                f"Please retry after {retry_after:.1f} seconds."
                            ),
                        )
                    ],
                    isError=True,
                )
                return error_result

            # Allow request
            result = await call_next()

            # Add rate limit headers to response metadata if available
            # Note: MCP doesn't have HTTP headers, but we can add metadata
            # This is a placeholder for future implementation
            return result

        logger.info("Rate limiting middleware registered")

        # Start cleanup task
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_old_entries())

    async def shutdown(self) -> None:
        """Shutdown the rate limiting middleware."""
        import contextlib

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    def get_metrics(self) -> RateLimitMetrics:
        """Get current rate limiting metrics.

        Returns:
            RateLimitMetrics snapshot
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset rate limiting metrics."""
        self._metrics = RateLimitMetrics()

    async def get_client_info(self, client_id: str) -> RateLimitInfo | None:
        """Get rate limit info for a specific client.

        Args:
            client_id: Client identifier

        Returns:
            RateLimitInfo if client is tracked, None otherwise
        """
        async with self._lock:
            return self._limits.get(client_id)

    async def clear_client(self, client_id: str) -> bool:
        """Clear rate limit state for a specific client.

        Args:
            client_id: Client identifier

        Returns:
            True if client was found and cleared, False otherwise
        """
        async with self._lock:
            if client_id in self._limits:
                del self._limits[client_id]
                self._metrics.unique_clients = len(self._limits)
                logger.info(f"Cleared rate limit state for client {client_id[:16]}...")
                return True
            return False

    async def clear_all(self) -> int:
        """Clear all rate limit state.

        Returns:
            Number of clients cleared
        """
        async with self._lock:
            count = len(self._limits)
            self._limits.clear()
            self._metrics.unique_clients = 0
            logger.info(f"Cleared all rate limit state ({count} clients)")
            return count


__all__ = [
    "ReasoningMiddleware",
    "RequestMetrics",
    "MiddlewareState",
    "CacheEntry",
    "CacheMetrics",
    "ResponseCacheMiddleware",
    "TokenBucket",
    "RateLimitInfo",
    "RateLimitMetrics",
    "RateLimitError",
    "RateLimitMiddleware",
    "create_logging_middleware",
    "create_full_middleware",
]
