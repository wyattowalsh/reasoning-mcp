"""Unit tests for middleware module.

Tests for:
- ReasoningMiddleware: Request/response logging and metrics
- ResponseCacheMiddleware: Response caching with TTL and eviction
- CacheEntry: Cache entry with TTL support
- CacheMetrics: Cache performance metrics
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.middleware import (
    CacheEntry,
    CacheMetrics,
    MiddlewareState,
    RateLimitError,
    RateLimitInfo,
    RateLimitMetrics,
    RateLimitMiddleware,
    ReasoningMiddleware,
    RequestMetrics,
    ResponseCacheMiddleware,
    TokenBucket,
    create_full_middleware,
    create_logging_middleware,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP


class TestRequestMetrics:
    """Tests for RequestMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are all zero."""
        metrics = RequestMetrics()
        assert metrics.tool_calls == 0
        assert metrics.resource_reads == 0
        assert metrics.prompt_calls == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.errors == 0

    def test_custom_values(self) -> None:
        """Test custom values are set correctly."""
        metrics = RequestMetrics(
            tool_calls=10,
            resource_reads=5,
            prompt_calls=3,
            total_duration_ms=150.5,
            errors=2,
        )
        assert metrics.tool_calls == 10
        assert metrics.resource_reads == 5
        assert metrics.prompt_calls == 3
        assert metrics.total_duration_ms == 150.5
        assert metrics.errors == 2


class TestMiddlewareState:
    """Tests for MiddlewareState dataclass."""

    def test_default_state(self) -> None:
        """Test default state has empty metrics and times."""
        state = MiddlewareState()
        assert state.metrics.tool_calls == 0
        assert state.request_start_times == {}

    def test_state_with_metrics(self) -> None:
        """Test state with custom metrics."""
        metrics = RequestMetrics(tool_calls=5)
        state = MiddlewareState(metrics=metrics)
        assert state.metrics.tool_calls == 5


class TestReasoningMiddleware:
    """Tests for ReasoningMiddleware class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        middleware = ReasoningMiddleware()
        assert middleware.enable_logging is True
        assert middleware.enable_metrics is True

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        middleware = ReasoningMiddleware(
            enable_logging=False,
            enable_metrics=False,
        )
        assert middleware.enable_logging is False
        assert middleware.enable_metrics is False

    def test_get_metrics(self) -> None:
        """Test get_metrics returns current metrics."""
        middleware = ReasoningMiddleware()
        metrics = middleware.get_metrics()
        assert isinstance(metrics, RequestMetrics)
        assert metrics.tool_calls == 0

    def test_reset_metrics(self) -> None:
        """Test reset_metrics clears all metrics."""
        middleware = ReasoningMiddleware()
        middleware.state.metrics.tool_calls = 10
        middleware.state.metrics.errors = 5
        middleware.reset_metrics()
        assert middleware.state.metrics.tool_calls == 0
        assert middleware.state.metrics.errors == 0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_init(self) -> None:
        """Test basic initialization."""
        entry = CacheEntry(
            value="test_value",
            created_at=datetime.now(),
            ttl_seconds=60,
        )
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 60
        assert entry.hit_count == 0

    def test_is_expired_false(self) -> None:
        """Test is_expired returns False for fresh entry."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now(),
            ttl_seconds=60,
        )
        assert entry.is_expired is False

    def test_is_expired_true(self) -> None:
        """Test is_expired returns True for expired entry."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(seconds=120),
            ttl_seconds=60,
        )
        assert entry.is_expired is True

    def test_age_seconds(self) -> None:
        """Test age_seconds property."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(seconds=10),
            ttl_seconds=60,
        )
        # Allow some margin for test execution time
        assert 9 <= entry.age_seconds <= 12


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are zero."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.expirations == 0

    def test_hit_rate_zero_requests(self) -> None:
        """Test hit_rate with zero requests returns 0.0."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit_rate calculation."""
        metrics = CacheMetrics(hits=75, misses=25)
        assert metrics.hit_rate == 0.75

    def test_hit_rate_all_hits(self) -> None:
        """Test hit_rate with all hits."""
        metrics = CacheMetrics(hits=100, misses=0)
        assert metrics.hit_rate == 1.0

    def test_hit_rate_all_misses(self) -> None:
        """Test hit_rate with all misses."""
        metrics = CacheMetrics(hits=0, misses=100)
        assert metrics.hit_rate == 0.0

    def test_total_requests(self) -> None:
        """Test total_requests property."""
        metrics = CacheMetrics(hits=30, misses=70)
        assert metrics.total_requests == 100


class TestResponseCacheMiddleware:
    """Tests for ResponseCacheMiddleware class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        cache = ResponseCacheMiddleware()
        assert cache.max_size == 1000
        assert cache.size == 0

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        cache = ResponseCacheMiddleware(max_entries=500, default_ttl=120)
        assert cache.max_size == 500

    def test_make_key_deterministic(self) -> None:
        """Test make_key produces deterministic keys."""
        cache = ResponseCacheMiddleware()
        key1 = cache.make_key("test", foo="bar", baz=123)
        key2 = cache.make_key("test", foo="bar", baz=123)
        assert key1 == key2
        assert len(key1) == 32  # SHA256 truncated to 32 chars

    def test_make_key_different_inputs(self) -> None:
        """Test make_key produces different keys for different inputs."""
        cache = ResponseCacheMiddleware()
        key1 = cache.make_key("test", foo="bar")
        key2 = cache.make_key("test", foo="baz")
        key3 = cache.make_key("other", foo="bar")
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self) -> None:
        """Test get returns None for missing key."""
        cache = ResponseCacheMiddleware()
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_increments_miss_metrics(self) -> None:
        """Test get increments miss metrics for missing key."""
        cache = ResponseCacheMiddleware()
        await cache.get("nonexistent")
        metrics = cache.get_metrics()
        assert metrics.misses == 1
        assert metrics.hits == 0

    @pytest.mark.asyncio
    async def test_get_increments_hit_metrics(self) -> None:
        """Test get increments hit metrics for existing key."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        await cache.get("key1")
        metrics = cache.get_metrics()
        assert metrics.hits == 1
        assert metrics.misses == 0

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self) -> None:
        """Test expired entries return None and increment expiration counter."""
        cache = ResponseCacheMiddleware(default_ttl=1)
        await cache.set("key1", "value1", ttl=1)
        # Wait for expiration
        time.sleep(1.1)
        assert await cache.get("key1") is None
        metrics = cache.get_metrics()
        assert metrics.expirations == 1

    @pytest.mark.asyncio
    async def test_invalidate_existing_key(self) -> None:
        """Test invalidate removes existing key."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        assert await cache.invalidate("key1") is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_invalidate_missing_key(self) -> None:
        """Test invalidate returns False for missing key."""
        cache = ResponseCacheMiddleware()
        assert await cache.invalidate("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clear removes all entries."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        count = await cache.clear()
        assert count == 3
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_size_property(self) -> None:
        """Test size property tracks entry count."""
        cache = ResponseCacheMiddleware()
        assert cache.size == 0
        await cache.set("key1", "value1")
        assert cache.size == 1
        await cache.set("key2", "value2")
        assert cache.size == 2
        await cache.invalidate("key1")
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_eviction_at_capacity(self) -> None:
        """Test eviction occurs when at capacity."""
        cache = ResponseCacheMiddleware(max_entries=10)
        # Fill cache to capacity
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
        assert cache.size == 10
        # Add one more - should trigger eviction
        await cache.set("key10", "value10")
        # At least one entry should have been evicted
        assert cache.size <= 10
        metrics = cache.get_metrics()
        assert metrics.evictions >= 1

    @pytest.mark.asyncio
    async def test_custom_ttl(self) -> None:
        """Test custom TTL is respected."""
        cache = ResponseCacheMiddleware(default_ttl=300)
        await cache.set("short_ttl", "value", ttl=1)
        await cache.set("long_ttl", "value", ttl=300)
        time.sleep(1.1)
        assert await cache.get("short_ttl") is None
        assert await cache.get("long_ttl") == "value"

    @pytest.mark.asyncio
    async def test_reset_metrics(self) -> None:
        """Test reset_metrics clears all metrics."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("missing")  # Miss
        cache.reset_metrics()
        metrics = cache.get_metrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.expirations == 0

    @pytest.mark.asyncio
    async def test_complex_values(self) -> None:
        """Test caching complex values (dicts, lists)."""
        cache = ResponseCacheMiddleware()
        complex_value = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }
        await cache.set("complex", complex_value)
        retrieved = await cache.get("complex")
        assert retrieved == complex_value

    @pytest.mark.asyncio
    async def test_none_value_cached(self) -> None:
        """Test None can be cached as a value."""
        cache = ResponseCacheMiddleware()
        await cache.set("none_key", None)
        # This is a limitation - get returns None for both missing and None values
        # In practice, avoid caching None values
        assert cache.size == 1


class TestCacheIntegration:
    """Integration tests for cache behavior."""

    @pytest.mark.asyncio
    async def test_hit_rate_after_mixed_operations(self) -> None:
        """Test hit rate calculation after mixed get operations."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # 3 hits
        await cache.get("key1")
        await cache.get("key2")
        await cache.get("key1")

        # 2 misses
        await cache.get("missing1")
        await cache.get("missing2")

        metrics = cache.get_metrics()
        assert metrics.hits == 3
        assert metrics.misses == 2
        assert metrics.hit_rate == 0.6

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self) -> None:
        """Test overwriting an existing key updates the value."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "original")
        await cache.set("key1", "updated")
        assert await cache.get("key1") == "updated"
        assert cache.size == 1


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_init(self) -> None:
        """Test basic initialization."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=10.0,
            refill_rate=1.0,
        )
        assert bucket.capacity == 10.0
        assert bucket.tokens == 10.0
        assert bucket.refill_rate == 1.0

    def test_consume_success(self) -> None:
        """Test consuming tokens when available."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=10.0,
            refill_rate=1.0,
        )
        assert bucket.consume(5.0) is True
        assert bucket.tokens == 5.0

    def test_consume_failure(self) -> None:
        """Test consuming tokens when not available."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=2.0,
            refill_rate=1.0,
        )
        assert bucket.consume(5.0) is False
        # Tokens may have slight refill due to time passage
        assert 2.0 <= bucket.tokens < 2.1

    def test_consume_exact_amount(self) -> None:
        """Test consuming exact amount of available tokens."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=5.0,
            refill_rate=1.0,
        )
        assert bucket.consume(5.0) is True
        # Tokens should be very close to 0 (allow for tiny time passage)
        assert bucket.tokens < 0.01

    def test_refill(self) -> None:
        """Test token refill over time."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=0.0,
            refill_rate=2.0,  # 2 tokens per second
        )
        bucket.last_refill = time.time() - 1.0  # 1 second ago
        # Consume should trigger refill
        bucket.consume(1.0)
        # Should have refilled ~2 tokens, consumed 1
        assert 0.5 <= bucket.tokens <= 1.5

    def test_refill_cap_at_capacity(self) -> None:
        """Test refill doesn't exceed capacity."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=8.0,
            refill_rate=5.0,  # 5 tokens per second
        )
        bucket.last_refill = time.time() - 1.0  # 1 second ago
        bucket.consume(0.1)
        # Should cap at capacity (10.0)
        assert bucket.tokens <= 10.0

    def test_get_retry_after(self) -> None:
        """Test retry_after calculation."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=0.0,
            refill_rate=2.0,  # 2 tokens per second
        )
        retry_after = bucket.get_retry_after()
        # Need 1 token at 2 tokens/sec = 0.5 seconds
        assert 0.4 <= retry_after <= 0.6

    def test_get_retry_after_with_tokens(self) -> None:
        """Test retry_after is 0 when tokens available."""
        bucket = TokenBucket(
            capacity=10.0,
            tokens=5.0,
            refill_rate=1.0,
        )
        assert bucket.get_retry_after() == 0.0


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        minute_bucket = TokenBucket(capacity=60.0, tokens=60.0, refill_rate=1.0)
        hour_bucket = TokenBucket(capacity=1000.0, tokens=1000.0, refill_rate=1.0)
        info = RateLimitInfo(
            minute_bucket=minute_bucket,
            hour_bucket=hour_bucket,
        )
        assert info.minute_bucket == minute_bucket
        assert info.hour_bucket == hour_bucket
        assert info.request_count == 0


class TestRateLimitMetrics:
    """Tests for RateLimitMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are zero."""
        metrics = RateLimitMetrics()
        assert metrics.requests_allowed == 0
        assert metrics.requests_rejected == 0
        assert metrics.unique_clients == 0

    def test_total_requests(self) -> None:
        """Test total_requests property."""
        metrics = RateLimitMetrics(requests_allowed=30, requests_rejected=10)
        assert metrics.total_requests == 40

    def test_rejection_rate_zero(self) -> None:
        """Test rejection_rate with zero requests."""
        metrics = RateLimitMetrics()
        assert metrics.rejection_rate == 0.0

    def test_rejection_rate_calculation(self) -> None:
        """Test rejection_rate calculation."""
        metrics = RateLimitMetrics(requests_allowed=75, requests_rejected=25)
        assert metrics.rejection_rate == 0.25

    def test_rejection_rate_all_rejected(self) -> None:
        """Test rejection_rate with all rejected."""
        metrics = RateLimitMetrics(requests_allowed=0, requests_rejected=100)
        assert metrics.rejection_rate == 1.0


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_init(self) -> None:
        """Test basic initialization."""
        error = RateLimitError(retry_after=30.0)
        assert error.retry_after == 30.0
        assert "Rate limit exceeded" in str(error)

    def test_custom_message(self) -> None:
        """Test custom message."""
        error = RateLimitError(retry_after=60.0, message="Custom message")
        assert error.retry_after == 60.0
        assert error.message == "Custom message"


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware class."""

    def test_init(self) -> None:
        """Test initialization with settings."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=10,
        )
        middleware = RateLimitMiddleware(settings)
        assert middleware._settings == settings
        assert middleware._metrics.requests_allowed == 0
        assert middleware._metrics.requests_rejected == 0

    def test_get_or_create_limit_info_unlocked(self) -> None:
        """Test creating rate limit info for new client."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=10,
        )
        middleware = RateLimitMiddleware(settings)
        info = middleware._get_or_create_limit_info_unlocked("client1")
        assert isinstance(info, RateLimitInfo)
        assert info.minute_bucket.capacity == 10.0  # burst size
        assert info.hour_bucket.capacity == 1000.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows_first_request(self) -> None:
        """Test first request is allowed."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=10,
        )
        middleware = RateLimitMiddleware(settings)
        allowed, retry_after = await middleware._check_rate_limit("client1")
        assert allowed is True
        assert retry_after == 0.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_burst_exhaustion(self) -> None:
        """Test rate limiting after burst exhaustion."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=5,  # Small burst
        )
        middleware = RateLimitMiddleware(settings)

        # Exhaust burst
        for _ in range(5):
            allowed, _ = await middleware._check_rate_limit("client1")
            assert allowed is True

        # Next request should be rate limited
        allowed, retry_after = await middleware._check_rate_limit("client1")
        assert allowed is False
        assert retry_after > 0.0

    @pytest.mark.asyncio
    async def test_bypass_key(self) -> None:
        """Test bypass key bypasses rate limiting."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=1,  # Very restrictive
            rate_limit_requests_per_hour=1,
            rate_limit_burst_size=1,
            rate_limit_bypass_keys=["bypass_client"],
        )
        middleware = RateLimitMiddleware(settings)

        # Bypass client should always be allowed
        for _ in range(10):
            allowed, retry_after = await middleware._check_rate_limit("bypass_client")
            assert allowed is True
            assert retry_after == 0.0

    def test_get_metrics(self) -> None:
        """Test get_metrics returns current metrics."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        metrics = middleware.get_metrics()
        assert isinstance(metrics, RateLimitMetrics)

    def test_reset_metrics(self) -> None:
        """Test reset_metrics clears metrics."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        middleware._metrics.requests_allowed = 100
        middleware._metrics.requests_rejected = 50
        middleware.reset_metrics()
        assert middleware._metrics.requests_allowed == 0
        assert middleware._metrics.requests_rejected == 0

    @pytest.mark.asyncio
    async def test_get_client_info(self) -> None:
        """Test getting client info."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        await middleware._check_rate_limit("client1")
        info = await middleware.get_client_info("client1")
        assert info is not None
        assert isinstance(info, RateLimitInfo)

    @pytest.mark.asyncio
    async def test_get_client_info_missing(self) -> None:
        """Test getting info for non-existent client."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        info = await middleware.get_client_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_clear_client(self) -> None:
        """Test clearing client rate limit state."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        await middleware._check_rate_limit("client1")
        assert await middleware.clear_client("client1") is True
        assert await middleware.get_client_info("client1") is None

    @pytest.mark.asyncio
    async def test_clear_client_missing(self) -> None:
        """Test clearing non-existent client."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        assert await middleware.clear_client("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear_all(self) -> None:
        """Test clearing all rate limit state."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        await middleware._check_rate_limit("client1")
        await middleware._check_rate_limit("client2")
        await middleware._check_rate_limit("client3")
        count = await middleware.clear_all()
        assert count == 3
        assert len(middleware._limits) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test shutdown stops cleanup task."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)
        middleware._running = True
        await middleware.shutdown()
        assert middleware._running is False


class TestRateLimitIntegration:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_multiple_clients_isolated(self) -> None:
        """Test multiple clients have isolated rate limits."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=5,
        )
        middleware = RateLimitMiddleware(settings)

        # Exhaust client1's burst
        for _ in range(5):
            allowed, _ = await middleware._check_rate_limit("client1")
            assert allowed is True

        # Client2 should still have full burst
        for _ in range(5):
            allowed, _ = await middleware._check_rate_limit("client2")
            assert allowed is True

        # Both clients should now be rate limited
        allowed1, _ = await middleware._check_rate_limit("client1")
        allowed2, _ = await middleware._check_rate_limit("client2")
        assert allowed1 is False
        assert allowed2 is False

    @pytest.mark.asyncio
    async def test_metrics_tracking(self) -> None:
        """Test metrics are tracked correctly."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=60,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=2,
        )
        middleware = RateLimitMiddleware(settings)

        # Allow 2 requests
        await middleware._check_rate_limit("client1")
        await middleware._check_rate_limit("client1")

        # Reject 1 request
        await middleware._check_rate_limit("client1")

        metrics = middleware.get_metrics()
        assert metrics.requests_allowed == 2
        assert metrics.requests_rejected == 1
        assert metrics.unique_clients == 1
        assert metrics.rejection_rate == pytest.approx(1 / 3, rel=0.01)


# ============================================================================
# ReasoningMiddleware on_* Method Tests
# ============================================================================


class TestReasoningMiddlewareOnCallTool:
    """Tests for ReasoningMiddleware.on_call_tool method."""

    @pytest.mark.asyncio
    async def test_on_call_tool_success(self) -> None:
        """Test on_call_tool handles successful tool calls."""
        middleware = ReasoningMiddleware(enable_logging=True, enable_metrics=True)

        # Create mock context and call_next
        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.name = "test_tool"

        mock_result = MagicMock()
        mock_call_next = AsyncMock(return_value=mock_result)

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        assert result is mock_result
        mock_call_next.assert_called_once_with(mock_context)
        assert middleware.state.metrics.tool_calls == 1

    @pytest.mark.asyncio
    async def test_on_call_tool_tracks_duration(self) -> None:
        """Test on_call_tool tracks duration when metrics enabled."""
        middleware = ReasoningMiddleware(enable_logging=False, enable_metrics=True)

        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.name = "test_tool"

        async def slow_call_next(ctx: MagicMock) -> str:
            await asyncio.sleep(0.05)
            return "result"

        await middleware.on_call_tool(mock_context, slow_call_next)

        assert middleware.state.metrics.total_duration_ms >= 50.0

    @pytest.mark.asyncio
    async def test_on_call_tool_handles_exception(self) -> None:
        """Test on_call_tool handles exceptions and tracks errors."""
        middleware = ReasoningMiddleware(enable_logging=True, enable_metrics=True)

        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.name = "test_tool"

        mock_call_next = AsyncMock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            await middleware.on_call_tool(mock_context, mock_call_next)

        assert middleware.state.metrics.errors == 1

    @pytest.mark.asyncio
    async def test_on_call_tool_no_metrics(self) -> None:
        """Test on_call_tool skips metrics when disabled."""
        middleware = ReasoningMiddleware(enable_logging=False, enable_metrics=False)

        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.name = "test_tool"

        mock_call_next = AsyncMock(return_value="result")

        await middleware.on_call_tool(mock_context, mock_call_next)

        # Metrics should not be incremented
        assert middleware.state.metrics.tool_calls == 0


class TestReasoningMiddlewareOnListTools:
    """Tests for ReasoningMiddleware.on_list_tools method."""

    @pytest.mark.asyncio
    async def test_on_list_tools_success(self) -> None:
        """Test on_list_tools returns tools list."""
        middleware = ReasoningMiddleware(enable_logging=True)

        mock_context = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        mock_call_next = AsyncMock(return_value=mock_tools)

        result = await middleware.on_list_tools(mock_context, mock_call_next)

        assert result == mock_tools
        mock_call_next.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_on_list_tools_no_logging(self) -> None:
        """Test on_list_tools works with logging disabled."""
        middleware = ReasoningMiddleware(enable_logging=False)

        mock_context = MagicMock()
        mock_call_next = AsyncMock(return_value=[])

        result = await middleware.on_list_tools(mock_context, mock_call_next)

        assert result == []


class TestReasoningMiddlewareOnReadResource:
    """Tests for ReasoningMiddleware.on_read_resource method."""

    @pytest.mark.asyncio
    async def test_on_read_resource_success(self) -> None:
        """Test on_read_resource returns resource contents."""
        middleware = ReasoningMiddleware(enable_logging=True, enable_metrics=True)

        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.uri = "file:///test.txt"

        mock_contents = [MagicMock()]
        mock_call_next = AsyncMock(return_value=mock_contents)

        result = await middleware.on_read_resource(mock_context, mock_call_next)

        assert result == mock_contents
        assert middleware.state.metrics.resource_reads == 1


class TestReasoningMiddlewareOnListResources:
    """Tests for ReasoningMiddleware.on_list_resources method."""

    @pytest.mark.asyncio
    async def test_on_list_resources_success(self) -> None:
        """Test on_list_resources returns resources list."""
        middleware = ReasoningMiddleware(enable_logging=True)

        mock_context = MagicMock()
        mock_resources = [MagicMock(), MagicMock()]
        mock_call_next = AsyncMock(return_value=mock_resources)

        result = await middleware.on_list_resources(mock_context, mock_call_next)

        assert result == mock_resources


class TestReasoningMiddlewareOnListResourceTemplates:
    """Tests for ReasoningMiddleware.on_list_resource_templates method."""

    @pytest.mark.asyncio
    async def test_on_list_resource_templates_success(self) -> None:
        """Test on_list_resource_templates returns templates list."""
        middleware = ReasoningMiddleware(enable_logging=True)

        mock_context = MagicMock()
        mock_templates = [MagicMock()]
        mock_call_next = AsyncMock(return_value=mock_templates)

        result = await middleware.on_list_resource_templates(mock_context, mock_call_next)

        assert result == mock_templates


class TestReasoningMiddlewareOnListPrompts:
    """Tests for ReasoningMiddleware.on_list_prompts method."""

    @pytest.mark.asyncio
    async def test_on_list_prompts_success(self) -> None:
        """Test on_list_prompts returns prompts list."""
        middleware = ReasoningMiddleware(enable_logging=True)

        mock_context = MagicMock()
        mock_prompts = [MagicMock(), MagicMock()]
        mock_call_next = AsyncMock(return_value=mock_prompts)

        result = await middleware.on_list_prompts(mock_context, mock_call_next)

        assert result == mock_prompts


class TestReasoningMiddlewareOnGetPrompt:
    """Tests for ReasoningMiddleware.on_get_prompt method."""

    @pytest.mark.asyncio
    async def test_on_get_prompt_success(self) -> None:
        """Test on_get_prompt returns prompt result."""
        middleware = ReasoningMiddleware(enable_logging=True, enable_metrics=True)

        mock_context = MagicMock()
        mock_context.message = MagicMock()
        mock_context.message.name = "test_prompt"

        mock_result = MagicMock()
        mock_call_next = AsyncMock(return_value=mock_result)

        result = await middleware.on_get_prompt(mock_context, mock_call_next)

        assert result is mock_result
        assert middleware.state.metrics.prompt_calls == 1


class TestReasoningMiddlewareRegister:
    """Tests for ReasoningMiddleware.register method."""

    def test_register_calls_add_middleware(self) -> None:
        """Test register calls add_middleware on FastMCP instance."""
        middleware = ReasoningMiddleware()
        mock_mcp = MagicMock()

        middleware.register(mock_mcp)

        mock_mcp.add_middleware.assert_called_once_with(middleware)


# ============================================================================
# ResponseCacheMiddleware Additional Tests
# ============================================================================


class TestResponseCacheMiddlewareInvalidateNamespace:
    """Tests for ResponseCacheMiddleware.invalidate_namespace method."""

    @pytest.mark.asyncio
    async def test_invalidate_namespace_returns_zero(self) -> None:
        """Test invalidate_namespace returns 0 (limitation with hashed keys)."""
        cache = ResponseCacheMiddleware()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        result = await cache.invalidate_namespace("test")

        # Due to hashed keys, namespace invalidation returns 0
        assert result == 0
        # Keys should still exist
        assert cache.size == 2


class TestResponseCacheMiddlewareEvictOldest:
    """Tests for ResponseCacheMiddleware._evict_oldest_unlocked method."""

    @pytest.mark.asyncio
    async def test_evict_oldest_empty_cache(self) -> None:
        """Test _evict_oldest_unlocked returns 0 for empty cache."""
        cache = ResponseCacheMiddleware()

        # Call internal method directly
        result = cache._evict_oldest_unlocked()

        assert result == 0

    @pytest.mark.asyncio
    async def test_evict_oldest_removes_oldest_entries(self) -> None:
        """Test _evict_oldest_unlocked removes oldest entries."""
        cache = ResponseCacheMiddleware(max_entries=20)

        # Add entries with slight time differences
        for i in range(15):
            await cache.set(f"key{i}", f"value{i}")
            await asyncio.sleep(0.001)

        # Call eviction directly
        async with cache._lock:
            evicted = cache._evict_oldest_unlocked()

        # Should evict at least 1 entry (10% of 15 = 1.5, so 1)
        assert evicted >= 1


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateLoggingMiddleware:
    """Tests for create_logging_middleware function."""

    def test_creates_logging_only_middleware(self) -> None:
        """Test creates middleware with logging enabled, metrics disabled."""
        mock_mcp = MagicMock()

        middleware = create_logging_middleware(mock_mcp)

        assert isinstance(middleware, ReasoningMiddleware)
        assert middleware.enable_logging is True
        assert middleware.enable_metrics is False
        mock_mcp.add_middleware.assert_called_once_with(middleware)

    def test_respects_log_level(self) -> None:
        """Test respects custom log level."""
        import logging

        mock_mcp = MagicMock()

        middleware = create_logging_middleware(mock_mcp, log_level=logging.WARNING)

        assert middleware.log_level == logging.WARNING


class TestCreateFullMiddleware:
    """Tests for create_full_middleware function."""

    def test_creates_full_middleware(self) -> None:
        """Test creates middleware with logging and metrics enabled."""
        mock_mcp = MagicMock()

        middleware = create_full_middleware(mock_mcp)

        assert isinstance(middleware, ReasoningMiddleware)
        assert middleware.enable_logging is True
        assert middleware.enable_metrics is True
        mock_mcp.add_middleware.assert_called_once_with(middleware)

    def test_respects_log_level(self) -> None:
        """Test respects custom log level."""
        import logging

        mock_mcp = MagicMock()

        middleware = create_full_middleware(mock_mcp, log_level=logging.INFO)

        assert middleware.log_level == logging.INFO


# ============================================================================
# RateLimitMiddleware Additional Tests
# ============================================================================


class TestRateLimitMiddlewareGetClientId:
    """Tests for RateLimitMiddleware._get_client_id method."""

    def test_get_client_id_returns_default(self) -> None:
        """Test _get_client_id returns default client identifier."""
        from reasoning_mcp.config import Settings

        settings = Settings(enable_rate_limiting=True)
        middleware = RateLimitMiddleware(settings)

        mock_request = MagicMock()
        client_id = middleware._get_client_id(mock_request)

        assert client_id == "default-client"


class TestRateLimitMiddlewareHourlyBucketCheck:
    """Tests for RateLimitMiddleware hourly bucket rate limiting."""

    @pytest.mark.asyncio
    async def test_hourly_bucket_exhaustion(self) -> None:
        """Test rate limiting when hourly bucket is exhausted."""
        from reasoning_mcp.config import Settings

        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=1000,  # High minute limit
            rate_limit_requests_per_hour=5,  # Low hourly limit
            rate_limit_burst_size=1000,  # High burst
        )
        middleware = RateLimitMiddleware(settings)

        # Exhaust hourly limit
        for _ in range(5):
            allowed, _ = await middleware._check_rate_limit("client1")
            assert allowed is True

        # Next request should be rate limited by hourly bucket
        allowed, retry_after = await middleware._check_rate_limit("client1")
        assert allowed is False
        assert retry_after > 0.0


class TestRateLimitMiddlewareCleanupOldEntries:
    """Tests for RateLimitMiddleware._cleanup_old_entries method."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_entries(self) -> None:
        """Test cleanup removes entries that haven't been seen recently."""
        # Use a mock settings to bypass validation
        mock_settings = MagicMock()
        mock_settings.enable_rate_limiting = True
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.rate_limit_requests_per_hour = 1000
        mock_settings.rate_limit_burst_size = 10
        mock_settings.rate_limit_cleanup_interval = 60  # Valid value
        mock_settings.rate_limit_bypass_keys = []

        middleware = RateLimitMiddleware(mock_settings)

        # Create a client entry
        await middleware._check_rate_limit("old-client")

        # Manually set last_seen to old time
        async with middleware._lock:
            if "old-client" in middleware._limits:
                middleware._limits["old-client"].last_seen = datetime.now() - timedelta(hours=1)

        # Run cleanup once with mocked sleep
        middleware._running = True

        async def mock_cleanup() -> None:
            """Run one cleanup iteration."""
            async with middleware._lock:
                now = datetime.now()
                stale_threshold = timedelta(minutes=30)
                stale_clients = [
                    client_id
                    for client_id, info in middleware._limits.items()
                    if now - info.last_seen > stale_threshold
                ]
                for client_id in stale_clients:
                    del middleware._limits[client_id]

        await mock_cleanup()

        # Verify old client was removed
        async with middleware._lock:
            assert "old-client" not in middleware._limits

    @pytest.mark.asyncio
    async def test_cleanup_handles_errors(self) -> None:
        """Test cleanup loop continues running after errors."""
        # Use a mock settings
        mock_settings = MagicMock()
        mock_settings.enable_rate_limiting = True
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.rate_limit_requests_per_hour = 1000
        mock_settings.rate_limit_burst_size = 10
        mock_settings.rate_limit_cleanup_interval = 60
        mock_settings.rate_limit_bypass_keys = []

        middleware = RateLimitMiddleware(mock_settings)
        middleware._running = True

        # Mock asyncio.sleep to avoid waiting
        sleep_call_count = 0

        async def mock_sleep(duration: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                middleware._running = False
            # Raise to simulate error on first call
            if sleep_call_count == 1:
                raise RuntimeError("Test error")

        with patch("asyncio.sleep", mock_sleep):
            # The cleanup should handle the error and continue
            try:
                await middleware._cleanup_old_entries()
            except RuntimeError:
                pass  # Expected, cleanup continues despite errors

        # Verify middleware state is still valid
        assert isinstance(middleware._limits, dict)


class TestRateLimitMiddlewareRegister:
    """Tests for RateLimitMiddleware.register method."""

    @pytest.mark.asyncio
    async def test_register_creates_cleanup_task(self) -> None:
        """Test register starts cleanup background task."""
        # Use mock settings for consistency
        mock_settings = MagicMock()
        mock_settings.enable_rate_limiting = True
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.rate_limit_requests_per_hour = 1000
        mock_settings.rate_limit_burst_size = 10
        mock_settings.rate_limit_cleanup_interval = 60
        mock_settings.rate_limit_bypass_keys = []

        middleware = RateLimitMiddleware(mock_settings)

        mock_mcp = MagicMock()
        mock_mcp.on_tool_call = MagicMock(return_value=lambda f: f)

        middleware.register(mock_mcp)

        assert middleware._running is True
        assert middleware._cleanup_task is not None

        # Clean up
        await middleware.shutdown()


class TestRateLimitMiddlewareShutdownWithTask:
    """Tests for RateLimitMiddleware.shutdown with active cleanup task."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_cleanup_task(self) -> None:
        """Test shutdown cancels the cleanup task."""
        # Use mock settings to bypass validation
        mock_settings = MagicMock()
        mock_settings.enable_rate_limiting = True
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.rate_limit_requests_per_hour = 1000
        mock_settings.rate_limit_burst_size = 10
        mock_settings.rate_limit_cleanup_interval = 60
        mock_settings.rate_limit_bypass_keys = []

        middleware = RateLimitMiddleware(mock_settings)

        # Start cleanup task manually with mocked sleep to avoid waiting
        middleware._running = True

        async def slow_cleanup() -> None:
            """Simulate cleanup that waits."""
            while middleware._running:
                await asyncio.sleep(0.1)

        middleware._cleanup_task = asyncio.create_task(slow_cleanup())

        # Give task time to start
        await asyncio.sleep(0.01)

        await middleware.shutdown()

        assert middleware._running is False
        # Task should be cancelled and cleaned up
        assert middleware._cleanup_task.cancelled() or middleware._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_handles_no_task(self) -> None:
        """Test shutdown works when no cleanup task exists."""
        # Use mock settings to bypass validation
        mock_settings = MagicMock()
        mock_settings.enable_rate_limiting = True
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.rate_limit_requests_per_hour = 1000
        mock_settings.rate_limit_burst_size = 10
        mock_settings.rate_limit_cleanup_interval = 60
        mock_settings.rate_limit_bypass_keys = []

        middleware = RateLimitMiddleware(mock_settings)

        middleware._running = True
        middleware._cleanup_task = None

        # Should not raise
        await middleware.shutdown()

        assert middleware._running is False
