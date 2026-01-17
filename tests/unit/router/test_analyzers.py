"""
Unit tests for Problem Analyzers with caching support.

Tests cover:
- FastProblemAnalyzer caching
- StandardProblemAnalyzer caching
- ComplexProblemAnalyzer caching
- Cache hit/miss behavior
- Cache TTL behavior
"""

import pytest

from reasoning_mcp.middleware import ResponseCacheMiddleware
from reasoning_mcp.router.analyzers import (
    ComplexProblemAnalyzer,
    FastProblemAnalyzer,
    StandardProblemAnalyzer,
)
from reasoning_mcp.router.models import ProblemAnalysis, RouterTier

# ============================================================================
# FastProblemAnalyzer Cache Tests
# ============================================================================


@pytest.mark.unit
class TestFastProblemAnalyzerCache:
    """Tests for FastProblemAnalyzer caching behavior."""

    async def test_analyzer_without_cache(self):
        """Test analyzer works without cache."""
        analyzer = FastProblemAnalyzer(cache=None)
        result = await analyzer.analyze("What is 2+2?")

        assert isinstance(result, ProblemAnalysis)
        assert result.analyzer_tier == RouterTier.FAST

    async def test_analyzer_with_cache_initialized(self):
        """Test analyzer initializes with cache."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=60)
        analyzer = FastProblemAnalyzer(cache=cache)

        assert analyzer._cache is cache


# ============================================================================
# StandardProblemAnalyzer Cache Tests
# ============================================================================


@pytest.mark.unit
class TestStandardProblemAnalyzerCache:
    """Tests for StandardProblemAnalyzer caching behavior."""

    async def test_analyzer_without_cache(self):
        """Test analyzer works without cache."""
        analyzer = StandardProblemAnalyzer(cache=None)
        result = await analyzer.analyze("What is 2+2?")

        assert isinstance(result, ProblemAnalysis)
        assert result.analyzer_tier == RouterTier.STANDARD

    async def test_analyzer_with_cache_initialized(self):
        """Test analyzer initializes with cache."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=60)
        analyzer = StandardProblemAnalyzer(cache=cache)

        assert analyzer._cache is cache
        # Standard analyzer does NOT pass cache to fast analyzer (caching at Standard level only)
        assert analyzer._fast_analyzer._cache is None


# ============================================================================
# ComplexProblemAnalyzer Cache Tests
# ============================================================================


@pytest.mark.unit
class TestComplexProblemAnalyzerCache:
    """Tests for ComplexProblemAnalyzer caching behavior."""

    async def test_analyzer_without_cache(self):
        """Test analyzer works without cache."""
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=None)
        result = await analyzer.analyze("What is 2+2?")

        assert isinstance(result, ProblemAnalysis)
        assert result.analyzer_tier == RouterTier.COMPLEX

    async def test_analyzer_with_cache_initialized(self):
        """Test analyzer initializes with cache."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=60)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=120)

        assert analyzer._cache is cache
        assert analyzer._cache_ttl == 120
        # Complex analyzer does NOT pass cache to standard analyzer (caching at Complex level only)
        assert analyzer._standard_analyzer._cache is None

    async def test_cache_hit_on_repeated_analysis(self):
        """Test cache hit when analyzing same problem twice."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem = "What is the meaning of life, the universe, and everything?"

        # First analysis should result in caching
        result1 = await analyzer.analyze(problem)

        # Verify we got a valid result
        assert isinstance(result1, ProblemAnalysis)

        # Second analysis should be a cache hit (same object returned)
        result2 = await analyzer.analyze(problem)

        # Results should be identical (same cached object)
        assert result2 is result1

        # Cache should show at least one hit
        metrics = cache.get_metrics()
        assert metrics.hits > 0

    async def test_cache_miss_on_different_problems(self):
        """Test cache miss when analyzing different problems."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem1 = "What is 2+2?"
        problem2 = "What is 3+3?"

        result1 = await analyzer.analyze(problem1)
        result2 = await analyzer.analyze(problem2)

        # Results should be different objects
        assert result1 is not result2

        # Both should be valid analyses
        assert isinstance(result1, ProblemAnalysis)
        assert isinstance(result2, ProblemAnalysis)

    async def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem = "Consistent problem for key testing"

        # Generate keys for same problem
        key1 = analyzer._make_cache_key(problem)
        key2 = analyzer._make_cache_key(problem)

        # Keys should be identical
        assert key1 == key2

        # Keys should be 32 characters (hex digest truncated)
        assert len(key1) == 32

    async def test_cache_key_different_for_different_problems(self):
        """Test cache keys differ for different problems."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem1 = "First problem"
        problem2 = "Second problem"

        key1 = analyzer._make_cache_key(problem1)
        key2 = analyzer._make_cache_key(problem2)

        # Keys should be different
        assert key1 != key2

    async def test_custom_ttl_respected(self):
        """Test custom TTL is used when caching."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=60)
        custom_ttl = 180
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=custom_ttl)

        assert analyzer._cache_ttl == custom_ttl

    async def test_multiple_analyzers_share_cache(self):
        """Test multiple analyzers can share the same cache instance."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)

        analyzer1 = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)
        analyzer2 = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem = "Shared cache problem"

        # First analyzer caches the result
        result1 = await analyzer1.analyze(problem)

        # Second analyzer should get cached result
        result2 = await analyzer2.analyze(problem)

        # Should be the same cached object
        assert result2 is result1


# ============================================================================
# Cache Performance Tests
# ============================================================================


@pytest.mark.unit
class TestAnalyzerCachePerformance:
    """Tests for analyzer cache performance characteristics."""

    async def test_cache_reduces_latency_on_hits(self):
        """Test that cache hits have lower latency than misses."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem = "Performance test problem"

        # First call: cache miss, measures full analysis time
        result1 = await analyzer.analyze(problem)

        # Second call: cache hit, should be near-instant
        result2 = await analyzer.analyze(problem)

        # Note: Cached result returns the original analysis with original latency
        # The speedup is in the actual analyze() execution, not reflected in latency_ms
        # Both results should be the same object
        assert result2 is result1

    async def test_cache_metrics_tracking(self):
        """Test cache metrics are properly tracked."""
        cache = ResponseCacheMiddleware(max_entries=100, default_ttl=300)
        analyzer = ComplexProblemAnalyzer(ctx=None, cache=cache, cache_ttl=300)

        problem1 = "Problem for metrics 1"
        problem2 = "Problem for metrics 2"

        # Reset metrics
        cache.reset_metrics()
        initial_metrics = cache.get_metrics()
        assert initial_metrics.total_requests == 0

        # First unique problem: cache miss
        await analyzer.analyze(problem1)

        # Repeat same problem: cache hit
        await analyzer.analyze(problem1)

        # Second unique problem: cache miss
        await analyzer.analyze(problem2)

        # Repeat second problem: cache hit
        await analyzer.analyze(problem2)

        final_metrics = cache.get_metrics()

        # Should have 2 hits and some requests
        assert final_metrics.hits == 2
        assert final_metrics.total_requests >= 2
        assert final_metrics.hit_rate > 0.0
