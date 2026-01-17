"""Performance regression tests.

These tests establish performance baselines and fail if performance degrades
beyond acceptable thresholds. Run with: uv run pytest tests/benchmarks/ -v
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


class TestStartupPerformance:
    """Tests for server startup time regression."""

    # Maximum allowed cold start time in seconds
    MAX_COLD_START_SECONDS = 2.0  # Target: <500ms, threshold: 2s for CI variance

    def test_import_time(self):
        """Test that core module imports complete within threshold."""
        import sys

        # Clear any cached imports
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("reasoning_mcp")]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start = time.perf_counter()

        # Import the main server module
        from reasoning_mcp import server  # noqa: F401

        elapsed = time.perf_counter() - start

        # Allow generous threshold for CI environments
        assert elapsed < self.MAX_COLD_START_SECONDS, (
            f"Cold start took {elapsed:.2f}s, exceeds {self.MAX_COLD_START_SECONDS}s threshold. "
            "Consider lazy loading for heavy modules."
        )

    def test_method_registry_initialization(self):
        """Test that method registry initializes within threshold."""
        MAX_REGISTRY_INIT_SECONDS = 1.0

        start = time.perf_counter()

        from reasoning_mcp.registry import MethodRegistry

        registry = MethodRegistry()
        # Just instantiation is sufficient; registration happens via register() calls

        elapsed = time.perf_counter() - start

        assert elapsed < MAX_REGISTRY_INIT_SECONDS, (
            f"Registry initialization took {elapsed:.2f}s, "
            f"exceeds {MAX_REGISTRY_INIT_SECONDS}s threshold."
        )


class TestLatencyRegression:
    """Tests for request latency regression."""

    @pytest.fixture
    def session(self):
        """Create a session for testing."""
        from reasoning_mcp.models import Session

        return Session()

    @pytest.mark.asyncio
    async def test_simple_method_execution_latency(self, session):
        """Test that simple method execution completes within threshold."""
        MAX_EXECUTION_MS = 100  # 100ms for simple operations

        from reasoning_mcp.methods.native.chain_of_thought import ChainOfThought

        method = ChainOfThought()
        await method.initialize()

        start = time.perf_counter()
        await method.execute(session, "What is 2+2?")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < MAX_EXECUTION_MS, (
            f"Method execution took {elapsed_ms:.1f}ms, "
            f"exceeds {MAX_EXECUTION_MS}ms threshold."
        )

    @pytest.mark.asyncio
    async def test_concurrent_method_execution(self, session):
        """Test that concurrent execution scales properly."""
        MAX_CONCURRENT_MS = 500  # 500ms for 10 concurrent operations
        CONCURRENT_COUNT = 10

        from reasoning_mcp.methods.native.chain_of_thought import ChainOfThought

        method = ChainOfThought()
        await method.initialize()

        async def execute_one():
            from reasoning_mcp.models import Session

            s = Session()
            return await method.execute(s, "Simple test")

        start = time.perf_counter()
        await asyncio.gather(*[execute_one() for _ in range(CONCURRENT_COUNT)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < MAX_CONCURRENT_MS, (
            f"{CONCURRENT_COUNT} concurrent executions took {elapsed_ms:.1f}ms, "
            f"exceeds {MAX_CONCURRENT_MS}ms threshold."
        )


class TestMemoryRegression:
    """Tests for memory usage regression."""

    def test_method_memory_footprint(self):
        """Test that method instances don't consume excessive memory."""
        import sys

        from reasoning_mcp.methods.native.chain_of_thought import ChainOfThought

        # Create multiple instances to measure per-instance overhead
        instances = [ChainOfThought() for _ in range(100)]

        # Get approximate size (this is a rough estimate)
        total_size = sum(sys.getsizeof(i) for i in instances)
        avg_size = total_size / len(instances)

        # Each instance should be < 10KB
        MAX_INSTANCE_BYTES = 10 * 1024
        assert avg_size < MAX_INSTANCE_BYTES, (
            f"Average method instance size is {avg_size:.0f} bytes, "
            f"exceeds {MAX_INSTANCE_BYTES} byte threshold."
        )

    def test_session_memory_growth(self):
        """Test that session memory doesn't grow unbounded."""
        import sys

        from reasoning_mcp.models import Session, ThoughtNode
        from reasoning_mcp.models.core import MethodIdentifier, ThoughtType

        session = Session()
        initial_size = sys.getsizeof(session)

        # Simulate adding thoughts (without actual content to isolate structure overhead)
        for i in range(100):
            thought = ThoughtNode(
                content=f"Thought {i}",
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                type=ThoughtType.REASONING,
            )
            session = session.add_thought(thought)

        final_size = sys.getsizeof(session)
        growth = final_size - initial_size

        # Growth should be reasonable (< 1MB for 100 thoughts)
        MAX_GROWTH_BYTES = 1024 * 1024
        assert growth < MAX_GROWTH_BYTES, (
            f"Session grew by {growth} bytes after 100 thoughts, "
            f"exceeds {MAX_GROWTH_BYTES} byte threshold."
        )


class TestThroughputRegression:
    """Tests for throughput regression."""

    @pytest.mark.asyncio
    async def test_method_throughput(self):
        """Test that method execution maintains minimum throughput."""
        MIN_OPS_PER_SECOND = 50  # At least 50 operations per second

        from reasoning_mcp.methods.native.chain_of_thought import ChainOfThought
        from reasoning_mcp.models import Session

        method = ChainOfThought()
        await method.initialize()

        count = 0
        start = time.perf_counter()
        duration = 1.0  # Run for 1 second

        while time.perf_counter() - start < duration:
            session = Session()
            await method.execute(session, "Test")
            count += 1

        ops_per_second = count / duration

        assert ops_per_second >= MIN_OPS_PER_SECOND, (
            f"Throughput is {ops_per_second:.1f} ops/s, "
            f"below {MIN_OPS_PER_SECOND} ops/s minimum."
        )
