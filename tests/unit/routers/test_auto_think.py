"""Unit tests for AutoThink router.

Tests adaptive CoT activation via classifier.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.auto_think import AUTO_THINK_METADATA, AutoThink


class TestAutoThinkMetadata:
    """Tests for AutoThink metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert AUTO_THINK_METADATA.identifier == RouterIdentifier.AUTO_THINK

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert AUTO_THINK_METADATA.name == "AutoThink"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "adaptive" in AUTO_THINK_METADATA.tags
        assert "classifier" in AUTO_THINK_METADATA.tags
        assert "cot-routing" in AUTO_THINK_METADATA.tags

    def test_metadata_supports_budget_control(self) -> None:
        """Test metadata indicates budget control support."""
        assert AUTO_THINK_METADATA.supports_budget_control is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= AUTO_THINK_METADATA.complexity <= 10


class TestAutoThinkInitialization:
    """Tests for AutoThink initialization."""

    def test_create_instance(self) -> None:
        """Test creating AutoThink instance."""
        router = AutoThink()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = AutoThink()
        assert router.identifier == RouterIdentifier.AUTO_THINK

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = AutoThink()
        assert router.name == "AutoThink"

    def test_description_property(self) -> None:
        """Test description property returns correct value."""
        router = AutoThink()
        assert "adaptive" in router.description.lower() or "cot" in router.description.lower()

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = AutoThink()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_resets_complexity(self) -> None:
        """Test initialize resets query complexity."""
        router = AutoThink()
        router._query_complexity = 0.9
        await router.initialize()
        assert router._query_complexity == 0.0


class TestAutoThinkRouting:
    """Tests for AutoThink routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> AutoThink:
        """Create an initialized AutoThink router."""
        router = AutoThink()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = AutoThink()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_simple_query_low_complexity(self, initialized_router: AutoThink) -> None:
        """Test routing simple query returns low complexity method."""
        result = await initialized_router.route("What is 2+2?")
        assert result == MethodIdentifier.ZERO_SHOT_COT
        assert initialized_router._query_complexity < 0.3

    async def test_route_medium_complexity_query(self, initialized_router: AutoThink) -> None:
        """Test routing medium complexity query returns CoT."""
        result = await initialized_router.route(
            "Calculate and explain the sum of first 10 prime numbers"
        )
        # Multiple complexity indicators: "calculate", "explain"
        assert result in (
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.TREE_OF_THOUGHTS,
        )

    async def test_route_high_complexity_query(self, initialized_router: AutoThink) -> None:
        """Test routing high complexity query returns ToT."""
        result = await initialized_router.route(
            "Calculate, solve, and analyze this problem. "
            "Explain why the solution works and compare it to alternatives. "
            "Evaluate the proof and prove correctness."
        )
        # Multiple complexity indicators should trigger ToT
        assert result == MethodIdentifier.TREE_OF_THOUGHTS
        assert initialized_router._query_complexity >= 0.6

    async def test_route_with_context(self, initialized_router: AutoThink) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "Simple question",
            context={"domain": "math"},
        )
        assert result is not None

    async def test_route_empty_query(self, initialized_router: AutoThink) -> None:
        """Test routing empty query."""
        result = await initialized_router.route("")
        assert result == MethodIdentifier.ZERO_SHOT_COT

    async def test_route_very_long_query(self, initialized_router: AutoThink) -> None:
        """Test routing very long query."""
        # Long query without complexity indicators
        result = await initialized_router.route("test " * 500)
        assert result is not None

    async def test_complexity_indicators_detected(self, initialized_router: AutoThink) -> None:
        """Test each complexity indicator is detected."""
        indicators = [
            "calculate",
            "compute",
            "solve",
            "analyze",
            "explain",
            "why",
            "how",
            "compare",
            "evaluate",
            "prove",
        ]

        for indicator in indicators:
            await initialized_router.route(indicator)
            assert initialized_router._query_complexity > 0.0


class TestAutoThinkBudgetAllocation:
    """Tests for AutoThink budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> AutoThink:
        """Create an initialized AutoThink router."""
        router = AutoThink()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = AutoThink()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(self, initialized_router: AutoThink) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("Simple query", 100)
        assert isinstance(allocation, dict)
        assert len(allocation) == 1
        assert 100 in allocation.values()

    async def test_allocate_budget_uses_routed_method(self, initialized_router: AutoThink) -> None:
        """Test budget allocation uses method from routing."""
        allocation = await initialized_router.allocate_budget("Calculate this", 200)

        # Should allocate to one method
        assert len(allocation) == 1
        assert list(allocation.values())[0] == 200


class TestAutoThinkHealthCheck:
    """Tests for AutoThink health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = AutoThink()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = AutoThink()
        await router.initialize()
        assert await router.health_check() is True
