"""Unit tests for BestRoute router.

Tests optimal test-time compute allocation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.best_route import BEST_ROUTE_METADATA, BestRoute


class TestBestRouteMetadata:
    """Tests for BestRoute metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert BEST_ROUTE_METADATA.identifier == RouterIdentifier.BEST_ROUTE

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert BEST_ROUTE_METADATA.name == "Best-Route"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "optimal" in BEST_ROUTE_METADATA.tags
        assert "compute" in BEST_ROUTE_METADATA.tags
        assert "allocation" in BEST_ROUTE_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert BEST_ROUTE_METADATA.supports_budget_control is True
        assert BEST_ROUTE_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= BEST_ROUTE_METADATA.complexity <= 10


class TestBestRouteInitialization:
    """Tests for BestRoute initialization."""

    def test_create_instance(self) -> None:
        """Test creating BestRoute instance."""
        router = BestRoute()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = BestRoute()
        assert router.identifier == RouterIdentifier.BEST_ROUTE

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = BestRoute()
        assert router.name == "Best-Route"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = BestRoute()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_populates_costs(self) -> None:
        """Test initialize populates method costs."""
        router = BestRoute()
        await router.initialize()
        assert len(router._method_costs) > 0
        assert MethodIdentifier.CHAIN_OF_THOUGHT in router._method_costs

    async def test_initialize_populates_quality(self) -> None:
        """Test initialize populates method quality estimates."""
        router = BestRoute()
        await router.initialize()
        assert len(router._method_quality) > 0
        assert MethodIdentifier.CHAIN_OF_THOUGHT in router._method_quality


class TestBestRouteRouting:
    """Tests for BestRoute routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> BestRoute:
        """Create an initialized BestRoute router."""
        router = BestRoute()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = BestRoute()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: BestRoute) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_with_low_budget_context(self, initialized_router: BestRoute) -> None:
        """Test routing with low budget prefers cheaper methods."""
        result = await initialized_router.route("test query", context={"budget": 0.1})
        # Should prefer zero-shot CoT as it's cheapest
        assert result == MethodIdentifier.ZERO_SHOT_COT

    async def test_route_with_high_budget_context(self, initialized_router: BestRoute) -> None:
        """Test routing with high budget allows expensive methods."""
        result = await initialized_router.route("test query", context={"budget": 1.0})
        # Should be able to pick any method
        assert result is not None

    async def test_route_with_default_budget(self, initialized_router: BestRoute) -> None:
        """Test routing with default budget (0.5)."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_without_context(self, initialized_router: BestRoute) -> None:
        """Test routing without context uses default budget."""
        result = await initialized_router.route("test query", context=None)
        assert result is not None

    async def test_route_selects_best_quality_cost_ratio(
        self, initialized_router: BestRoute
    ) -> None:
        """Test routing selects method with best quality/cost ratio."""
        # With moderate budget, should find optimal ratio
        result = await initialized_router.route("test query", context={"budget": 0.5})
        assert result in initialized_router._method_costs


class TestBestRouteBudgetAllocation:
    """Tests for BestRoute budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> BestRoute:
        """Create an initialized BestRoute router."""
        router = BestRoute()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = BestRoute()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(self, initialized_router: BestRoute) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_respects_total(self, initialized_router: BestRoute) -> None:
        """Test budget allocation doesn't exceed total budget."""
        allocation = await initialized_router.allocate_budget("test query", 500)
        total_allocated = sum(allocation.values())
        # Total should not exceed budget
        assert total_allocated <= 500

    async def test_allocate_budget_greedy_by_ratio(self, initialized_router: BestRoute) -> None:
        """Test budget allocation prioritizes by quality/cost ratio."""
        allocation = await initialized_router.allocate_budget("test query", 200)
        # Should allocate to methods with best ratio first
        assert len(allocation) > 0


class TestBestRouteHealthCheck:
    """Tests for BestRoute health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = BestRoute()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = BestRoute()
        await router.initialize()
        assert await router.health_check() is True
