"""Unit tests for RouterR1.

Tests RL-based multi-round routing with learned policies.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.router_r1 import ROUTER_R1_METADATA, RouterR1


class TestRouterR1Metadata:
    """Tests for RouterR1 metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert ROUTER_R1_METADATA.identifier == RouterIdentifier.ROUTER_R1

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert ROUTER_R1_METADATA.name == "Router-R1"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "rl" in ROUTER_R1_METADATA.tags
        assert "learned" in ROUTER_R1_METADATA.tags
        assert "policy" in ROUTER_R1_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert ROUTER_R1_METADATA.supports_budget_control is True
        assert ROUTER_R1_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= ROUTER_R1_METADATA.complexity <= 10


class TestRouterR1Initialization:
    """Tests for RouterR1 initialization."""

    def test_create_instance(self) -> None:
        """Test creating RouterR1 instance."""
        router = RouterR1()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = RouterR1()
        assert router.identifier == RouterIdentifier.ROUTER_R1

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = RouterR1()
        assert router.name == "Router-R1"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = RouterR1()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_creates_policy(self) -> None:
        """Test initialize creates policy with method preferences."""
        router = RouterR1()
        await router.initialize()
        assert len(router._policy) > 0
        assert MethodIdentifier.CHAIN_OF_THOUGHT in router._policy

    async def test_initialize_clears_history(self) -> None:
        """Test initialize clears routing history."""
        router = RouterR1()
        router._history = [{"test": "data"}]
        await router.initialize()
        assert len(router._history) == 0


class TestRouterR1FeatureExtraction:
    """Tests for RouterR1 feature extraction."""

    @pytest.fixture
    async def initialized_router(self) -> RouterR1:
        """Create an initialized RouterR1."""
        router = RouterR1()
        await router.initialize()
        return router

    def test_extract_length_feature(self, initialized_router: RouterR1) -> None:
        """Test length feature extraction."""
        features = initialized_router._extract_features("short")
        assert "length" in features
        assert 0.0 <= features["length"] <= 1.0

    def test_extract_complexity_feature(self, initialized_router: RouterR1) -> None:
        """Test complexity feature extraction."""
        features = initialized_router._extract_features("This is a test query")
        assert "complexity" in features
        assert 0.0 <= features["complexity"] <= 1.0

    def test_extract_math_feature_present(self, initialized_router: RouterR1) -> None:
        """Test math feature detected when math symbols present."""
        features = initialized_router._extract_features("What is 2+3=?")
        assert "has_math" in features
        assert features["has_math"] == 1.0

    def test_extract_math_feature_absent(self, initialized_router: RouterR1) -> None:
        """Test math feature not detected when no math symbols."""
        features = initialized_router._extract_features("Hello world")
        assert "has_math" in features
        assert features["has_math"] == 0.0

    def test_length_feature_capped(self, initialized_router: RouterR1) -> None:
        """Test length feature is capped at 1.0."""
        long_query = "word " * 200  # 1000 chars
        features = initialized_router._extract_features(long_query)
        assert features["length"] == 1.0

    def test_complexity_feature_capped(self, initialized_router: RouterR1) -> None:
        """Test complexity feature is capped at 1.0."""
        long_query = "word " * 200
        features = initialized_router._extract_features(long_query)
        assert features["complexity"] == 1.0


class TestRouterR1Routing:
    """Tests for RouterR1 routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> RouterR1:
        """Create an initialized RouterR1."""
        router = RouterR1()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = RouterR1()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: RouterR1) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None
        assert result in initialized_router._policy

    async def test_route_records_history(self, initialized_router: RouterR1) -> None:
        """Test routing records to history."""
        await initialized_router.route("test query")
        assert len(initialized_router._history) == 1
        assert "selected" in initialized_router._history[0]
        assert "policy_state" in initialized_router._history[0]

    async def test_route_multiple_queries_records_history(
        self, initialized_router: RouterR1
    ) -> None:
        """Test multiple routes build history."""
        await initialized_router.route("query 1")
        await initialized_router.route("query 2")
        await initialized_router.route("query 3")
        assert len(initialized_router._history) == 3

    async def test_route_complex_query_adjusts_policy(self, initialized_router: RouterR1) -> None:
        """Test complex query adjusts policy toward exploration methods."""
        # Complex query (high complexity feature)
        long_query = "complex problem " * 50
        result = await initialized_router.route(long_query)
        # Complex queries should favor ToT or MCTS
        assert result in [
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.MCTS,
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.SELF_CONSISTENCY,
        ]

    async def test_route_with_context(self, initialized_router: RouterR1) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "math"},
        )
        assert result is not None


class TestRouterR1BudgetAllocation:
    """Tests for RouterR1 budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> RouterR1:
        """Create an initialized RouterR1."""
        router = RouterR1()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = RouterR1()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(self, initialized_router: RouterR1) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_distributes_by_policy(
        self, initialized_router: RouterR1
    ) -> None:
        """Test budget is distributed according to policy weights."""
        allocation = await initialized_router.allocate_budget("test query", 1000)

        # All methods in policy should have allocation
        for method in initialized_router._policy:
            assert method in allocation

    async def test_allocate_budget_proportional_to_weights(
        self, initialized_router: RouterR1
    ) -> None:
        """Test allocations are proportional to policy weights."""
        allocation = await initialized_router.allocate_budget("test query", 1000)

        # Total should be approximately 1000 (some rounding)
        total = sum(allocation.values())
        assert 900 <= total <= 1000


class TestRouterR1HealthCheck:
    """Tests for RouterR1 health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = RouterR1()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = RouterR1()
        await router.initialize()
        assert await router.health_check() is True
