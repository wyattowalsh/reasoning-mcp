"""Unit tests for MasRouter.

Tests multi-agent system routing for collaborative reasoning.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import RouterIdentifier
from reasoning_mcp.routers.mas_router import MAS_ROUTER_METADATA, MasRouter


class TestMasRouterMetadata:
    """Tests for MasRouter metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert MAS_ROUTER_METADATA.identifier == RouterIdentifier.MAS_ROUTER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert MAS_ROUTER_METADATA.name == "MasRouter"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "multi-agent" in MAS_ROUTER_METADATA.tags
        assert "collaborative" in MAS_ROUTER_METADATA.tags
        assert "orchestration" in MAS_ROUTER_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert MAS_ROUTER_METADATA.supports_budget_control is True
        assert MAS_ROUTER_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= MAS_ROUTER_METADATA.complexity <= 10


class TestMasRouterInitialization:
    """Tests for MasRouter initialization."""

    def test_create_instance(self) -> None:
        """Test creating MasRouter instance."""
        router = MasRouter()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = MasRouter()
        assert router.identifier == RouterIdentifier.MAS_ROUTER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = MasRouter()
        assert router.name == "MasRouter"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = MasRouter()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_creates_agents(self) -> None:
        """Test initialize creates agent pool."""
        router = MasRouter()
        await router.initialize()
        assert len(router._agents) > 0
        assert "analyzer" in router._agents
        assert "reasoner" in router._agents

    async def test_initialize_creates_capabilities(self) -> None:
        """Test initialize creates agent capabilities."""
        router = MasRouter()
        await router.initialize()
        assert len(router._agent_capabilities) > 0


class TestMasRouterAgentSelection:
    """Tests for MasRouter agent selection."""

    @pytest.fixture
    async def initialized_router(self) -> MasRouter:
        """Create an initialized MasRouter."""
        router = MasRouter()
        await router.initialize()
        return router

    def test_select_validator_for_verify_query(self, initialized_router: MasRouter) -> None:
        """Test validator is selected for verification queries."""
        agent = initialized_router._select_agent("Please verify this result")
        assert agent == "validator"

    def test_select_validator_for_check_query(self, initialized_router: MasRouter) -> None:
        """Test validator is selected for check queries."""
        agent = initialized_router._select_agent("Check if this is correct")
        assert agent == "validator"

    def test_select_reasoner_for_plan_query(self, initialized_router: MasRouter) -> None:
        """Test reasoner is selected for planning queries."""
        agent = initialized_router._select_agent("Plan how to solve this")
        assert agent == "reasoner"

    def test_select_reasoner_for_think_query(self, initialized_router: MasRouter) -> None:
        """Test reasoner is selected for thinking queries."""
        agent = initialized_router._select_agent("Think about this problem")
        assert agent == "reasoner"

    def test_select_synthesizer_for_combine_query(self, initialized_router: MasRouter) -> None:
        """Test synthesizer is selected for combination queries."""
        agent = initialized_router._select_agent("Combine these ideas")
        assert agent == "synthesizer"

    def test_select_synthesizer_for_conclude_query(self, initialized_router: MasRouter) -> None:
        """Test synthesizer is selected for conclusion queries."""
        agent = initialized_router._select_agent("Conclude from the evidence")
        assert agent == "synthesizer"

    def test_select_analyzer_as_default(self, initialized_router: MasRouter) -> None:
        """Test analyzer is selected as default."""
        agent = initialized_router._select_agent("Random query without keywords")
        assert agent == "analyzer"


class TestMasRouterRouting:
    """Tests for MasRouter routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> MasRouter:
        """Create an initialized MasRouter."""
        router = MasRouter()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = MasRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: MasRouter) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_verify_query(self, initialized_router: MasRouter) -> None:
        """Test routing for verification query."""
        result = await initialized_router.route("Verify this is correct")
        # Should use validator agent's method
        assert result == initialized_router._agents["validator"]["method"]

    async def test_route_with_context(self, initialized_router: MasRouter) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "math"},
        )
        assert result is not None


class TestMasRouterAgentRouting:
    """Tests for MasRouter multi-agent routing."""

    @pytest.fixture
    async def initialized_router(self) -> MasRouter:
        """Create an initialized MasRouter."""
        router = MasRouter()
        await router.initialize()
        return router

    async def test_route_agents_raises_when_not_initialized(self) -> None:
        """Test route_agents raises error when not initialized."""
        router = MasRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route_agents("test query")

    async def test_route_agents_returns_assignments(self, initialized_router: MasRouter) -> None:
        """Test route_agents returns role assignments."""
        assignments = await initialized_router.route_agents("test query")
        assert isinstance(assignments, dict)
        assert len(assignments) > 0

    async def test_route_agents_with_available_agents(self, initialized_router: MasRouter) -> None:
        """Test route_agents with specific available agents."""
        assignments = await initialized_router.route_agents(
            "test query",
            available_agents=["analyzer", "validator"],
        )
        assert "analyzer" in assignments
        assert "validator" in assignments

    async def test_route_agents_assigns_roles(self, initialized_router: MasRouter) -> None:
        """Test route_agents assigns appropriate roles."""
        assignments = await initialized_router.route_agents("test query")
        roles = list(assignments.values())
        # Should assign from predefined roles
        valid_roles = ["primary", "secondary", "validator", "synthesizer"]
        for role in roles:
            assert role in valid_roles


class TestMasRouterBudgetAllocation:
    """Tests for MasRouter budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> MasRouter:
        """Create an initialized MasRouter."""
        router = MasRouter()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = MasRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(self, initialized_router: MasRouter) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_primary_agent_gets_more(
        self, initialized_router: MasRouter
    ) -> None:
        """Test primary agent gets 40% of budget."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        # At least one allocation should be around 40%
        max_allocation = max(allocation.values())
        assert max_allocation >= 400  # 40% of 1000


class TestMasRouterHealthCheck:
    """Tests for MasRouter health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = MasRouter()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = MasRouter()
        await router.initialize()
        assert await router.health_check() is True
