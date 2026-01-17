"""Unit tests for GraphRouter.

Tests graph-based model routing using reasoning graphs.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import RouterIdentifier
from reasoning_mcp.routers.graph_router import GRAPH_ROUTER_METADATA, GraphRouter


class TestGraphRouterMetadata:
    """Tests for GraphRouter metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert GRAPH_ROUTER_METADATA.identifier == RouterIdentifier.GRAPH_ROUTER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert GRAPH_ROUTER_METADATA.name == "GraphRouter"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "graph" in GRAPH_ROUTER_METADATA.tags
        assert "structured" in GRAPH_ROUTER_METADATA.tags
        assert "routing" in GRAPH_ROUTER_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert GRAPH_ROUTER_METADATA.supports_budget_control is True
        assert GRAPH_ROUTER_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= GRAPH_ROUTER_METADATA.complexity <= 10


class TestGraphRouterInitialization:
    """Tests for GraphRouter initialization."""

    def test_create_instance(self) -> None:
        """Test creating GraphRouter instance."""
        router = GraphRouter()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = GraphRouter()
        assert router.identifier == RouterIdentifier.GRAPH_ROUTER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = GraphRouter()
        assert router.name == "GraphRouter"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = GraphRouter()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_creates_graph(self) -> None:
        """Test initialize creates reasoning graph."""
        router = GraphRouter()
        await router.initialize()
        assert len(router._graph) > 0
        assert "start" in router._graph

    async def test_initialize_creates_node_methods(self) -> None:
        """Test initialize creates node-to-method mapping."""
        router = GraphRouter()
        await router.initialize()
        assert len(router._node_methods) > 0
        assert "start" in router._node_methods


class TestGraphRouterGraphStructure:
    """Tests for GraphRouter graph structure."""

    @pytest.fixture
    async def initialized_router(self) -> GraphRouter:
        """Create an initialized GraphRouter."""
        router = GraphRouter()
        await router.initialize()
        return router

    def test_graph_has_start_node(self, initialized_router: GraphRouter) -> None:
        """Test graph has a start node."""
        assert "start" in initialized_router._graph

    def test_graph_nodes_have_successors(self, initialized_router: GraphRouter) -> None:
        """Test graph nodes have successor nodes."""
        for node, successors in initialized_router._graph.items():
            if node != "end":
                assert len(successors) > 0

    def test_node_methods_cover_graph(self, initialized_router: GraphRouter) -> None:
        """Test node methods cover graph nodes."""
        for node in initialized_router._graph:
            if node != "end":
                assert node in initialized_router._node_methods


class TestGraphRouterPathFinding:
    """Tests for GraphRouter path finding."""

    @pytest.fixture
    async def initialized_router(self) -> GraphRouter:
        """Create an initialized GraphRouter."""
        router = GraphRouter()
        await router.initialize()
        return router

    def test_short_query_path(self, initialized_router: GraphRouter) -> None:
        """Test path for short query."""
        path = initialized_router._find_optimal_path("Hi")
        assert len(path) > 0
        assert path[0] == "start"
        assert "end" in path

    def test_medium_query_path(self, initialized_router: GraphRouter) -> None:
        """Test path for medium length query."""
        query = "This is a medium length query " * 4  # ~120 chars
        path = initialized_router._find_optimal_path(query)
        assert len(path) > 0
        assert "analyze" in path

    def test_long_query_path(self, initialized_router: GraphRouter) -> None:
        """Test path for long query."""
        query = "This is a very long query " * 10  # ~250 chars
        path = initialized_router._find_optimal_path(query)
        assert len(path) > 0
        assert "decompose" in path


class TestGraphRouterRouting:
    """Tests for GraphRouter routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> GraphRouter:
        """Create an initialized GraphRouter."""
        router = GraphRouter()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = GraphRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: GraphRouter) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_short_query(self, initialized_router: GraphRouter) -> None:
        """Test routing for short query."""
        result = await initialized_router.route("Hi there")
        # Short query should start with analyze path
        assert result in initialized_router._node_methods.values()

    async def test_route_long_query(self, initialized_router: GraphRouter) -> None:
        """Test routing for long query."""
        query = "This is a complex problem " * 15  # > 200 chars
        result = await initialized_router.route(query)
        # Long query should use decomposition path
        assert result is not None

    async def test_route_with_context(self, initialized_router: GraphRouter) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "math"},
        )
        assert result is not None


class TestGraphRouterBudgetAllocation:
    """Tests for GraphRouter budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> GraphRouter:
        """Create an initialized GraphRouter."""
        router = GraphRouter()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = GraphRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(
        self, initialized_router: GraphRouter
    ) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_distributes_across_path(
        self, initialized_router: GraphRouter
    ) -> None:
        """Test budget is distributed across path nodes."""
        allocation = await initialized_router.allocate_budget("test query", 500)
        # Budget should be distributed to methods in the path
        total = sum(allocation.values())
        assert total > 0

    async def test_allocate_budget_long_query(self, initialized_router: GraphRouter) -> None:
        """Test budget allocation for long query with more nodes."""
        query = "Complex problem " * 20
        allocation = await initialized_router.allocate_budget(query, 1000)
        assert len(allocation) > 0


class TestGraphRouterHealthCheck:
    """Tests for GraphRouter health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = GraphRouter()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = GraphRouter()
        await router.initialize()
        assert await router.health_check() is True
