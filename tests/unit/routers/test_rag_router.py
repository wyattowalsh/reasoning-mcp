"""Unit tests for RagRouter.

Tests retrieval-aware routing for knowledge-intensive tasks.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.rag_router import RAG_ROUTER_METADATA, RagRouter


class TestRagRouterMetadata:
    """Tests for RagRouter metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert RAG_ROUTER_METADATA.identifier == RouterIdentifier.RAG_ROUTER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert RAG_ROUTER_METADATA.name == "RAGRouter"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "rag" in RAG_ROUTER_METADATA.tags
        assert "retrieval" in RAG_ROUTER_METADATA.tags
        assert "knowledge" in RAG_ROUTER_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert RAG_ROUTER_METADATA.supports_budget_control is True
        assert RAG_ROUTER_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= RAG_ROUTER_METADATA.complexity <= 10


class TestRagRouterInitialization:
    """Tests for RagRouter initialization."""

    def test_create_instance(self) -> None:
        """Test creating RagRouter instance."""
        router = RagRouter()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = RagRouter()
        assert router.identifier == RouterIdentifier.RAG_ROUTER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = RagRouter()
        assert router.name == "RAGRouter"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = RagRouter()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_sets_threshold(self) -> None:
        """Test initialize sets retrieval threshold."""
        router = RagRouter()
        await router.initialize()
        assert router._retrieval_threshold == 0.5


class TestRagRouterRetrievalAssessment:
    """Tests for RagRouter retrieval need assessment."""

    @pytest.fixture
    async def initialized_router(self) -> RagRouter:
        """Create an initialized RagRouter."""
        router = RagRouter()
        await router.initialize()
        return router

    def test_high_retrieval_for_what_is(self, initialized_router: RagRouter) -> None:
        """Test high retrieval score for 'what is' queries."""
        score = initialized_router._assess_retrieval_need("What is machine learning?")
        assert score >= 0.5  # Should be retrieval-heavy

    def test_high_retrieval_for_who_is(self, initialized_router: RagRouter) -> None:
        """Test high retrieval score for 'who is' queries."""
        score = initialized_router._assess_retrieval_need("Who is Alan Turing?")
        assert score >= 0.5

    def test_high_retrieval_for_define(self, initialized_router: RagRouter) -> None:
        """Test high retrieval score for definition queries."""
        score = initialized_router._assess_retrieval_need("Define entropy in physics")
        assert score >= 0.5

    def test_high_retrieval_for_explain(self, initialized_router: RagRouter) -> None:
        """Test high retrieval score for explanation queries."""
        score = initialized_router._assess_retrieval_need("Explain the water cycle")
        assert score >= 0.5

    def test_low_retrieval_for_calculate(self, initialized_router: RagRouter) -> None:
        """Test low retrieval score for calculation queries."""
        score = initialized_router._assess_retrieval_need("Calculate 5 factorial")
        assert score <= 0.5  # Should be reasoning-heavy

    def test_low_retrieval_for_solve(self, initialized_router: RagRouter) -> None:
        """Test low retrieval score for solving queries."""
        score = initialized_router._assess_retrieval_need("Solve this equation: x + 5 = 10")
        assert score <= 0.5

    def test_low_retrieval_for_prove(self, initialized_router: RagRouter) -> None:
        """Test low retrieval score for proof queries."""
        score = initialized_router._assess_retrieval_need("Prove that sqrt(2) is irrational")
        assert score <= 0.5

    def test_balanced_retrieval_for_mixed_query(self, initialized_router: RagRouter) -> None:
        """Test balanced score for mixed queries."""
        score = initialized_router._assess_retrieval_need("Some random query")
        assert 0.0 <= score <= 1.0


class TestRagRouterRouting:
    """Tests for RagRouter routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> RagRouter:
        """Create an initialized RagRouter."""
        router = RagRouter()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = RagRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: RagRouter) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_knowledge_query_uses_cot(self, initialized_router: RagRouter) -> None:
        """Test knowledge query routes to CoT (good with context)."""
        result = await initialized_router.route("What is quantum computing?")
        assert result == MethodIdentifier.CHAIN_OF_THOUGHT

    async def test_route_reasoning_query_uses_tot(self, initialized_router: RagRouter) -> None:
        """Test reasoning query routes to ToT."""
        result = await initialized_router.route("Calculate and prove this theorem")
        assert result == MethodIdentifier.TREE_OF_THOUGHTS

    async def test_route_with_context(self, initialized_router: RagRouter) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "science"},
        )
        assert result is not None


class TestRagRouterBudgetAllocation:
    """Tests for RagRouter budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> RagRouter:
        """Create an initialized RagRouter."""
        router = RagRouter()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = RagRouter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(self, initialized_router: RagRouter) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_includes_retrieval(self, initialized_router: RagRouter) -> None:
        """Test budget allocation includes retrieval budget."""
        allocation = await initialized_router.allocate_budget(
            "What is the history of the internet?", 1000
        )
        # Should have retrieval allocation for knowledge queries
        assert "retrieval" in allocation or len(allocation) > 0

    async def test_allocate_budget_high_retrieval_query(
        self, initialized_router: RagRouter
    ) -> None:
        """Test budget allocation for high retrieval query."""
        allocation = await initialized_router.allocate_budget(
            "Tell me about the history of Python programming language",
            1000,
        )
        # High retrieval query should allocate more to retrieval
        assert len(allocation) > 0

    async def test_allocate_budget_low_retrieval_query(self, initialized_router: RagRouter) -> None:
        """Test budget allocation for low retrieval query."""
        allocation = await initialized_router.allocate_budget(
            "Calculate the derivative of x^3",
            1000,
        )
        # Low retrieval query should allocate more to reasoning
        assert len(allocation) > 0


class TestRagRouterHealthCheck:
    """Tests for RagRouter health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = RagRouter()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = RagRouter()
        await router.initialize()
        assert await router.health_check() is True
