"""Unit tests for SelfBudgeter router.

Tests token allocation optimization based on problem difficulty.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.self_budgeter import SELF_BUDGETER_METADATA, SelfBudgeter


class TestSelfBudgeterMetadata:
    """Tests for SelfBudgeter metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert SELF_BUDGETER_METADATA.identifier == RouterIdentifier.SELF_BUDGETER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert SELF_BUDGETER_METADATA.name == "SelfBudgeter"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "budget" in SELF_BUDGETER_METADATA.tags
        assert "allocation" in SELF_BUDGETER_METADATA.tags
        assert "difficulty" in SELF_BUDGETER_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert SELF_BUDGETER_METADATA.supports_budget_control is True
        assert SELF_BUDGETER_METADATA.supports_multi_model is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= SELF_BUDGETER_METADATA.complexity <= 10


class TestSelfBudgeterInitialization:
    """Tests for SelfBudgeter initialization."""

    def test_create_instance(self) -> None:
        """Test creating SelfBudgeter instance."""
        router = SelfBudgeter()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = SelfBudgeter()
        assert router.identifier == RouterIdentifier.SELF_BUDGETER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = SelfBudgeter()
        assert router.name == "SelfBudgeter"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = SelfBudgeter()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_resets_difficulty(self) -> None:
        """Test initialize resets difficulty estimate."""
        router = SelfBudgeter()
        router._difficulty_estimate = 0.9
        await router.initialize()
        assert router._difficulty_estimate == 0.0


class TestSelfBudgeterDifficultyEstimation:
    """Tests for SelfBudgeter difficulty estimation."""

    @pytest.fixture
    async def initialized_router(self) -> SelfBudgeter:
        """Create an initialized SelfBudgeter."""
        router = SelfBudgeter()
        await router.initialize()
        return router

    def test_short_query_low_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test short query has low difficulty."""
        difficulty = initialized_router._estimate_difficulty("Hi")
        assert difficulty < 0.3

    def test_long_query_higher_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test long query has higher difficulty."""
        long_query = "word " * 100  # 500 chars
        difficulty = initialized_router._estimate_difficulty(long_query)
        assert difficulty > 0.5

    def test_prove_keyword_increases_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test 'prove' keyword increases difficulty."""
        base = initialized_router._estimate_difficulty("Show this")
        with_prove = initialized_router._estimate_difficulty("Prove this")
        assert with_prove > base

    def test_derive_keyword_increases_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test 'derive' keyword increases difficulty."""
        base = initialized_router._estimate_difficulty("Show the formula")
        with_derive = initialized_router._estimate_difficulty("Derive the formula")
        assert with_derive > base

    def test_optimize_keyword_increases_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test 'optimize' keyword increases difficulty."""
        base = initialized_router._estimate_difficulty("Solve this")
        with_optimize = initialized_router._estimate_difficulty("Optimize this")
        assert with_optimize > base

    def test_complex_keyword_increases_difficulty(self, initialized_router: SelfBudgeter) -> None:
        """Test 'complex' keyword increases difficulty."""
        base = initialized_router._estimate_difficulty("Solve the problem")
        with_complex = initialized_router._estimate_difficulty("Solve the complex problem")
        assert with_complex > base

    def test_multi_step_keyword_increases_difficulty(
        self, initialized_router: SelfBudgeter
    ) -> None:
        """Test 'multi-step' keyword increases difficulty."""
        base = initialized_router._estimate_difficulty("Solve the problem")
        with_multistep = initialized_router._estimate_difficulty("Solve the multi-step problem")
        assert with_multistep > base

    def test_difficulty_capped_at_one(self, initialized_router: SelfBudgeter) -> None:
        """Test difficulty is capped at 1.0."""
        # Use many hard keywords and long length
        hard_query = "prove derive optimize complex multi-step " * 20
        difficulty = initialized_router._estimate_difficulty(hard_query)
        assert difficulty == 1.0


class TestSelfBudgeterRouting:
    """Tests for SelfBudgeter routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> SelfBudgeter:
        """Create an initialized SelfBudgeter."""
        router = SelfBudgeter()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = SelfBudgeter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: SelfBudgeter) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_easy_query_returns_cot(self, initialized_router: SelfBudgeter) -> None:
        """Test easy query routes to CoT."""
        result = await initialized_router.route("Hi")
        assert result == MethodIdentifier.CHAIN_OF_THOUGHT
        assert initialized_router._difficulty_estimate < 0.3

    async def test_route_medium_query_returns_self_consistency(
        self, initialized_router: SelfBudgeter
    ) -> None:
        """Test medium difficulty query routes to self-consistency."""
        # Medium length, no hard keywords
        medium_query = "Explain the process " * 10  # ~200 chars
        result = await initialized_router.route(medium_query)
        assert result == MethodIdentifier.SELF_CONSISTENCY
        assert 0.3 <= initialized_router._difficulty_estimate < 0.7

    async def test_route_hard_query_returns_mcts(self, initialized_router: SelfBudgeter) -> None:
        """Test hard query routes to MCTS."""
        hard_query = "Prove and derive the complex multi-step optimization"
        result = await initialized_router.route(hard_query)
        assert result == MethodIdentifier.MCTS
        assert initialized_router._difficulty_estimate >= 0.7

    async def test_route_with_context(self, initialized_router: SelfBudgeter) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "math"},
        )
        assert result is not None


class TestSelfBudgeterBudgetAllocation:
    """Tests for SelfBudgeter budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> SelfBudgeter:
        """Create an initialized SelfBudgeter."""
        router = SelfBudgeter()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = SelfBudgeter()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(
        self, initialized_router: SelfBudgeter
    ) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) > 0

    async def test_allocate_budget_easy_single_method(
        self, initialized_router: SelfBudgeter
    ) -> None:
        """Test easy query allocates to single method."""
        allocation = await initialized_router.allocate_budget("Hi", 1000)
        assert len(allocation) == 1
        assert MethodIdentifier.CHAIN_OF_THOUGHT in allocation
        assert allocation[MethodIdentifier.CHAIN_OF_THOUGHT] == 1000

    async def test_allocate_budget_medium_split(self, initialized_router: SelfBudgeter) -> None:
        """Test medium difficulty splits between main and verification."""
        medium_query = "Explain this process " * 10
        allocation = await initialized_router.allocate_budget(medium_query, 1000)
        assert len(allocation) == 2
        assert MethodIdentifier.CHAIN_OF_THOUGHT in allocation
        assert MethodIdentifier.SELF_VERIFICATION in allocation
        # Main method gets 60%
        assert allocation[MethodIdentifier.CHAIN_OF_THOUGHT] == 600

    async def test_allocate_budget_hard_multi_method(
        self, initialized_router: SelfBudgeter
    ) -> None:
        """Test hard query uses multi-method approach."""
        hard_query = "Prove and derive the complex multi-step optimization"
        allocation = await initialized_router.allocate_budget(hard_query, 1000)
        assert len(allocation) == 3
        assert MethodIdentifier.MCTS in allocation
        assert MethodIdentifier.SELF_CONSISTENCY in allocation
        assert MethodIdentifier.SELF_VERIFICATION in allocation
        # MCTS gets 50%
        assert allocation[MethodIdentifier.MCTS] == 500


class TestSelfBudgeterHealthCheck:
    """Tests for SelfBudgeter health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = SelfBudgeter()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = SelfBudgeter()
        await router.initialize()
        assert await router.health_check() is True
