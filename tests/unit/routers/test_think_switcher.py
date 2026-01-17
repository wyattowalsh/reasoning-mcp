"""Unit tests for ThinkSwitcher router.

Tests fast/normal/slow mode selection for compute efficiency.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.think_switcher import THINK_SWITCHER_METADATA, ThinkSwitcher


class TestThinkSwitcherMetadata:
    """Tests for ThinkSwitcher metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert THINK_SWITCHER_METADATA.identifier == RouterIdentifier.THINK_SWITCHER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert THINK_SWITCHER_METADATA.name == "ThinkSwitcher"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "mode-selection" in THINK_SWITCHER_METADATA.tags
        assert "efficiency" in THINK_SWITCHER_METADATA.tags
        assert "adaptive" in THINK_SWITCHER_METADATA.tags

    def test_metadata_supports_budget_control(self) -> None:
        """Test metadata indicates budget control support."""
        assert THINK_SWITCHER_METADATA.supports_budget_control is True

    def test_metadata_does_not_support_multi_model(self) -> None:
        """Test metadata indicates no multi-model support."""
        assert THINK_SWITCHER_METADATA.supports_multi_model is False

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= THINK_SWITCHER_METADATA.complexity <= 10


class TestThinkSwitcherInitialization:
    """Tests for ThinkSwitcher initialization."""

    def test_create_instance(self) -> None:
        """Test creating ThinkSwitcher instance."""
        router = ThinkSwitcher()
        assert router is not None
        assert router._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        router = ThinkSwitcher()
        assert router.identifier == RouterIdentifier.THINK_SWITCHER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        router = ThinkSwitcher()
        assert router.name == "ThinkSwitcher"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        router = ThinkSwitcher()
        await router.initialize()
        assert router._initialized is True

    async def test_initialize_sets_normal_mode(self) -> None:
        """Test initialize sets mode to normal."""
        router = ThinkSwitcher()
        await router.initialize()
        assert router._current_mode == "normal"


class TestThinkSwitcherModeSelection:
    """Tests for ThinkSwitcher mode selection."""

    @pytest.fixture
    async def initialized_router(self) -> ThinkSwitcher:
        """Create an initialized ThinkSwitcher."""
        router = ThinkSwitcher()
        await router.initialize()
        return router

    def test_select_fast_for_short_query(self, initialized_router: ThinkSwitcher) -> None:
        """Test fast mode selected for short queries (<50 chars)."""
        mode = initialized_router._select_mode("Hi")
        assert mode == "fast"

    def test_select_fast_boundary(self, initialized_router: ThinkSwitcher) -> None:
        """Test fast mode boundary at 49 chars."""
        query = "x" * 49
        mode = initialized_router._select_mode(query)
        assert mode == "fast"

    def test_select_normal_at_50_chars(self, initialized_router: ThinkSwitcher) -> None:
        """Test normal mode at exactly 50 chars."""
        query = "x" * 50
        mode = initialized_router._select_mode(query)
        assert mode == "normal"

    def test_select_normal_for_medium_query(self, initialized_router: ThinkSwitcher) -> None:
        """Test normal mode for medium queries (50-200 chars)."""
        query = "x" * 100
        mode = initialized_router._select_mode(query)
        assert mode == "normal"

    def test_select_normal_boundary(self, initialized_router: ThinkSwitcher) -> None:
        """Test normal mode boundary at 199 chars."""
        query = "x" * 199
        mode = initialized_router._select_mode(query)
        assert mode == "normal"

    def test_select_slow_at_200_chars(self, initialized_router: ThinkSwitcher) -> None:
        """Test slow mode at exactly 200 chars."""
        query = "x" * 200
        mode = initialized_router._select_mode(query)
        assert mode == "slow"

    def test_select_slow_for_long_query(self, initialized_router: ThinkSwitcher) -> None:
        """Test slow mode for long queries (>=200 chars)."""
        query = "x" * 500
        mode = initialized_router._select_mode(query)
        assert mode == "slow"


class TestThinkSwitcherRouting:
    """Tests for ThinkSwitcher routing functionality."""

    @pytest.fixture
    async def initialized_router(self) -> ThinkSwitcher:
        """Create an initialized ThinkSwitcher."""
        router = ThinkSwitcher()
        await router.initialize()
        return router

    async def test_route_raises_when_not_initialized(self) -> None:
        """Test routing raises error when not initialized."""
        router = ThinkSwitcher()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.route("test query")

    async def test_route_returns_method(self, initialized_router: ThinkSwitcher) -> None:
        """Test routing returns a valid method."""
        result = await initialized_router.route("test query")
        assert result is not None

    async def test_route_fast_returns_zero_shot_cot(
        self, initialized_router: ThinkSwitcher
    ) -> None:
        """Test fast mode routes to zero-shot CoT."""
        result = await initialized_router.route("Hi")
        assert result == MethodIdentifier.ZERO_SHOT_COT
        assert initialized_router._current_mode == "fast"

    async def test_route_normal_returns_cot(self, initialized_router: ThinkSwitcher) -> None:
        """Test normal mode routes to CoT."""
        query = "This is a medium length query for testing purposes"
        result = await initialized_router.route(query)
        assert result == MethodIdentifier.CHAIN_OF_THOUGHT
        assert initialized_router._current_mode == "normal"

    async def test_route_slow_returns_tot(self, initialized_router: ThinkSwitcher) -> None:
        """Test slow mode routes to Tree of Thoughts."""
        query = "x" * 250  # Long query
        result = await initialized_router.route(query)
        assert result == MethodIdentifier.TREE_OF_THOUGHTS
        assert initialized_router._current_mode == "slow"

    async def test_route_updates_current_mode(self, initialized_router: ThinkSwitcher) -> None:
        """Test routing updates current mode."""
        await initialized_router.route("Hi")
        assert initialized_router._current_mode == "fast"

        await initialized_router.route("x" * 100)
        assert initialized_router._current_mode == "normal"

        await initialized_router.route("x" * 300)
        assert initialized_router._current_mode == "slow"

    async def test_route_with_context(self, initialized_router: ThinkSwitcher) -> None:
        """Test routing with context parameter."""
        result = await initialized_router.route(
            "test query",
            context={"domain": "math"},
        )
        assert result is not None


class TestThinkSwitcherBudgetAllocation:
    """Tests for ThinkSwitcher budget allocation."""

    @pytest.fixture
    async def initialized_router(self) -> ThinkSwitcher:
        """Create an initialized ThinkSwitcher."""
        router = ThinkSwitcher()
        await router.initialize()
        return router

    async def test_allocate_budget_raises_when_not_initialized(self) -> None:
        """Test budget allocation raises error when not initialized."""
        router = ThinkSwitcher()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await router.allocate_budget("test query", 100)

    async def test_allocate_budget_returns_allocation(
        self, initialized_router: ThinkSwitcher
    ) -> None:
        """Test budget allocation returns valid allocation."""
        allocation = await initialized_router.allocate_budget("test query", 1000)
        assert isinstance(allocation, dict)
        assert len(allocation) == 1

    async def test_allocate_budget_fast_mode_30_percent(
        self, initialized_router: ThinkSwitcher
    ) -> None:
        """Test fast mode allocates 30% of budget."""
        allocation = await initialized_router.allocate_budget("Hi", 1000)
        assert MethodIdentifier.ZERO_SHOT_COT in allocation
        assert allocation[MethodIdentifier.ZERO_SHOT_COT] == 300

    async def test_allocate_budget_normal_mode_60_percent(
        self, initialized_router: ThinkSwitcher
    ) -> None:
        """Test normal mode allocates 60% of budget."""
        query = "x" * 100
        allocation = await initialized_router.allocate_budget(query, 1000)
        assert MethodIdentifier.CHAIN_OF_THOUGHT in allocation
        assert allocation[MethodIdentifier.CHAIN_OF_THOUGHT] == 600

    async def test_allocate_budget_slow_mode_100_percent(
        self, initialized_router: ThinkSwitcher
    ) -> None:
        """Test slow mode allocates 100% of budget."""
        query = "x" * 300
        allocation = await initialized_router.allocate_budget(query, 1000)
        assert MethodIdentifier.TREE_OF_THOUGHTS in allocation
        assert allocation[MethodIdentifier.TREE_OF_THOUGHTS] == 1000


class TestThinkSwitcherHealthCheck:
    """Tests for ThinkSwitcher health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        router = ThinkSwitcher()
        assert await router.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        router = ThinkSwitcher()
        await router.initialize()
        assert await router.health_check() is True
