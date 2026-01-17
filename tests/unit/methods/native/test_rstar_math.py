"""Unit tests for RStarMath reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.rstar_math import (
    RSTAR_MATH_METADATA,
    RStarMath,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> RStarMath:
    """Create a RStarMath instance for testing."""
    return RStarMath()


class TestRStarMathInitialization:
    """Tests for RStarMath initialization."""

    def test_create_method(self, method: RStarMath):
        """Test that RStarMath can be instantiated."""
        assert method is not None
        assert isinstance(method, RStarMath)

    def test_initial_state(self, method: RStarMath):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "initialize"

    async def test_initialize(self, method: RStarMath):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: RStarMath):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: RStarMath):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestRStarMathProperties:
    """Tests for RStarMath property accessors."""

    def test_identifier_property(self, method: RStarMath):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.RSTAR_MATH

    def test_name_property(self, method: RStarMath):
        """Test that name returns the correct human-readable name."""
        assert "rStar" in method.name

    def test_description_property(self, method: RStarMath):
        """Test that description returns the correct method description."""
        assert "mcts" in method.description.lower()

    def test_category_property(self, method: RStarMath):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestRStarMathMetadata:
    """Tests for RStarMath metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert RSTAR_MATH_METADATA.identifier == MethodIdentifier.RSTAR_MATH

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert RSTAR_MATH_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"mcts", "math"}
        assert expected_tags.issubset(RSTAR_MATH_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= RSTAR_MATH_METADATA.complexity <= 10


class TestRStarMathExecution:
    """Tests for RStarMath execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: RStarMath, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: RStarMath, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Prove: 1 + 1 = 2")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.RSTAR_MATH
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: RStarMath, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "initialize"


class TestRStarMathContinuation:
    """Tests for RStarMath continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: RStarMath, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "initialize"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "explore"

    async def test_continue_sets_parent(
        self, method: RStarMath, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: RStarMath, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1


class TestCustomConfiguration:
    """Tests for custom configuration options."""

    def test_custom_max_depth(self):
        """Test custom max depth setting."""
        method = RStarMath(max_depth=10)
        assert method._max_depth == 10

    def test_custom_num_rollouts(self):
        """Test custom number of rollouts."""
        method = RStarMath(num_rollouts=8)
        assert method._num_rollouts == 8

    def test_default_configuration(self, method: RStarMath):
        """Test default configuration values."""
        assert method._max_depth == 8
        assert method._num_rollouts == 4
