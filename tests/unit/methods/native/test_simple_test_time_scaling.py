"""Unit tests for SimpleTestTimeScaling reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.simple_test_time_scaling import (
    SIMPLE_TEST_TIME_SCALING_METADATA,
    SimpleTestTimeScaling,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> SimpleTestTimeScaling:
    """Create a SimpleTestTimeScaling instance for testing."""
    return SimpleTestTimeScaling()


class TestSimpleTestTimeScalingInitialization:
    """Tests for SimpleTestTimeScaling initialization."""

    def test_create_method(self, method: SimpleTestTimeScaling):
        """Test that SimpleTestTimeScaling can be instantiated."""
        assert method is not None
        assert isinstance(method, SimpleTestTimeScaling)

    def test_initial_state(self, method: SimpleTestTimeScaling):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "budget"

    async def test_initialize(self, method: SimpleTestTimeScaling):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: SimpleTestTimeScaling):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: SimpleTestTimeScaling):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestSimpleTestTimeScalingProperties:
    """Tests for SimpleTestTimeScaling property accessors."""

    def test_identifier_property(self, method: SimpleTestTimeScaling):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.SIMPLE_TEST_TIME_SCALING

    def test_name_property(self, method: SimpleTestTimeScaling):
        """Test that name returns the correct human-readable name."""
        assert "s1" in method.name or "Simple" in method.name

    def test_description_property(self, method: SimpleTestTimeScaling):
        """Test that description returns the correct method description."""
        assert "budget" in method.description.lower()

    def test_category_property(self, method: SimpleTestTimeScaling):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestSimpleTestTimeScalingMetadata:
    """Tests for SimpleTestTimeScaling metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert SIMPLE_TEST_TIME_SCALING_METADATA.identifier == MethodIdentifier.SIMPLE_TEST_TIME_SCALING

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert SIMPLE_TEST_TIME_SCALING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"test-time-scaling", "s1"}
        assert expected_tags.issubset(SIMPLE_TEST_TIME_SCALING_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= SIMPLE_TEST_TIME_SCALING_METADATA.complexity <= 10


class TestSimpleTestTimeScalingExecution:
    """Tests for SimpleTestTimeScaling execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: SimpleTestTimeScaling, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: SimpleTestTimeScaling, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Solve: 2 + 2")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.SIMPLE_TEST_TIME_SCALING
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: SimpleTestTimeScaling, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "budget"


class TestSimpleTestTimeScalingContinuation:
    """Tests for SimpleTestTimeScaling continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: SimpleTestTimeScaling, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "budget"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "think"

    async def test_continue_sets_parent(
        self, method: SimpleTestTimeScaling, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: SimpleTestTimeScaling, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1


class TestBudgetLevels:
    """Tests for budget level constants."""

    def test_budget_levels_defined(self, method: SimpleTestTimeScaling):
        """Test that budget levels are defined."""
        assert hasattr(method, "BUDGET_LOW")
        assert hasattr(method, "BUDGET_MEDIUM")
        assert hasattr(method, "BUDGET_HIGH")

    def test_budget_levels_ordering(self, method: SimpleTestTimeScaling):
        """Test that budget levels are in order."""
        assert method.BUDGET_LOW < method.BUDGET_MEDIUM < method.BUDGET_HIGH
