"""Unit tests for ReasoningViaPlanning reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.reasoning_via_planning import (
    REASONING_VIA_PLANNING_METADATA,
    ReasoningViaPlanning,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> ReasoningViaPlanning:
    """Create a ReasoningViaPlanning instance for testing."""
    return ReasoningViaPlanning()


class TestReasoningViaPlanningInitialization:
    """Tests for ReasoningViaPlanning initialization."""

    def test_create_method(self, method: ReasoningViaPlanning):
        """Test that ReasoningViaPlanning can be instantiated."""
        assert method is not None
        assert isinstance(method, ReasoningViaPlanning)

    def test_initial_state(self, method: ReasoningViaPlanning):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "model"

    async def test_initialize(self, method: ReasoningViaPlanning):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: ReasoningViaPlanning):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: ReasoningViaPlanning):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestReasoningViaPlanningProperties:
    """Tests for ReasoningViaPlanning property accessors."""

    def test_identifier_property(self, method: ReasoningViaPlanning):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.REASONING_VIA_PLANNING

    def test_name_property(self, method: ReasoningViaPlanning):
        """Test that name returns the correct human-readable name."""
        assert "Planning" in method.name

    def test_description_property(self, method: ReasoningViaPlanning):
        """Test that description returns the correct method description."""
        assert "mcts" in method.description.lower() or "planning" in method.description.lower()

    def test_category_property(self, method: ReasoningViaPlanning):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestReasoningViaPlanningMetadata:
    """Tests for ReasoningViaPlanning metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert REASONING_VIA_PLANNING_METADATA.identifier == MethodIdentifier.REASONING_VIA_PLANNING

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert REASONING_VIA_PLANNING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"planning", "mcts"}
        assert expected_tags.issubset(REASONING_VIA_PLANNING_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= REASONING_VIA_PLANNING_METADATA.complexity <= 10


class TestReasoningViaPlanningExecution:
    """Tests for ReasoningViaPlanning execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: ReasoningViaPlanning, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: ReasoningViaPlanning, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Plan a strategy")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.REASONING_VIA_PLANNING
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: ReasoningViaPlanning, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "model"


class TestReasoningViaPlanningContinuation:
    """Tests for ReasoningViaPlanning continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: ReasoningViaPlanning, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "model"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "plan"

    async def test_continue_sets_parent(
        self, method: ReasoningViaPlanning, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: ReasoningViaPlanning, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
