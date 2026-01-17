"""Unit tests for S2R reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.s2r import (
    S2R,
    S2R_METADATA,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> S2R:
    """Create an S2R instance for testing."""
    return S2R()


class TestS2RInitialization:
    """Tests for S2R initialization."""

    def test_create_method(self, method: S2R):
        """Test that S2R can be instantiated."""
        assert method is not None
        assert isinstance(method, S2R)

    def test_initial_state(self, method: S2R):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_initialize(self, method: S2R):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: S2R):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: S2R):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestS2RProperties:
    """Tests for S2R property accessors."""

    def test_identifier_property(self, method: S2R):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.S2R

    def test_name_property(self, method: S2R):
        """Test that name returns the correct human-readable name."""
        assert method.name == "S2R"

    def test_description_property(self, method: S2R):
        """Test that description returns the correct method description."""
        assert "reinforcement" in method.description.lower()

    def test_category_property(self, method: S2R):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestS2RMetadata:
    """Tests for S2R metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert S2R_METADATA.identifier == MethodIdentifier.S2R

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert S2R_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"reinforcement-learning", "self-verification"}
        assert expected_tags.issubset(S2R_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= S2R_METADATA.complexity <= 10


class TestS2RExecution:
    """Tests for S2R execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: S2R, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: S2R, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Solve with RL correction")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.S2R
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: S2R, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "generate"


class TestS2RContinuation:
    """Tests for S2R continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: S2R, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "generate"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "self_verify"

    async def test_continue_sets_parent(
        self, method: S2R, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: S2R, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
