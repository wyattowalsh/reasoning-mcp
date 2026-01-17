"""Unit tests for SyzygyOfThoughts reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.syzygy_of_thoughts import (
    SYZYGY_OF_THOUGHTS_METADATA,
    SyzygyOfThoughts,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> SyzygyOfThoughts:
    """Create a SyzygyOfThoughts instance for testing."""
    return SyzygyOfThoughts()


class TestSyzygyOfThoughtsInitialization:
    """Tests for SyzygyOfThoughts initialization."""

    def test_create_method(self, method: SyzygyOfThoughts):
        """Test that SyzygyOfThoughts can be instantiated."""
        assert method is not None
        assert isinstance(method, SyzygyOfThoughts)

    def test_initial_state(self, method: SyzygyOfThoughts):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "identify"

    async def test_initialize(self, method: SyzygyOfThoughts):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: SyzygyOfThoughts):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: SyzygyOfThoughts):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestSyzygyOfThoughtsProperties:
    """Tests for SyzygyOfThoughts property accessors."""

    def test_identifier_property(self, method: SyzygyOfThoughts):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.SYZYGY_OF_THOUGHTS

    def test_name_property(self, method: SyzygyOfThoughts):
        """Test that name returns the correct human-readable name."""
        assert "Syzygy" in method.name

    def test_description_property(self, method: SyzygyOfThoughts):
        """Test that description returns the correct method description."""
        assert "alignment" in method.description.lower() or "perspective" in method.description.lower()

    def test_category_property(self, method: SyzygyOfThoughts):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.HOLISTIC


class TestSyzygyOfThoughtsMetadata:
    """Tests for SyzygyOfThoughts metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert SYZYGY_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.SYZYGY_OF_THOUGHTS

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert SYZYGY_OF_THOUGHTS_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"multi-perspective", "alignment"}
        assert expected_tags.issubset(SYZYGY_OF_THOUGHTS_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= SYZYGY_OF_THOUGHTS_METADATA.complexity <= 10


class TestSyzygyOfThoughtsExecution:
    """Tests for SyzygyOfThoughts execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: SyzygyOfThoughts, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: SyzygyOfThoughts, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Analyze from multiple perspectives")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.SYZYGY_OF_THOUGHTS
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: SyzygyOfThoughts, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "identify"


class TestSyzygyOfThoughtsContinuation:
    """Tests for SyzygyOfThoughts continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: SyzygyOfThoughts, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "identify"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "align"

    async def test_continue_sets_parent(
        self, method: SyzygyOfThoughts, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: SyzygyOfThoughts, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
