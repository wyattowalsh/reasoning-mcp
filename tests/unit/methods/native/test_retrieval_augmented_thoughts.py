"""Unit tests for RetrievalAugmentedThoughts reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.retrieval_augmented_thoughts import (
    RETRIEVAL_AUGMENTED_THOUGHTS_METADATA,
    RetrievalAugmentedThoughts,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> RetrievalAugmentedThoughts:
    """Create a RetrievalAugmentedThoughts instance for testing."""
    return RetrievalAugmentedThoughts()


class TestRetrievalAugmentedThoughtsInitialization:
    """Tests for RetrievalAugmentedThoughts initialization."""

    def test_create_method(self, method: RetrievalAugmentedThoughts):
        """Test that RetrievalAugmentedThoughts can be instantiated."""
        assert method is not None
        assert isinstance(method, RetrievalAugmentedThoughts)

    def test_initial_state(self, method: RetrievalAugmentedThoughts):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "query"

    async def test_initialize(self, method: RetrievalAugmentedThoughts):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: RetrievalAugmentedThoughts):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: RetrievalAugmentedThoughts):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestRetrievalAugmentedThoughtsProperties:
    """Tests for RetrievalAugmentedThoughts property accessors."""

    def test_identifier_property(self, method: RetrievalAugmentedThoughts):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.RETRIEVAL_AUGMENTED_THOUGHTS

    def test_name_property(self, method: RetrievalAugmentedThoughts):
        """Test that name returns the correct human-readable name."""
        assert "Retrieval" in method.name

    def test_description_property(self, method: RetrievalAugmentedThoughts):
        """Test that description returns the correct method description."""
        assert "rag" in method.description.lower() or "retrieval" in method.description.lower()

    def test_category_property(self, method: RetrievalAugmentedThoughts):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestRetrievalAugmentedThoughtsMetadata:
    """Tests for RetrievalAugmentedThoughts metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert RETRIEVAL_AUGMENTED_THOUGHTS_METADATA.identifier == MethodIdentifier.RETRIEVAL_AUGMENTED_THOUGHTS

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert RETRIEVAL_AUGMENTED_THOUGHTS_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"retrieval", "rag"}
        assert expected_tags.issubset(RETRIEVAL_AUGMENTED_THOUGHTS_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= RETRIEVAL_AUGMENTED_THOUGHTS_METADATA.complexity <= 10


class TestRetrievalAugmentedThoughtsExecution:
    """Tests for RetrievalAugmentedThoughts execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: RetrievalAugmentedThoughts, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: RetrievalAugmentedThoughts, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "What are the facts about X?")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.RETRIEVAL_AUGMENTED_THOUGHTS
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: RetrievalAugmentedThoughts, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "query"


class TestRetrievalAugmentedThoughtsContinuation:
    """Tests for RetrievalAugmentedThoughts continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: RetrievalAugmentedThoughts, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "query"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "retrieve"

    async def test_continue_sets_parent(
        self, method: RetrievalAugmentedThoughts, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: RetrievalAugmentedThoughts, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
