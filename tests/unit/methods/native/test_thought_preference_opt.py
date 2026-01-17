"""Unit tests for ThoughtPreferenceOpt reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.thought_preference_opt import (
    THOUGHT_PREFERENCE_OPT_METADATA,
    ThoughtPreferenceOpt,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> ThoughtPreferenceOpt:
    """Create a ThoughtPreferenceOpt instance for testing."""
    return ThoughtPreferenceOpt()


class TestThoughtPreferenceOptInitialization:
    """Tests for ThoughtPreferenceOpt initialization."""

    def test_create_method(self, method: ThoughtPreferenceOpt):
        """Test that ThoughtPreferenceOpt can be instantiated."""
        assert method is not None
        assert isinstance(method, ThoughtPreferenceOpt)

    def test_initial_state(self, method: ThoughtPreferenceOpt):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "think"

    async def test_initialize(self, method: ThoughtPreferenceOpt):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: ThoughtPreferenceOpt):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: ThoughtPreferenceOpt):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestThoughtPreferenceOptProperties:
    """Tests for ThoughtPreferenceOpt property accessors."""

    def test_identifier_property(self, method: ThoughtPreferenceOpt):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.THOUGHT_PREFERENCE_OPT

    def test_name_property(self, method: ThoughtPreferenceOpt):
        """Test that name returns the correct human-readable name."""
        assert "Thought" in method.name or "Preference" in method.name

    def test_description_property(self, method: ThoughtPreferenceOpt):
        """Test that description returns the correct method description."""
        assert "thought" in method.description.lower() or "preference" in method.description.lower()

    def test_category_property(self, method: ThoughtPreferenceOpt):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestThoughtPreferenceOptMetadata:
    """Tests for ThoughtPreferenceOpt metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert THOUGHT_PREFERENCE_OPT_METADATA.identifier == MethodIdentifier.THOUGHT_PREFERENCE_OPT

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert THOUGHT_PREFERENCE_OPT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"preference", "internal-thoughts"}
        assert expected_tags.issubset(THOUGHT_PREFERENCE_OPT_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= THOUGHT_PREFERENCE_OPT_METADATA.complexity <= 10


class TestThoughtPreferenceOptExecution:
    """Tests for ThoughtPreferenceOpt execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: ThoughtPreferenceOpt, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: ThoughtPreferenceOpt, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Complete the task")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.THOUGHT_PREFERENCE_OPT
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: ThoughtPreferenceOpt, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "think"


class TestThoughtPreferenceOptContinuation:
    """Tests for ThoughtPreferenceOpt continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: ThoughtPreferenceOpt, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "think"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "prefer"

    async def test_continue_sets_parent(
        self, method: ThoughtPreferenceOpt, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: ThoughtPreferenceOpt, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
