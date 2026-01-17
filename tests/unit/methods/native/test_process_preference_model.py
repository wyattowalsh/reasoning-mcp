"""Unit tests for ProcessPreferenceModel reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.process_preference_model import (
    PROCESS_PREFERENCE_MODEL_METADATA,
    ProcessPreferenceModel,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> ProcessPreferenceModel:
    """Create a ProcessPreferenceModel instance for testing."""
    return ProcessPreferenceModel()


class TestProcessPreferenceModelInitialization:
    """Tests for ProcessPreferenceModel initialization."""

    def test_create_method(self, method: ProcessPreferenceModel):
        """Test that ProcessPreferenceModel can be instantiated."""
        assert method is not None
        assert isinstance(method, ProcessPreferenceModel)

    def test_initial_state(self, method: ProcessPreferenceModel):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_initialize(self, method: ProcessPreferenceModel):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: ProcessPreferenceModel):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: ProcessPreferenceModel):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestProcessPreferenceModelProperties:
    """Tests for ProcessPreferenceModel property accessors."""

    def test_identifier_property(self, method: ProcessPreferenceModel):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.PROCESS_PREFERENCE_MODEL

    def test_name_property(self, method: ProcessPreferenceModel):
        """Test that name returns the correct human-readable name."""
        assert method.name == "Process Preference Model"

    def test_description_property(self, method: ProcessPreferenceModel):
        """Test that description returns the correct method description."""
        assert "preference" in method.description.lower()

    def test_category_property(self, method: ProcessPreferenceModel):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestProcessPreferenceModelMetadata:
    """Tests for ProcessPreferenceModel metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert PROCESS_PREFERENCE_MODEL_METADATA.identifier == MethodIdentifier.PROCESS_PREFERENCE_MODEL

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert PROCESS_PREFERENCE_MODEL_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"preference", "pairwise"}
        assert expected_tags.issubset(PROCESS_PREFERENCE_MODEL_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= PROCESS_PREFERENCE_MODEL_METADATA.complexity <= 10


class TestProcessPreferenceModelExecution:
    """Tests for ProcessPreferenceModel execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: ProcessPreferenceModel, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: ProcessPreferenceModel, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Solve: 2 + 2")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.PROCESS_PREFERENCE_MODEL
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: ProcessPreferenceModel, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "generate"


class TestProcessPreferenceModelContinuation:
    """Tests for ProcessPreferenceModel continue_reasoning() method."""

    async def test_continue_reasoning_without_initialization_raises(
        self, method: ProcessPreferenceModel, session: Session
    ):
        """Test that continue_reasoning raises if not initialized."""
        other_method = ProcessPreferenceModel()
        await other_method.initialize()
        initial = await other_method.execute(session, "Test")

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, initial)

    async def test_continue_reasoning_advances_phase(
        self, method: ProcessPreferenceModel, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "generate"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "compare"

        step3 = await method.continue_reasoning(session, step2)
        assert step3.metadata["phase"] == "rank"

        step4 = await method.continue_reasoning(session, step3)
        assert step4.metadata["phase"] == "select"

        step5 = await method.continue_reasoning(session, step4)
        assert step5.metadata["phase"] == "conclude"

    async def test_continue_sets_parent(
        self, method: ProcessPreferenceModel, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: ProcessPreferenceModel, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1


class TestCustomConfiguration:
    """Tests for custom configuration options."""

    def test_custom_num_trajectories(self):
        """Test custom number of trajectories."""
        method = ProcessPreferenceModel(num_trajectories=5)
        assert method._num_trajectories == 5

    def test_default_num_trajectories(self, method: ProcessPreferenceModel):
        """Test default number of trajectories."""
        assert method._num_trajectories == 3
