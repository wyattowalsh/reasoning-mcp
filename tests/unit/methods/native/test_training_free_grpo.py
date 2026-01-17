"""Unit tests for TrainingFreeGrpo reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.training_free_grpo import (
    TRAINING_FREE_GRPO_METADATA,
    TrainingFreeGrpo,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> TrainingFreeGrpo:
    """Create a TrainingFreeGrpo instance for testing."""
    return TrainingFreeGrpo()


class TestTrainingFreeGrpoInitialization:
    """Tests for TrainingFreeGrpo initialization."""

    def test_create_method(self, method: TrainingFreeGrpo):
        """Test that TrainingFreeGrpo can be instantiated."""
        assert method is not None
        assert isinstance(method, TrainingFreeGrpo)

    def test_initial_state(self, method: TrainingFreeGrpo):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "sample_candidates"

    async def test_initialize(self, method: TrainingFreeGrpo):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: TrainingFreeGrpo):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: TrainingFreeGrpo):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestTrainingFreeGrpoProperties:
    """Tests for TrainingFreeGrpo property accessors."""

    def test_identifier_property(self, method: TrainingFreeGrpo):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.TRAINING_FREE_GRPO

    def test_name_property(self, method: TrainingFreeGrpo):
        """Test that name returns the correct human-readable name."""
        assert "GRPO" in method.name

    def test_description_property(self, method: TrainingFreeGrpo):
        """Test that description returns the correct method description."""
        assert "training" in method.description.lower()

    def test_category_property(self, method: TrainingFreeGrpo):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestTrainingFreeGrpoMetadata:
    """Tests for TrainingFreeGrpo metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert TRAINING_FREE_GRPO_METADATA.identifier == MethodIdentifier.TRAINING_FREE_GRPO

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert TRAINING_FREE_GRPO_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"training-free", "grpo"}
        assert expected_tags.issubset(TRAINING_FREE_GRPO_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= TRAINING_FREE_GRPO_METADATA.complexity <= 10


class TestTrainingFreeGrpoExecution:
    """Tests for TrainingFreeGrpo execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: TrainingFreeGrpo, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: TrainingFreeGrpo, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Generate and rank candidates")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.TRAINING_FREE_GRPO
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: TrainingFreeGrpo, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "sample_candidates"


class TestTrainingFreeGrpoContinuation:
    """Tests for TrainingFreeGrpo continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: TrainingFreeGrpo, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "sample_candidates"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "rank_relatively"

    async def test_continue_sets_parent(
        self, method: TrainingFreeGrpo, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: TrainingFreeGrpo, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
