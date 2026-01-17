"""Unit tests for TwoStageGeneration reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.two_stage_generation import (
    TWO_STAGE_GENERATION_METADATA,
    TwoStageGeneration,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> TwoStageGeneration:
    """Create a TwoStageGeneration instance for testing."""
    return TwoStageGeneration()


class TestTwoStageGenerationInitialization:
    """Tests for TwoStageGeneration initialization."""

    def test_create_method(self, method: TwoStageGeneration):
        """Test that TwoStageGeneration can be instantiated."""
        assert method is not None
        assert isinstance(method, TwoStageGeneration)

    def test_initial_state(self, method: TwoStageGeneration):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "think"

    async def test_initialize(self, method: TwoStageGeneration):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: TwoStageGeneration):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: TwoStageGeneration):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestTwoStageGenerationProperties:
    """Tests for TwoStageGeneration property accessors."""

    def test_identifier_property(self, method: TwoStageGeneration):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.TWO_STAGE_GENERATION

    def test_name_property(self, method: TwoStageGeneration):
        """Test that name returns the correct human-readable name."""
        assert "Two-Stage" in method.name or "Generation" in method.name

    def test_description_property(self, method: TwoStageGeneration):
        """Test that description returns the correct method description."""
        assert "think" in method.description.lower()

    def test_category_property(self, method: TwoStageGeneration):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestTwoStageGenerationMetadata:
    """Tests for TwoStageGeneration metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert TWO_STAGE_GENERATION_METADATA.identifier == MethodIdentifier.TWO_STAGE_GENERATION

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert TWO_STAGE_GENERATION_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"two-stage", "think-then-answer"}
        assert expected_tags.issubset(TWO_STAGE_GENERATION_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= TWO_STAGE_GENERATION_METADATA.complexity <= 10


class TestTwoStageGenerationExecution:
    """Tests for TwoStageGeneration execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: TwoStageGeneration, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: TwoStageGeneration, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Think through this problem")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.TWO_STAGE_GENERATION
        assert thought.type == ThoughtType.REASONING

    async def test_execute_sets_metadata(self, method: TwoStageGeneration, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "think"


class TestTwoStageGenerationContinuation:
    """Tests for TwoStageGeneration continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: TwoStageGeneration, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "think"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "verify"

    async def test_continue_sets_parent(
        self, method: TwoStageGeneration, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: TwoStageGeneration, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
