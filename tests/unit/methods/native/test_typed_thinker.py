"""Unit tests for TypedThinker reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.typed_thinker import (
    REASONING_TYPES,
    TYPED_THINKER_METADATA,
    TypedThinker,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> TypedThinker:
    """Create a TypedThinker instance for testing."""
    return TypedThinker()


class TestTypedThinkerInitialization:
    """Tests for TypedThinker initialization."""

    def test_create_method(self, method: TypedThinker):
        """Test that TypedThinker can be instantiated."""
        assert method is not None
        assert isinstance(method, TypedThinker)

    def test_initial_state(self, method: TypedThinker):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "classify"

    async def test_initialize(self, method: TypedThinker):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: TypedThinker):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: TypedThinker):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestTypedThinkerProperties:
    """Tests for TypedThinker property accessors."""

    def test_identifier_property(self, method: TypedThinker):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.TYPED_THINKER

    def test_name_property(self, method: TypedThinker):
        """Test that name returns the correct human-readable name."""
        assert "TypedThinker" in method.name

    def test_description_property(self, method: TypedThinker):
        """Test that description returns the correct method description."""
        assert "reasoning" in method.description.lower() or "type" in method.description.lower()

    def test_category_property(self, method: TypedThinker):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestTypedThinkerMetadata:
    """Tests for TypedThinker metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert TYPED_THINKER_METADATA.identifier == MethodIdentifier.TYPED_THINKER

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert TYPED_THINKER_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"diversity", "typed-reasoning"}
        assert expected_tags.issubset(TYPED_THINKER_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= TYPED_THINKER_METADATA.complexity <= 10


class TestReasoningTypes:
    """Tests for reasoning types constants."""

    def test_deductive_type_exists(self):
        """Test that deductive reasoning type is defined."""
        assert "deductive" in REASONING_TYPES
        assert "name" in REASONING_TYPES["deductive"]
        assert "description" in REASONING_TYPES["deductive"]

    def test_inductive_type_exists(self):
        """Test that inductive reasoning type is defined."""
        assert "inductive" in REASONING_TYPES
        assert "name" in REASONING_TYPES["inductive"]

    def test_abductive_type_exists(self):
        """Test that abductive reasoning type is defined."""
        assert "abductive" in REASONING_TYPES
        assert "name" in REASONING_TYPES["abductive"]

    def test_analogical_type_exists(self):
        """Test that analogical reasoning type is defined."""
        assert "analogical" in REASONING_TYPES
        assert "name" in REASONING_TYPES["analogical"]

    def test_all_types_have_approach(self):
        """Test that all reasoning types have an approach."""
        for type_key, type_data in REASONING_TYPES.items():
            assert "approach" in type_data, f"{type_key} missing approach"


class TestTypedThinkerExecution:
    """Tests for TypedThinker execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: TypedThinker, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: TypedThinker, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Classify this reasoning task")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.TYPED_THINKER
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: TypedThinker, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "classify"


class TestTypedThinkerContinuation:
    """Tests for TypedThinker continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: TypedThinker, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "classify"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "generate"

    async def test_continue_sets_parent(
        self, method: TypedThinker, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: TypedThinker, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
