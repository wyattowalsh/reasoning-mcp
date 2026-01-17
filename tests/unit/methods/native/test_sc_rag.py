"""Unit tests for SCRAG reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.sc_rag import (
    SC_RAG_METADATA,
    SCRAG,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> SCRAG:
    """Create an SCRAG instance for testing."""
    return SCRAG()


class TestSCRAGInitialization:
    """Tests for SCRAG initialization."""

    def test_create_method(self, method: SCRAG):
        """Test that SCRAG can be instantiated."""
        assert method is not None
        assert isinstance(method, SCRAG)

    def test_initial_state(self, method: SCRAG):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "extract"

    async def test_initialize(self, method: SCRAG):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: SCRAG):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: SCRAG):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestSCRAGProperties:
    """Tests for SCRAG property accessors."""

    def test_identifier_property(self, method: SCRAG):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.SC_RAG

    def test_name_property(self, method: SCRAG):
        """Test that name returns the correct human-readable name."""
        assert "Self-Corrective RAG" in method.name

    def test_description_property(self, method: SCRAG):
        """Test that description returns the correct method description."""
        assert "evidence" in method.description.lower()

    def test_category_property(self, method: SCRAG):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestSCRAGMetadata:
    """Tests for SCRAG metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert SC_RAG_METADATA.identifier == MethodIdentifier.SC_RAG

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert SC_RAG_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"self-correction", "rag"}
        assert expected_tags.issubset(SC_RAG_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= SC_RAG_METADATA.complexity <= 10


class TestSCRAGExecution:
    """Tests for SCRAG execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: SCRAG, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: SCRAG, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Find evidence for X")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.SC_RAG
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: SCRAG, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "extract"


class TestSCRAGContinuation:
    """Tests for SCRAG continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: SCRAG, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "extract"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "assess"

    async def test_continue_sets_parent(
        self, method: SCRAG, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: SCRAG, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
