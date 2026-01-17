"""Unit tests for CoT-RAG reasoning method.

This module provides comprehensive tests for the CoTRAG method implementation,
covering initialization, execution, KG modulation, RAG retrieval, pseudo-program
execution, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.cot_rag import (
    COT_RAG_METADATA,
    CoTRAG,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> CoTRAG:
    """Create a CoTRAG method instance for testing.

    Returns:
        A fresh CoTRAG instance
    """
    return CoTRAG()


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A new Session instance in ACTIVE status
    """
    return Session().start()


@pytest.fixture
def sample_problem() -> str:
    """Provide a sample problem for testing.

    Returns:
        A sample problem string
    """
    return "What is the relationship between quantum entanglement and information transfer?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="path: Entity_A -> relation -> Entity_B, confidence: 0.85")
    return ctx


class TestCoTRAGInitialization:
    """Tests for CoTRAG initialization and setup."""

    def test_create_method(self, method: CoTRAG) -> None:
        """Test that CoTRAG can be instantiated."""
        assert method is not None
        assert isinstance(method, CoTRAG)

    def test_initial_state(self, method: CoTRAG) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "kg_modulate"

    async def test_initialize(self, method: CoTRAG) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "kg_modulate"
        assert method._kg_paths == []
        assert method._retrieved_cases == []
        assert method._program_steps == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = CoTRAG()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "kg_modulate"

    async def test_health_check_not_initialized(self, method: CoTRAG) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: CoTRAG) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestCoTRAGProperties:
    """Tests for CoTRAG property accessors."""

    def test_identifier_property(self, method: CoTRAG) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.COT_RAG

    def test_name_property(self, method: CoTRAG) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "CoT-RAG"

    def test_description_property(self, method: CoTRAG) -> None:
        """Test that description returns the correct method description."""
        assert "knowledge" in method.description.lower()
        assert "rag" in method.description.lower()

    def test_category_property(self, method: CoTRAG) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestCoTRAGMetadata:
    """Tests for CoTRAG metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert COT_RAG_METADATA.identifier == MethodIdentifier.COT_RAG

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert COT_RAG_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"knowledge-graph", "rag", "pseudo-program"}
        assert expected_tags.issubset(COT_RAG_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert COT_RAG_METADATA.supports_branching is True

    def test_metadata_requires_context(self) -> None:
        """Test that metadata correctly indicates context requirement."""
        assert COT_RAG_METADATA.requires_context is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has high complexity rating."""
        assert COT_RAG_METADATA.complexity == 8


class TestCoTRAGExecution:
    """Tests for CoTRAG execute() method."""

    async def test_execute_basic(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.COT_RAG

    async def test_execute_without_initialization_raises(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_kg_modulate_phase(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets kg_modulate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "kg_modulate"
        assert "kg_paths" in thought.metadata

    async def test_execute_generates_kg_paths(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates KG paths."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert len(method._kg_paths) > 0
        for path in method._kg_paths:
            assert "path" in path
            assert "confidence" in path


class TestCoTRAGContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_kg_to_rag(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from kg_modulate to rag_retrieve phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "rag_retrieve"
        assert continuation.type == ThoughtType.REASONING

    async def test_continue_rag_to_pseudo_program(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from rag_retrieve to pseudo_program phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        rag = await method.continue_reasoning(session, initial)

        program = await method.continue_reasoning(session, rag)

        assert program is not None
        assert program.metadata.get("phase") == "pseudo_program"
        assert program.type == ThoughtType.SYNTHESIS

    async def test_continue_program_to_synthesize(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from pseudo_program to synthesize phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        rag = await method.continue_reasoning(session, initial)
        program = await method.continue_reasoning(session, rag)

        synthesize = await method.continue_reasoning(session, program)

        assert synthesize is not None
        assert synthesize.metadata.get("phase") == "synthesize"
        assert synthesize.type == ThoughtType.VERIFICATION

    async def test_continue_to_conclusion(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        rag = await method.continue_reasoning(session, initial)
        program = await method.continue_reasoning(session, rag)
        synthesize = await method.continue_reasoning(session, program)

        conclusion = await method.continue_reasoning(session, synthesize)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION


class TestKGPathGeneration:
    """Tests for knowledge graph path generation."""

    async def test_heuristic_kg_paths(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test heuristic KG path generation."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert len(method._kg_paths) >= 3
        for path in method._kg_paths:
            assert 0.0 <= path["confidence"] <= 1.0

    async def test_kg_paths_contain_entities(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that KG paths contain entity relationships."""
        await method.initialize()
        await method.execute(session, sample_problem)

        for path in method._kg_paths:
            # Heuristic paths should have Entity format
            assert "Entity" in path["path"] or "->" in path["path"]


class TestRetrievedCases:
    """Tests for RAG retrieval functionality."""

    async def test_retrieved_cases_generated(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that cases are retrieved during RAG phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert len(method._retrieved_cases) > 0
        for case in method._retrieved_cases:
            assert "case" in case
            assert "description" in case
            assert "match" in case


class TestPseudoProgram:
    """Tests for pseudo-program execution."""

    async def test_program_steps_generated(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that program steps are generated."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        rag = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, rag)

        assert len(method._program_steps) > 0
        # Heuristic program should have standard steps
        step_text = " ".join(method._program_steps)
        assert "INIT" in step_text or "problem" in step_text.lower()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: CoTRAG,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: CoTRAG,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Analyze the complex relationship: " + "context " * 500

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: CoTRAG,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Analyze: α → β → γ with ∑(x) = ∞"

        thought = await method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: CoTRAG,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "kg_modulate"

        # Continue through all phases
        rag = await method.continue_reasoning(session, initial)
        assert rag.metadata.get("phase") == "rag_retrieve"

        program = await method.continue_reasoning(session, rag)
        assert program.metadata.get("phase") == "pseudo_program"

        synthesize = await method.continue_reasoning(session, program)
        assert synthesize.metadata.get("phase") == "synthesize"

        conclude = await method.continue_reasoning(session, synthesize)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION
