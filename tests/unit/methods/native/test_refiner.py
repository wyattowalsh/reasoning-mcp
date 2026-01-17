"""Unit tests for REFINER reasoning method.

This module provides comprehensive tests for the Refiner method implementation,
covering initialization, execution, generator-critic feedback loop, refinement,
and edge cases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.refiner import (
    REFINER_METADATA,
    Refiner,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> Refiner:
    """Create a Refiner method instance for testing.

    Returns:
        A fresh Refiner instance
    """
    return Refiner()


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
        A sample math problem string
    """
    return "What is 5 multiplied by 3, then add 2?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    response = MagicMock()
    response.text = json.dumps({
        "hypothesis": "Mathematical calculation",
        "entities": ["5", "3", "2"],
        "relations": ["multiply", "add"],
        "reasoning_chain": [
            {"step": 1, "action": "multiply", "output": "15"},
            {"step": 2, "action": "add", "output": "17"},
        ],
        "confidence": 0.85,
    })
    ctx.sample = AsyncMock(return_value=response)
    return ctx


class TestRefinerInitialization:
    """Tests for Refiner initialization and setup."""

    def test_create_method(self, method: Refiner) -> None:
        """Test that Refiner can be instantiated."""
        assert method is not None
        assert isinstance(method, Refiner)

    def test_initial_state(self, method: Refiner) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_initialize(self, method: Refiner) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._intermediate_repr == {}
        assert method._critic_feedback == []
        assert method._refinement_history == []
        assert method._iteration == 0
        assert method._max_iterations == 3

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = Refiner()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._iteration = 2

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._iteration == 0

    async def test_health_check_not_initialized(self, method: Refiner) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: Refiner) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestRefinerProperties:
    """Tests for Refiner property accessors."""

    def test_identifier_property(self, method: Refiner) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.REFINER

    def test_name_property(self, method: Refiner) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "REFINER"

    def test_description_property(self, method: Refiner) -> None:
        """Test that description returns the correct method description."""
        assert "generator-critic" in method.description.lower()
        assert "feedback" in method.description.lower()

    def test_category_property(self, method: Refiner) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestRefinerMetadata:
    """Tests for REFINER metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert REFINER_METADATA.identifier == MethodIdentifier.REFINER

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert REFINER_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"generator-critic", "feedback", "refinement"}
        assert expected_tags.issubset(REFINER_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert REFINER_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert REFINER_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert REFINER_METADATA.complexity == 6


class TestRefinerExecution:
    """Tests for Refiner execute() method."""

    async def test_execute_basic(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.REFINER

    async def test_execute_without_initialization_raises(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_generate_phase(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets generate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "generate"
        assert thought.metadata.get("iteration") == 1

    async def test_execute_generates_intermediate_repr(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates intermediate representation."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert "hypothesis" in method._intermediate_repr
        assert "entities" in method._intermediate_repr
        assert "relations" in method._intermediate_repr
        assert "reasoning_chain" in method._intermediate_repr
        assert "confidence" in method._intermediate_repr

    async def test_execute_with_execution_context(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test execute with execution context for sampling."""
        await method.initialize()
        thought = await method.execute(
            session,
            sample_problem,
            execution_context=mock_execution_context,
        )

        assert thought is not None
        assert thought.content != ""


class TestRefinerContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_raises(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that continue_reasoning raises RuntimeError if not initialized."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        method._initialized = False

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, initial)

    async def test_continue_generate_to_critique(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from generate to critique phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "critique"
        assert continuation.type == ThoughtType.VERIFICATION

    async def test_continue_critique_to_refine(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from critique to refine phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)

        refine = await method.continue_reasoning(session, critique)

        assert refine is not None
        assert refine.metadata.get("phase") == "refine"
        assert refine.type == ThoughtType.REVISION

    async def test_continue_to_conclusion(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        refine = await method.continue_reasoning(session, critique)

        conclude = await method.continue_reasoning(session, refine)

        assert conclude is not None
        # May be either conclude or generate (if iteration continues)
        assert conclude.metadata.get("phase") in ("conclude", "generate")


class TestCriticFeedback:
    """Tests for critic feedback generation."""

    async def test_critic_feedback_generated(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that critic feedback is generated during critique phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert len(method._critic_feedback) > 0
        for feedback in method._critic_feedback:
            assert "aspect" in feedback
            assert "score" in feedback
            assert "feedback" in feedback
            assert "suggestion" in feedback

    async def test_critic_feedback_aspects(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that critic feedback covers expected aspects."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        aspects = {f["aspect"] for f in method._critic_feedback}
        expected_aspects = {"completeness", "correctness", "clarity"}
        assert expected_aspects == aspects


class TestRefinementHistory:
    """Tests for refinement history tracking."""

    async def test_refinement_history_recorded(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that refinement history is recorded."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, critique)

        assert len(method._refinement_history) > 0
        for entry in method._refinement_history:
            assert "iteration" in entry
            assert "refinements" in entry
            assert "confidence_before" in entry
            assert "confidence_after" in entry


class TestIntermediateRepresentation:
    """Tests for intermediate representation parsing."""

    def test_parse_intermediate_representation_valid_json(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing valid JSON intermediate representation."""
        valid_json = json.dumps({
            "hypothesis": "Test hypothesis",
            "entities": ["a", "b"],
            "relations": ["add"],
            "reasoning_chain": [{"step": 1, "action": "test", "output": "result"}],
            "confidence": 0.9,
        })

        result = method._parse_intermediate_representation(valid_json, "test input")

        assert result["hypothesis"] == "Test hypothesis"
        assert result["entities"] == ["a", "b"]
        assert result["confidence"] == 0.9

    def test_parse_intermediate_representation_invalid_json(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing invalid JSON falls back to heuristics."""
        invalid_json = "not valid json"

        result = method._parse_intermediate_representation(invalid_json, "test input")

        assert "hypothesis" in result
        assert "entities" in result
        assert "confidence" in result

    def test_parse_intermediate_representation_missing_fields(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing JSON with missing fields uses defaults."""
        partial_json = json.dumps({"hypothesis": "Partial"})

        result = method._parse_intermediate_representation(partial_json, "test input")

        assert result["hypothesis"] == "Partial"
        assert "entities" in result
        assert "relations" in result


class TestCriticFeedbackParsing:
    """Tests for critic feedback parsing."""

    def test_parse_critic_feedback_valid_json(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing valid JSON critic feedback."""
        valid_json = json.dumps([
            {"aspect": "completeness", "score": 0.9, "feedback": "Good", "suggestion": None},
            {"aspect": "correctness", "score": 0.8, "feedback": "OK", "suggestion": "Improve"},
        ])

        result = method._parse_critic_feedback(valid_json)

        assert len(result) == 2
        assert result[0]["aspect"] == "completeness"
        assert result[0]["score"] == 0.9
        assert result[1]["suggestion"] == "Improve"

    def test_parse_critic_feedback_invalid_json(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing invalid JSON falls back to default feedback."""
        invalid_json = "not valid json"

        result = method._parse_critic_feedback(invalid_json)

        assert len(result) == 3
        assert all("aspect" in f for f in result)
        assert all("score" in f for f in result)

    def test_parse_critic_feedback_empty_array(
        self,
        method: Refiner,
    ) -> None:
        """Test parsing empty array falls back to default feedback."""
        empty_json = json.dumps([])

        result = method._parse_critic_feedback(empty_json)

        assert len(result) == 3  # Falls back to default


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: Refiner,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_long_problem_string(
        self,
        method: Refiner,
        session: Session,
    ) -> None:
        """Test execution with long problem string."""
        await method.initialize()
        long_problem = "Calculate " + " then add 1" * 100

        thought = await method.execute(session, long_problem)

        assert thought is not None
        assert thought.content != ""

    async def test_non_math_problem(
        self,
        method: Refiner,
        session: Session,
    ) -> None:
        """Test execution with non-mathematical problem."""
        await method.initialize()
        problem = "What is the capital of France?"

        thought = await method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    def test_extract_entities_with_numbers(
        self,
        method: Refiner,
    ) -> None:
        """Test entity extraction with numbers."""
        text = "Calculate 5 plus 3 equals 8"

        entities = method._extract_entities(text)

        assert "5" in entities
        assert "3" in entities
        assert "8" in entities

    def test_extract_entities_no_numbers(
        self,
        method: Refiner,
    ) -> None:
        """Test entity extraction with no numbers returns defaults."""
        text = "What is the capital of France?"

        entities = method._extract_entities(text)

        assert len(entities) > 0  # Should return defaults


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "generate"

        # Continue through phases
        critique = await method.continue_reasoning(session, initial)
        assert critique.metadata.get("phase") == "critique"

        refine = await method.continue_reasoning(session, critique)
        assert refine.metadata.get("phase") == "refine"

        # Continue until conclusion
        current = refine
        max_steps = 10
        step = 0
        while current.metadata.get("phase") != "conclude" and step < max_steps:
            current = await method.continue_reasoning(session, current)
            step += 1

        # Verify chain structure
        assert session.thought_count >= 4
        assert current.type == ThoughtType.CONCLUSION or step == max_steps

    async def test_session_thought_count_updates(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1


class TestFallbackBehavior:
    """Tests for fallback behavior when sampling is unavailable."""

    async def test_fallback_intermediate_representation(
        self,
        method: Refiner,
    ) -> None:
        """Test fallback intermediate representation generation."""
        result = method._fallback_intermediate_representation("test input")

        assert "hypothesis" in result
        assert "entities" in result
        assert "relations" in result
        assert "reasoning_chain" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    async def test_fallback_critic_feedback(
        self,
        method: Refiner,
    ) -> None:
        """Test fallback critic feedback generation."""
        result = method._fallback_critic_feedback()

        assert len(result) == 3
        assert all(f["aspect"] in ("completeness", "correctness", "clarity") for f in result)
        assert all(0 <= f["score"] <= 1 for f in result)

    async def test_execute_without_execution_context(
        self,
        method: Refiner,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test execute falls back gracefully without execution context."""
        await method.initialize()

        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.REFINER
