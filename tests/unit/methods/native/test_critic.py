"""Unit tests for CRITIC reasoning method.

This module provides comprehensive tests for the Critic method implementation,
covering initialization, execution, tool-interactive critiquing, self-correction,
and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.critic import (
    CRITIC_METADATA,
    Critic,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> Critic:
    """Create a Critic method instance for testing.

    Returns:
        A fresh Critic instance
    """
    return Critic()


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
    response.text = "The answer is 17. First, 5 × 3 = 15. Then, 15 + 2 = 17."
    ctx.sample = AsyncMock(return_value=response)
    return ctx


class TestCriticInitialization:
    """Tests for Critic initialization and setup."""

    def test_create_method(self, method: Critic) -> None:
        """Test that Critic can be instantiated."""
        assert method is not None
        assert isinstance(method, Critic)

    def test_initial_state(self, method: Critic) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_initialize(self, method: Critic) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._initial_answer == {}
        assert method._tool_verifications == []
        assert method._corrections == []
        assert method._final_answer is None

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = Critic()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._final_answer = "test"

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._final_answer is None

    async def test_health_check_not_initialized(self, method: Critic) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: Critic) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestCriticProperties:
    """Tests for Critic property accessors."""

    def test_identifier_property(self, method: Critic) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.CRITIC

    def test_name_property(self, method: Critic) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "CRITIC"

    def test_description_property(self, method: Critic) -> None:
        """Test that description returns the correct method description."""
        assert "tool" in method.description.lower()
        assert "critiquing" in method.description.lower()

    def test_category_property(self, method: Critic) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestCriticMetadata:
    """Tests for CRITIC metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert CRITIC_METADATA.identifier == MethodIdentifier.CRITIC

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert CRITIC_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"tool-use", "self-correction", "verification"}
        assert expected_tags.issubset(CRITIC_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert CRITIC_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert CRITIC_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert CRITIC_METADATA.complexity == 7


class TestCriticExecution:
    """Tests for Critic execute() method."""

    async def test_execute_basic(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.CRITIC

    async def test_execute_without_initialization_raises(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: Critic,
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
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets generate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "generate"

    async def test_execute_generates_initial_answer(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates initial answer with claims."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert "reasoning" in method._initial_answer
        assert "answer" in method._initial_answer
        assert "claims" in method._initial_answer
        assert len(method._initial_answer["claims"]) > 0

    async def test_execute_with_execution_context(
        self,
        method: Critic,
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


class TestCriticContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_generate_to_critique(
        self,
        method: Critic,
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

    async def test_continue_critique_to_correct(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from critique to correct phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)

        correct = await method.continue_reasoning(session, critique)

        assert correct is not None
        assert correct.metadata.get("phase") == "correct"
        assert correct.type == ThoughtType.REVISION

    async def test_continue_correct_to_validate(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from correct to validate phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        correct = await method.continue_reasoning(session, critique)

        validate = await method.continue_reasoning(session, correct)

        assert validate is not None
        assert validate.metadata.get("phase") == "validate"
        assert validate.type == ThoughtType.VERIFICATION

    async def test_continue_to_conclusion(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        correct = await method.continue_reasoning(session, critique)
        validate = await method.continue_reasoning(session, correct)

        conclusion = await method.continue_reasoning(session, validate)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION


class TestToolVerification:
    """Tests for tool-interactive verification."""

    async def test_tool_verifications_generated(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that tool verifications are generated during critique."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert len(method._tool_verifications) > 0
        for verification in method._tool_verifications:
            assert "claim_id" in verification
            assert "tool" in verification
            assert "result" in verification
            assert "verified" in verification

    async def test_verifications_use_calculator_tool(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that verifications use calculator tool for math."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        tools_used = {v["tool"] for v in method._tool_verifications}
        assert "calculator" in tools_used


class TestSelfCorrection:
    """Tests for self-correction functionality."""

    async def test_corrections_applied_when_needed(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that corrections are applied based on verification results."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, critique)

        # Corrections list should exist (may be empty if all verified)
        assert isinstance(method._corrections, list)

    async def test_final_answer_set_after_correction(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that final answer is set after correction phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        critique = await method.continue_reasoning(session, initial)
        correct = await method.continue_reasoning(session, critique)
        await method.continue_reasoning(session, correct)

        assert method._final_answer is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: Critic,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_non_math_problem(
        self,
        method: Critic,
        session: Session,
    ) -> None:
        """Test execution with non-mathematical problem."""
        await method.initialize()
        problem = "What is the capital of France?"

        thought = await method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_complex_math_problem(
        self,
        method: Critic,
        session: Session,
    ) -> None:
        """Test execution with complex math problem."""
        await method.initialize()
        problem = "Calculate: (15 * 3) + (27 / 9) - 4^2"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "generate"

        # Continue through all phases
        critique = await method.continue_reasoning(session, initial)
        assert critique.metadata.get("phase") == "critique"

        correct = await method.continue_reasoning(session, critique)
        assert correct.metadata.get("phase") == "correct"

        validate = await method.continue_reasoning(session, correct)
        assert validate.metadata.get("phase") == "validate"

        conclude = await method.continue_reasoning(session, validate)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION
        assert method._final_answer is not None

    async def test_session_thought_count_updates(
        self,
        method: Critic,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1
