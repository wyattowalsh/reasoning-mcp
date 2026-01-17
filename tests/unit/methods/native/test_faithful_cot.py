"""Unit tests for Faithful Chain-of-Thought reasoning method.

This module provides comprehensive tests for the FaithfulCoT method implementation,
covering initialization, execution, symbolic translation, deterministic solving,
faithfulness verification, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.faithful_cot import (
    FAITHFUL_COT_METADATA,
    FaithfulCoT,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> FaithfulCoT:
    """Create a FaithfulCoT method instance for testing.

    Returns:
        A fresh FaithfulCoT instance
    """
    return FaithfulCoT()


@pytest.fixture
def method_no_elicitation() -> FaithfulCoT:
    """Create a FaithfulCoT method with elicitation disabled.

    Returns:
        A FaithfulCoT instance with elicitation disabled
    """
    return FaithfulCoT(enable_elicitation=False)


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
        A sample mathematical problem string
    """
    return "If a train travels at 60 mph for 2.5 hours, how far does it travel?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="distance = speed × time = 60 × 2.5 = 150 miles")
    ctx.ctx = None  # No elicitation context
    return ctx


class TestFaithfulCoTInitialization:
    """Tests for FaithfulCoT initialization and setup."""

    def test_create_method(self, method: FaithfulCoT) -> None:
        """Test that FaithfulCoT can be instantiated."""
        assert method is not None
        assert isinstance(method, FaithfulCoT)

    def test_initial_state(self, method: FaithfulCoT) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "translate"
        assert method._symbolic_form == ""
        assert method._solver_result == ""
        assert method._is_faithful is False

    def test_elicitation_enabled_by_default(self, method: FaithfulCoT) -> None:
        """Test that elicitation is enabled by default."""
        assert method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, method_no_elicitation: FaithfulCoT) -> None:
        """Test that elicitation can be disabled."""
        assert method_no_elicitation.enable_elicitation is False

    async def test_initialize(self, method: FaithfulCoT) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "translate"
        assert method._symbolic_form == ""
        assert method._solver_result == ""
        assert method._is_faithful is False

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = FaithfulCoT()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._symbolic_form = "test"
        method._is_faithful = True

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "translate"
        assert method._symbolic_form == ""
        assert method._is_faithful is False

    async def test_health_check_not_initialized(self, method: FaithfulCoT) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: FaithfulCoT) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestFaithfulCoTProperties:
    """Tests for FaithfulCoT property accessors."""

    def test_identifier_property(self, method: FaithfulCoT) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.FAITHFUL_COT

    def test_name_property(self, method: FaithfulCoT) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Faithful Chain-of-Thought"

    def test_description_property(self, method: FaithfulCoT) -> None:
        """Test that description returns the correct method description."""
        assert "faithful" in method.description.lower()
        assert "symbolic" in method.description.lower()

    def test_category_property(self, method: FaithfulCoT) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestFaithfulCoTMetadata:
    """Tests for FAITHFUL_COT metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert FAITHFUL_COT_METADATA.identifier == MethodIdentifier.FAITHFUL_COT

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert FAITHFUL_COT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"faithful", "symbolic", "deterministic", "verified"}
        assert expected_tags.issubset(FAITHFUL_COT_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert FAITHFUL_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert FAITHFUL_COT_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert FAITHFUL_COT_METADATA.complexity == 6


class TestFaithfulCoTExecution:
    """Tests for FaithfulCoT execute() method."""

    async def test_execute_basic(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.FAITHFUL_COT

    async def test_execute_without_initialization_raises(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_reasoning_thought(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates a REASONING thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.REASONING
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_translate_phase(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets translate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "translate"
        assert "symbolic" in thought.metadata

    async def test_execute_generates_symbolic_form(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates symbolic form."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert method._symbolic_form != ""

    async def test_execute_with_execution_context(
        self,
        method: FaithfulCoT,
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


class TestFaithfulCoTContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_translate_to_solve(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from translate to solve phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "solve"
        assert continuation.type == ThoughtType.REASONING

    async def test_continue_solve_to_verify(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from solve to verify phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)

        verify = await method.continue_reasoning(session, solve)

        assert verify is not None
        assert verify.metadata.get("phase") == "verify"
        assert verify.type == ThoughtType.VERIFICATION

    async def test_continue_to_conclusion(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)
        verify = await method.continue_reasoning(session, solve)

        conclusion = await method.continue_reasoning(session, verify)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: FaithfulCoT,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.FAITHFUL_COT,
            content="Test",
            metadata={"phase": "translate"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestSymbolicTranslation:
    """Tests for symbolic translation functionality."""

    async def test_heuristic_generates_symbolic_form(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that heuristic generates a symbolic form."""
        await method.initialize()
        await method.execute(session, sample_problem)

        # Heuristic should generate some form of symbolic representation
        assert "variables" in method._symbolic_form or "constraints" in method._symbolic_form

    async def test_symbolic_form_in_content(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that symbolic form appears in thought content."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        # Content should reference symbolic/programmatic translation
        assert "symbolic" in thought.content.lower() or "python" in thought.content.lower()


class TestDeterministicSolver:
    """Tests for deterministic solver functionality."""

    async def test_solver_result_generated(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that solver result is generated during solve phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert method._solver_result != ""


class TestFaithfulnessVerification:
    """Tests for faithfulness verification."""

    async def test_faithfulness_verified_during_verify_phase(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that faithfulness is verified during verify phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, solve)

        assert method._is_faithful is True

    async def test_verification_checks_in_content(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that verification content includes checks."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)
        verify = await method.continue_reasoning(session, solve)

        assert "faithfulness" in verify.content.lower() or "verify" in verify.content.lower()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: FaithfulCoT,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: FaithfulCoT,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Calculate the result: " + "x + " * 500 + "y = 0"

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: FaithfulCoT,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None

    async def test_logical_problem(
        self,
        method: FaithfulCoT,
        session: Session,
    ) -> None:
        """Test execution with logical problem."""
        await method.initialize()
        problem = "If P implies Q, and P is true, what can we conclude about Q?"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "translate"

        # Continue through all phases
        solve = await method.continue_reasoning(session, initial)
        assert solve.metadata.get("phase") == "solve"

        verify = await method.continue_reasoning(session, solve)
        assert verify.metadata.get("phase") == "verify"

        conclude = await method.continue_reasoning(session, verify)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 4
        assert conclude.type == ThoughtType.CONCLUSION
        assert method._is_faithful is True

    async def test_session_thought_count_updates(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_thought_parent_chain(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought parent chain is correct."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)
        verify = await method.continue_reasoning(session, solve)

        assert initial.parent_id is None
        assert solve.parent_id == initial.id
        assert verify.parent_id == solve.id

    async def test_thought_depth_increments(
        self,
        method: FaithfulCoT,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        solve = await method.continue_reasoning(session, initial)
        verify = await method.continue_reasoning(session, solve)

        assert initial.depth == 0
        assert solve.depth == 1
        assert verify.depth == 2
