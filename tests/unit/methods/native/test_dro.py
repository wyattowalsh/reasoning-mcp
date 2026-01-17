"""Unit tests for DRO (Direct Reasoning Optimization) reasoning method.

This module provides comprehensive tests for the Dro method implementation,
covering initialization, execution, self-scoring, optimization, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.dro import (
    DRO_METADATA,
    Dro,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> Dro:
    """Create a Dro method instance for testing.

    Returns:
        A fresh Dro instance
    """
    return Dro()


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
    return "What is 5 times 3 plus 2?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Initial reasoning: 5×3=15, then 15+2=17. Score: 0.85")
    return ctx


class TestDroInitialization:
    """Tests for Dro initialization and setup."""

    def test_create_method(self, method: Dro) -> None:
        """Test that Dro can be instantiated."""
        assert method is not None
        assert isinstance(method, Dro)

    def test_initial_state(self, method: Dro) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._reasoning == ""
        assert method._self_score == 0.0
        assert method._iteration == 0

    async def test_initialize(self, method: Dro) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._reasoning == ""
        assert method._self_score == 0.0
        assert method._iteration == 0

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = Dro()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._self_score = 0.95
        method._iteration = 3

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._self_score == 0.0
        assert method._iteration == 0

    async def test_health_check_not_initialized(self, method: Dro) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: Dro) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestDroProperties:
    """Tests for Dro property accessors."""

    def test_identifier_property(self, method: Dro) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.DRO

    def test_name_property(self, method: Dro) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "DRO"

    def test_description_property(self, method: Dro) -> None:
        """Test that description returns the correct method description."""
        assert "self-reward" in method.description.lower()
        assert "self-refine" in method.description.lower()

    def test_category_property(self, method: Dro) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestDroMetadata:
    """Tests for DRO metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert DRO_METADATA.identifier == MethodIdentifier.DRO

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert DRO_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"self-reward", "self-refine", "optimization", "autonomous"}
        assert expected_tags.issubset(DRO_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert DRO_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert DRO_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert DRO_METADATA.complexity == 6


class TestDroExecution:
    """Tests for Dro execute() method."""

    async def test_execute_basic(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.DRO

    async def test_execute_without_initialization_raises(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: Dro,
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
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets generate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "generate"
        assert thought.metadata.get("iteration") == 1

    async def test_execute_sets_iteration(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets iteration to 1."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert method._iteration == 1

    async def test_execute_with_execution_context(
        self,
        method: Dro,
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


class TestDroContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_generate_to_self_score(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from generate to self_score phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "self_score"
        assert continuation.type == ThoughtType.VERIFICATION

    async def test_continue_self_score_to_optimize(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from self_score to optimize phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        self_score = await method.continue_reasoning(session, initial)

        optimize = await method.continue_reasoning(session, self_score)

        assert optimize is not None
        assert optimize.metadata.get("phase") == "optimize"
        assert optimize.type == ThoughtType.REVISION

    async def test_continue_to_conclusion(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        self_score = await method.continue_reasoning(session, initial)
        optimize = await method.continue_reasoning(session, self_score)

        conclusion = await method.continue_reasoning(session, optimize)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: Dro,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DRO,
            content="Test",
            metadata={"phase": "generate"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestSelfScoring:
    """Tests for self-scoring functionality."""

    async def test_self_score_set_during_phase(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that self_score is set during self_score phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        # Self score should be set (heuristic default is 0.78)
        assert method._self_score > 0.0
        assert method._self_score <= 1.0

    async def test_self_score_heuristic_value(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that heuristic self-score uses expected value."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        # Heuristic sets score to 0.78
        assert method._self_score == 0.78


class TestSelfRefinement:
    """Tests for self-refinement/optimization functionality."""

    async def test_reasoning_updated_during_optimize(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that reasoning is updated during optimize phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        self_score = await method.continue_reasoning(session, initial)

        # Get initial reasoning

        await method.continue_reasoning(session, self_score)

        # Reasoning should be updated (heuristic adds PEMDAS note)
        assert "PEMDAS" in method._reasoning


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: Dro,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: Dro,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Calculate: " + "1 + " * 500 + "1"

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: Dro,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None

    async def test_unicode_in_problem(
        self,
        method: Dro,
        session: Session,
    ) -> None:
        """Test execution with Unicode characters."""
        await method.initialize()
        problem = "解决这个数学问题: 5 × 3 + 2 = ?"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "generate"

        # Continue through all phases
        self_score = await method.continue_reasoning(session, initial)
        assert self_score.metadata.get("phase") == "self_score"

        optimize = await method.continue_reasoning(session, self_score)
        assert optimize.metadata.get("phase") == "optimize"

        conclude = await method.continue_reasoning(session, optimize)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 4
        assert conclude.type == ThoughtType.CONCLUSION

    async def test_session_thought_count_updates(
        self,
        method: Dro,
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
        method: Dro,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought parent chain is correct."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        self_score = await method.continue_reasoning(session, initial)
        optimize = await method.continue_reasoning(session, self_score)

        assert initial.parent_id is None
        assert self_score.parent_id == initial.id
        assert optimize.parent_id == self_score.id
