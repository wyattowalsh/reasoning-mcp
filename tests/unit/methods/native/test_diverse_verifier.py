"""Unit tests for DiVeRSe (Diverse Verifier on Reasoning Steps) reasoning method.

This module provides comprehensive tests for the DiverseVerifier method implementation,
covering initialization, execution, diverse path sampling, step-level verification,
voting aggregation, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.diverse_verifier import (
    DIVERSE_VERIFIER_METADATA,
    DiverseVerifier,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> DiverseVerifier:
    """Create a DiverseVerifier method instance for testing.

    Returns:
        A fresh DiverseVerifier instance
    """
    return DiverseVerifier()


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
    return "If x=5, y=3, and z=2, what is x × y + z?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Reasoning path with answer: 17")
    return ctx


class TestDiverseVerifierInitialization:
    """Tests for DiverseVerifier initialization and setup."""

    def test_create_method(self, method: DiverseVerifier) -> None:
        """Test that DiverseVerifier can be instantiated."""
        assert method is not None
        assert isinstance(method, DiverseVerifier)

    def test_initial_state(self, method: DiverseVerifier) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "sample"

    async def test_initialize(self, method: DiverseVerifier) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "sample"
        assert method._diverse_paths == []
        assert method._verified_paths == []
        assert method._vote_results == {}
        assert method._final_answer is None

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = DiverseVerifier()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._final_answer = "test"

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "sample"
        assert method._final_answer is None

    async def test_health_check_not_initialized(self, method: DiverseVerifier) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: DiverseVerifier) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestDiverseVerifierProperties:
    """Tests for DiverseVerifier property accessors."""

    def test_identifier_property(self, method: DiverseVerifier) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.DIVERSE_VERIFIER

    def test_name_property(self, method: DiverseVerifier) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "DiVeRSe"

    def test_description_property(self, method: DiverseVerifier) -> None:
        """Test that description returns the correct method description."""
        assert "diverse" in method.description.lower()
        assert "verifier" in method.description.lower()

    def test_category_property(self, method: DiverseVerifier) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestDiverseVerifierMetadata:
    """Tests for DIVERSE_VERIFIER metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert DIVERSE_VERIFIER_METADATA.identifier == MethodIdentifier.DIVERSE_VERIFIER

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert DIVERSE_VERIFIER_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"diversity", "verification", "voting", "sampling"}
        assert expected_tags.issubset(DIVERSE_VERIFIER_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert DIVERSE_VERIFIER_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert DIVERSE_VERIFIER_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert DIVERSE_VERIFIER_METADATA.complexity == 6


class TestDiverseVerifierExecution:
    """Tests for DiverseVerifier execute() method."""

    async def test_execute_basic(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.DIVERSE_VERIFIER

    async def test_execute_without_initialization_raises(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_sample_phase(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets sample phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "sample"
        assert "paths" in thought.metadata

    async def test_execute_generates_diverse_paths(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates diverse paths."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert len(method._diverse_paths) > 0
        for path in method._diverse_paths:
            assert "id" in path
            assert "prompt_style" in path
            assert "reasoning" in path
            assert "answer" in path
            assert "steps" in path

    async def test_execute_with_execution_context(
        self,
        method: DiverseVerifier,
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


class TestDiverseVerifierContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_sample_to_verify(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from sample to verify phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "verify"
        assert continuation.type == ThoughtType.VERIFICATION

    async def test_continue_verify_to_vote(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from verify to vote phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)

        vote = await method.continue_reasoning(session, verify)

        assert vote is not None
        assert vote.metadata.get("phase") == "vote"
        assert vote.type == ThoughtType.SYNTHESIS

    async def test_continue_vote_to_select(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from vote to select phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)
        vote = await method.continue_reasoning(session, verify)

        select = await method.continue_reasoning(session, vote)

        assert select is not None
        assert select.metadata.get("phase") == "select"
        assert select.type == ThoughtType.REASONING

    async def test_continue_to_conclusion(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)
        vote = await method.continue_reasoning(session, verify)
        select = await method.continue_reasoning(session, vote)

        conclusion = await method.continue_reasoning(session, select)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: DiverseVerifier,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIVERSE_VERIFIER,
            content="Test",
            metadata={"phase": "sample"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestStepLevelVerification:
    """Tests for step-level verification."""

    async def test_verified_paths_generated(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that verified paths are generated during verify phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert len(method._verified_paths) > 0
        for verified in method._verified_paths:
            assert "id" in verified
            assert "step_scores" in verified
            assert "avg_score" in verified
            assert "valid" in verified

    async def test_verification_threshold(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that verification uses correct threshold."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        # Check that valid paths have avg_score >= 0.70
        for verified in method._verified_paths:
            if verified["valid"]:
                assert verified["avg_score"] >= 0.70
            else:
                assert verified["avg_score"] < 0.70


class TestVotingAggregation:
    """Tests for voting aggregation."""

    async def test_vote_results_generated(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that vote results are generated during vote phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, verify)

        assert "answers" in method._vote_results
        assert "winner" in method._vote_results
        assert "confidence" in method._vote_results

    async def test_winner_has_most_votes(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that winner is the answer with most votes."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)
        await method.continue_reasoning(session, verify)

        winner = method._vote_results["winner"]
        winner_count = method._vote_results["answers"][winner]["count"]

        # Winner should have highest or tied highest count
        for _ans, data in method._vote_results["answers"].items():
            assert winner_count >= data["count"]


class TestFinalSelection:
    """Tests for final answer selection."""

    async def test_final_answer_set(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that final answer is set after select phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        verify = await method.continue_reasoning(session, initial)
        vote = await method.continue_reasoning(session, verify)
        await method.continue_reasoning(session, vote)

        assert method._final_answer is not None
        assert method._final_answer == method._vote_results["winner"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: DiverseVerifier,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: DiverseVerifier,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Solve: " + "x + " * 500 + "y = 0"

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: DiverseVerifier,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "sample"

        # Continue through all phases
        verify = await method.continue_reasoning(session, initial)
        assert verify.metadata.get("phase") == "verify"

        vote = await method.continue_reasoning(session, verify)
        assert vote.metadata.get("phase") == "vote"

        select = await method.continue_reasoning(session, vote)
        assert select.metadata.get("phase") == "select"

        conclude = await method.continue_reasoning(session, select)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION
        assert method._final_answer is not None

    async def test_session_thought_count_updates(
        self,
        method: DiverseVerifier,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1
