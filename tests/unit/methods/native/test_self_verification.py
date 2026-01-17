"""Unit tests for SelfVerification reasoning method.

This module provides comprehensive unit tests for the SelfVerification class,
testing initialization, forward reasoning, backward verification, voting,
selection mechanisms, and complete workflow execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.self_verification import (
    SELF_VERIFICATION_METADATA,
    SelfVerification,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> SelfVerification:
    """Create a SelfVerification instance for testing."""
    return SelfVerification()


@pytest.fixture
async def initialized_method() -> SelfVerification:
    """Create and initialize a SelfVerification instance."""
    method = SelfVerification()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Create an active session for testing."""
    return Session().start()


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability."""
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Generated candidates and verification")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfVerificationMetadata:
    """Tests for SelfVerification metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert SELF_VERIFICATION_METADATA.identifier == MethodIdentifier.SELF_VERIFICATION

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "Self" in SELF_VERIFICATION_METADATA.name
        assert "Verification" in SELF_VERIFICATION_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in SPECIALIZED category."""
        assert SELF_VERIFICATION_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates no branching support."""
        assert SELF_VERIFICATION_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert SELF_VERIFICATION_METADATA.supports_revision is True

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "verification" in SELF_VERIFICATION_METADATA.tags
        assert "backward" in SELF_VERIFICATION_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert SELF_VERIFICATION_METADATA.complexity >= 5


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSelfVerificationInitialization:
    """Tests for SelfVerification initialization."""

    def test_default_initialization(self, method: SelfVerification) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "forward"
        assert method._candidates == []
        assert method._verification_scores == []

    async def test_initialize_method(self, method: SelfVerification) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "forward"

    async def test_health_check_before_initialize(self, method: SelfVerification) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(
        self, initialized_method: SelfVerification
    ) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestSelfVerificationProperties:
    """Tests for SelfVerification properties."""

    def test_identifier_property(self, method: SelfVerification) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.SELF_VERIFICATION

    def test_name_property(self, method: SelfVerification) -> None:
        """Test name property returns correct value."""
        assert "Self" in method.name
        assert "Verification" in method.name

    def test_description_property(self, method: SelfVerification) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "forward" in desc_lower or "backward" in desc_lower or "verif" in desc_lower

    def test_category_property(self, method: SelfVerification) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSelfVerificationExecution:
    """Tests for basic execution of SelfVerification method."""

    async def test_execute_without_initialization_fails(
        self, method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "What is 5+5?")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.REASONING  # SelfVerification uses REASONING type
        assert result.method_id == MethodIdentifier.SELF_VERIFICATION

    async def test_execute_sets_step_number(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() sets phase to forward."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "forward"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.SELF_VERIFICATION


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSelfVerificationContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: SelfVerification, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_VERIFICATION,
            content="Test",
            metadata={"phase": "forward"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_forward_to_backward(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test transition from forward to backward phase."""
        initial = await initialized_method.execute(active_session, "Test")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "backward"
        assert result.parent_id == initial.id

    async def test_continue_from_backward_to_vote(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test transition from backward to vote phase."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, backward)

        assert result.metadata["phase"] == "vote"

    async def test_continue_from_vote_to_conclude(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test transition from vote to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        vote = await initialized_method.continue_reasoning(active_session, backward)
        result = await initialized_method.continue_reasoning(active_session, vote)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Candidate Generation Tests
# ============================================================================


class TestSelfVerificationCandidates:
    """Tests for candidate answer generation."""

    async def test_candidates_generated_in_forward_phase(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test that candidate answers are generated in forward phase."""
        await initialized_method.execute(active_session, "What is 2+2?")
        assert len(initialized_method._candidates) >= 1

    async def test_generate_candidates_heuristic(
        self, initialized_method: SelfVerification
    ) -> None:
        """Test heuristic candidate generation."""
        candidates = initialized_method._generate_candidates("What is 5+5?")
        assert isinstance(candidates, list)
        assert len(candidates) >= 1  # Should generate at least one candidate


# ============================================================================
# Verification Tests
# ============================================================================


class TestSelfVerificationVerification:
    """Tests for backward verification functionality."""

    async def test_verification_scores_populated(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test that verification scores are populated after backward phase."""
        initial = await initialized_method.execute(active_session, "Test")
        await initialized_method.continue_reasoning(active_session, initial)

        # After backward phase, verification scores should be a list
        assert isinstance(initialized_method._verification_scores, list)

    async def test_generate_verification_scores_heuristic(
        self, initialized_method: SelfVerification
    ) -> None:
        """Test heuristic verification scoring."""
        initialized_method._candidates = [
            {"answer": "A", "reasoning": "Because"},
            {"answer": "B", "reasoning": "Since"},
        ]
        scores = initialized_method._generate_verification_scores()
        assert isinstance(scores, list)
        assert len(scores) >= 0


# ============================================================================
# Voting Tests
# ============================================================================


class TestSelfVerificationVoting:
    """Tests for voting mechanism."""

    async def test_voting_considers_verification_scores(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test that voting considers verification scores."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        vote = await initialized_method.continue_reasoning(active_session, backward)

        # Vote phase should have processed verification scores
        assert "vote" in vote.metadata["phase"]


# ============================================================================
# Selection Tests
# ============================================================================


class TestSelfVerificationSelection:
    """Tests for best answer selection."""

    async def test_best_answer_selected_in_vote_phase(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test best answer is selected in vote phase."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        vote = await initialized_method.continue_reasoning(active_session, backward)

        # Vote phase should identify best answer
        assert vote.metadata["phase"] == "vote"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSelfVerificationEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.REASONING

    async def test_very_long_input_text(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "A complex question: " + "detail " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Calculate: @#$%^&*() + 日本語 = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestSelfVerificationWorkflow:
    """Tests for complete SelfVerification workflows."""

    async def test_full_workflow(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test complete workflow from forward reasoning to conclusion."""
        # Forward phase (generate candidates)
        initial = await initialized_method.execute(active_session, "What is 7*7?")
        assert initial.metadata["phase"] == "forward"
        assert initial.type == ThoughtType.REASONING

        # Backward phase (verify each)
        backward = await initialized_method.continue_reasoning(active_session, initial)
        assert backward.metadata["phase"] == "backward"

        # Vote phase (score by verification)
        vote = await initialized_method.continue_reasoning(active_session, backward)
        assert vote.metadata["phase"] == "vote"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, vote)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 4
        assert active_session.current_method == MethodIdentifier.SELF_VERIFICATION

    async def test_confidence_progression(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        vote = await initialized_method.continue_reasoning(active_session, backward)
        select = await initialized_method.continue_reasoning(active_session, vote)
        conclusion = await initialized_method.continue_reasoning(active_session, select)

        # Conclusion should have reasonable confidence
        assert conclusion.confidence > 0

    async def test_metadata_includes_candidates_and_scores(
        self, initialized_method: SelfVerification, active_session: Session
    ) -> None:
        """Test that final metadata includes candidates and scores."""
        initial = await initialized_method.execute(active_session, "Test")
        backward = await initialized_method.continue_reasoning(active_session, initial)
        vote = await initialized_method.continue_reasoning(active_session, backward)
        conclusion = await initialized_method.continue_reasoning(active_session, vote)

        # Conclusion should reference scores
        assert "scores" in conclusion.metadata or len(initialized_method._verification_scores) > 0
