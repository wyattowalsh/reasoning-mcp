"""Unit tests for OutcomeRewardModel reasoning method.

This module provides comprehensive unit tests for the OutcomeRewardModel class,
testing initialization, execution, phase transitions, scoring mechanisms,
acceptance/rejection logic, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.outcome_reward_model import (
    OUTCOME_REWARD_MODEL_METADATA,
    OutcomeRewardModel,
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
def method() -> OutcomeRewardModel:
    """Create an OutcomeRewardModel instance for testing."""
    return OutcomeRewardModel()


@pytest.fixture
async def initialized_method() -> OutcomeRewardModel:
    """Create and initialize an OutcomeRewardModel instance."""
    method = OutcomeRewardModel()
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
    ctx.sample = AsyncMock(return_value={"content": "Generated solution for the problem"})
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestOutcomeRewardModelMetadata:
    """Tests for OutcomeRewardModel metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert OUTCOME_REWARD_MODEL_METADATA.identifier == MethodIdentifier.OUTCOME_REWARD_MODEL

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert OUTCOME_REWARD_MODEL_METADATA.name == "Outcome Reward Model"

    def test_metadata_category(self) -> None:
        """Test metadata is in SPECIALIZED category."""
        assert OUTCOME_REWARD_MODEL_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates no branching support."""
        assert OUTCOME_REWARD_MODEL_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert OUTCOME_REWARD_MODEL_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test metadata indicates no context required."""
        assert OUTCOME_REWARD_MODEL_METADATA.requires_context is False

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "outcome" in OUTCOME_REWARD_MODEL_METADATA.tags
        assert "reward-model" in OUTCOME_REWARD_MODEL_METADATA.tags
        assert "verification" in OUTCOME_REWARD_MODEL_METADATA.tags
        assert "orm" in OUTCOME_REWARD_MODEL_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert OUTCOME_REWARD_MODEL_METADATA.complexity == 5

    def test_metadata_min_thoughts(self) -> None:
        """Test metadata specifies minimum thoughts."""
        assert OUTCOME_REWARD_MODEL_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self) -> None:
        """Test metadata specifies maximum thoughts."""
        assert OUTCOME_REWARD_MODEL_METADATA.max_thoughts == 6


# ============================================================================
# Initialization Tests
# ============================================================================


class TestOutcomeRewardModelInitialization:
    """Tests for OutcomeRewardModel initialization."""

    def test_default_initialization(self, method: OutcomeRewardModel) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "solve"
        assert method._candidate_solution == ""
        assert method._outcome_score == 0.0
        assert method._accepted is False

    async def test_initialize_method(self, method: OutcomeRewardModel) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "solve"

    async def test_health_check_before_initialize(self, method: OutcomeRewardModel) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestOutcomeRewardModelProperties:
    """Tests for OutcomeRewardModel properties."""

    def test_identifier_property(self, method: OutcomeRewardModel) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.OUTCOME_REWARD_MODEL

    def test_name_property(self, method: OutcomeRewardModel) -> None:
        """Test name property returns correct value."""
        assert method.name == "Outcome Reward Model"

    def test_description_property(self, method: OutcomeRewardModel) -> None:
        """Test description property returns correct value."""
        assert "outcome" in method.description.lower()
        assert "reward" in method.description.lower()

    def test_category_property(self, method: OutcomeRewardModel) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestOutcomeRewardModelExecution:
    """Tests for basic execution of OutcomeRewardModel method."""

    async def test_execute_without_initialization_fails(
        self, method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "What is 2+2?")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.OUTCOME_REWARD_MODEL

    async def test_execute_sets_step_number(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test problem")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() sets phase to solve."""
        result = await initialized_method.execute(active_session, "Test problem")
        assert result.metadata["phase"] == "solve"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test problem")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test problem")
        assert active_session.current_method == MethodIdentifier.OUTCOME_REWARD_MODEL

    async def test_execute_content_includes_problem(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execute() content includes the problem text."""
        problem = "Calculate the area of a circle"
        result = await initialized_method.execute(active_session, problem)
        assert problem in result.content

    async def test_execute_with_sampling(
        self,
        initialized_method: OutcomeRewardModel,
        active_session: Session,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test execute() with sampling capability."""
        result = await initialized_method.execute(
            active_session, "Test problem", execution_context=mock_execution_context
        )
        assert result is not None
        assert result.type == ThoughtType.INITIAL


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestOutcomeRewardModelContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.OUTCOME_REWARD_MODEL,
            content="Test",
            metadata={"phase": "solve"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_solve_to_score(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test transition from solve to score phase."""
        initial = await initialized_method.execute(active_session, "Test problem")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "score"
        assert result.type == ThoughtType.VERIFICATION
        assert result.parent_id == initial.id

    async def test_continue_from_score_to_decide(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test transition from score to decide phase."""
        initial = await initialized_method.execute(active_session, "Test problem")
        score_thought = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, score_thought)

        assert result.metadata["phase"] == "decide"
        assert result.type == ThoughtType.REASONING

    async def test_continue_from_decide_to_conclude(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test transition from decide to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test problem")
        score = await initialized_method.continue_reasoning(active_session, initial)
        decide = await initialized_method.continue_reasoning(active_session, score)
        result = await initialized_method.continue_reasoning(active_session, decide)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test problem")
        assert initial.step_number == 1

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test problem")
        assert initial.depth == 0

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == 1


# ============================================================================
# Scoring Tests
# ============================================================================


class TestOutcomeRewardModelScoring:
    """Tests for scoring mechanisms."""

    async def test_score_threshold_constant(self) -> None:
        """Test SCORE_THRESHOLD constant value."""
        assert OutcomeRewardModel.SCORE_THRESHOLD == 0.75

    async def test_calculate_fallback_score_empty_solution(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test fallback score for empty solution."""
        score = initialized_method._calculate_fallback_score("")
        assert score == 0.65

    async def test_calculate_fallback_score_generated_placeholder(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test fallback score for generated placeholder solution."""
        score = initialized_method._calculate_fallback_score("[Generated solution]")
        assert score == 0.65

    async def test_calculate_fallback_score_detailed_solution(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test fallback score for detailed solution with structure."""
        detailed = "This is a detailed solution with multiple parts.\nIt has structure and steps."
        score = initialized_method._calculate_fallback_score(detailed)
        assert score > 0.7  # Should be higher than base score

    async def test_calculate_fallback_score_max_cap(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test fallback score is capped at 0.95."""
        very_long = "x" * 1000 + "\n" * 10 + "." * 100
        score = initialized_method._calculate_fallback_score(very_long)
        assert score <= 0.95

    async def test_parse_orm_scores_valid_scores(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test parsing valid ORM scores."""
        score_text = "0.89\n0.92\n0.81"
        result = initialized_method._parse_orm_scores(score_text)
        expected = (0.89 + 0.92 + 0.81) / 3
        assert abs(result - expected) < 0.01

    async def test_parse_orm_scores_no_scores_found(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test parsing when no scores found."""
        result = initialized_method._parse_orm_scores("No scores here")
        assert result == 0.7  # Default fallback

    async def test_parse_orm_scores_partial_scores(
        self, initialized_method: OutcomeRewardModel
    ) -> None:
        """Test parsing partial scores."""
        score_text = "0.8"
        result = initialized_method._parse_orm_scores(score_text)
        assert abs(result - 0.8) < 0.01


# ============================================================================
# Acceptance Logic Tests
# ============================================================================


class TestOutcomeRewardModelAcceptance:
    """Tests for acceptance/rejection logic."""

    async def test_accepted_when_score_above_threshold(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test solution is accepted when score >= threshold."""
        initial = await initialized_method.execute(active_session, "Test")
        score = await initialized_method.continue_reasoning(active_session, initial)

        # Set high score manually
        initialized_method._outcome_score = 0.85
        decide = await initialized_method.continue_reasoning(active_session, score)

        assert initialized_method._accepted is True
        assert "ACCEPTED" in decide.content

    async def test_rejected_when_score_below_threshold(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test solution is rejected when score < threshold."""
        # Create a custom scenario by manipulating internal state
        await initialized_method.execute(active_session, "Test")

        # Manually set low score for testing
        initialized_method._outcome_score = 0.5

        score_thought = ThoughtNode(
            id="test-score",
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.OUTCOME_REWARD_MODEL,
            content="Score phase",
            metadata={"phase": "score"},
        )
        active_session.add_thought(score_thought)

        result = await initialized_method.continue_reasoning(active_session, score_thought)

        assert initialized_method._accepted is False
        assert "REJECTED" in result.content

    async def test_conclusion_reflects_acceptance(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test conclusion includes acceptance status."""
        initial = await initialized_method.execute(active_session, "Test")
        score = await initialized_method.continue_reasoning(active_session, initial)
        decide = await initialized_method.continue_reasoning(active_session, score)
        conclusion = await initialized_method.continue_reasoning(active_session, decide)

        assert "accepted" in conclusion.metadata or "Status:" in conclusion.content


# ============================================================================
# Fallback Solution Tests
# ============================================================================


class TestOutcomeRewardModelFallback:
    """Tests for fallback solution generation."""

    async def test_generate_fallback_solution(self, initialized_method: OutcomeRewardModel) -> None:
        """Test fallback solution generation."""
        input_text = "What is the capital of France?"
        result = initialized_method._generate_fallback_solution(input_text)

        assert "[Generated solution for:" in result
        assert "What is the capital" in result


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestOutcomeRewardModelEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "A" * 5000
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Test: @#$%^&*() 测试 émojis 🎯"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None

    async def test_none_context(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test execution with None context."""
        result = await initialized_method.execute(active_session, "Test", context=None)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestOutcomeRewardModelWorkflow:
    """Tests for complete ORM workflows."""

    async def test_full_workflow(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test complete ORM workflow from solve to conclusion."""
        # Solve phase
        initial = await initialized_method.execute(active_session, "What is 2+2?")
        assert initial.metadata["phase"] == "solve"
        assert initial.type == ThoughtType.INITIAL

        # Score phase
        score = await initialized_method.continue_reasoning(active_session, initial)
        assert score.metadata["phase"] == "score"
        assert score.type == ThoughtType.VERIFICATION

        # Decide phase
        decide = await initialized_method.continue_reasoning(active_session, score)
        assert decide.metadata["phase"] == "decide"
        assert decide.type == ThoughtType.REASONING

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, decide)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 4
        assert active_session.current_method == MethodIdentifier.OUTCOME_REWARD_MODEL

    async def test_confidence_progression(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test that confidence changes through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        score = await initialized_method.continue_reasoning(active_session, initial)
        await initialized_method.continue_reasoning(active_session, score)

        # Initial confidence should be moderate
        assert initial.confidence == 0.6

        # Score phase should have higher confidence
        assert score.confidence == 0.75

    async def test_metadata_propagation(
        self, initialized_method: OutcomeRewardModel, active_session: Session
    ) -> None:
        """Test that metadata is properly propagated."""
        initial = await initialized_method.execute(active_session, "Test")
        score = await initialized_method.continue_reasoning(active_session, initial)
        decide = await initialized_method.continue_reasoning(active_session, score)
        conclusion = await initialized_method.continue_reasoning(active_session, decide)

        # Conclusion should have score and acceptance metadata
        assert "score" in conclusion.metadata
        assert "accepted" in conclusion.metadata
