"""Comprehensive tests for ChainOfVerification reasoning method.

This module provides complete test coverage for the ChainOfVerification method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Verification phases (baseline -> questions -> answers -> verified)
- Configuration options (num_questions)
- Continue reasoning flow
- Quality improvement tracking through phases
- Verification question and answer generation
- Phase transitions
- Edge cases

The tests aim for 90%+ coverage of the ChainOfVerification implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.chain_of_verification import (
    CHAIN_OF_VERIFICATION_METADATA,
    ChainOfVerification,
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
def method() -> ChainOfVerification:
    """Provide a ChainOfVerification method instance for testing.

    Returns:
        ChainOfVerification instance (uninitialized).
    """
    return ChainOfVerification()


@pytest.fixture
async def initialized_method() -> ChainOfVerification:
    """Provide an initialized ChainOfVerification method instance.

    Returns:
        Initialized ChainOfVerification instance.
    """
    method = ChainOfVerification()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.fixture
def factual_input() -> str:
    """Provide a factual test input suitable for verification.

    Returns:
        Factual question for testing.
    """
    return "What are the capital cities of Scandinavia?"


@pytest.fixture
def knowledge_input() -> str:
    """Provide a knowledge-based test input.

    Returns:
        Knowledge question requiring verification.
    """
    return "Who were the first five presidents of the United States?"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMetadata:
    """Test suite for ChainOfVerification metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert CHAIN_OF_VERIFICATION_METADATA.identifier == MethodIdentifier.CHAIN_OF_VERIFICATION

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert CHAIN_OF_VERIFICATION_METADATA.name == "Chain of Verification"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert CHAIN_OF_VERIFICATION_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert CHAIN_OF_VERIFICATION_METADATA.complexity == 5
        assert 1 <= CHAIN_OF_VERIFICATION_METADATA.complexity <= 10

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert CHAIN_OF_VERIFICATION_METADATA.supports_revision is True

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert CHAIN_OF_VERIFICATION_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "verification",
            "fact-checking",
            "hallucination-reduction",
            "multi-phase",
            "accuracy-focused",
        }
        assert expected_tags.issubset(CHAIN_OF_VERIFICATION_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert CHAIN_OF_VERIFICATION_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert CHAIN_OF_VERIFICATION_METADATA.max_thoughts == 15

    def test_metadata_description(self):
        """Test that metadata has comprehensive description."""
        assert len(CHAIN_OF_VERIFICATION_METADATA.description) > 50
        assert "verification" in CHAIN_OF_VERIFICATION_METADATA.description.lower()

    def test_metadata_best_for(self):
        """Test that metadata specifies appropriate use cases."""
        assert len(CHAIN_OF_VERIFICATION_METADATA.best_for) > 0
        assert any("fact" in use.lower() for use in CHAIN_OF_VERIFICATION_METADATA.best_for)

    def test_metadata_not_recommended_for(self):
        """Test that metadata specifies inappropriate use cases."""
        assert len(CHAIN_OF_VERIFICATION_METADATA.not_recommended_for) > 0


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test suite for ChainOfVerification initialization."""

    def test_create_method(self, method: ChainOfVerification):
        """Test creating a ChainOfVerification instance."""
        assert isinstance(method, ChainOfVerification)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: ChainOfVerification):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.CHAIN_OF_VERIFICATION
        assert method.name == "Chain of Verification"
        assert method.category == MethodCategory.SPECIALIZED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: ChainOfVerification):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "baseline"

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = ChainOfVerification()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "verified"

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._current_phase == "baseline"

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: ChainOfVerification):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: ChainOfVerification):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    def test_default_num_questions(self):
        """Test that DEFAULT_NUM_QUESTIONS is properly defined."""
        assert ChainOfVerification.DEFAULT_NUM_QUESTIONS == 3
        assert isinstance(ChainOfVerification.DEFAULT_NUM_QUESTIONS, int)

    def test_max_num_questions(self):
        """Test that MAX_NUM_QUESTIONS is properly defined."""
        assert ChainOfVerification.MAX_NUM_QUESTIONS == 10
        assert isinstance(ChainOfVerification.MAX_NUM_QUESTIONS, int)
        assert ChainOfVerification.MAX_NUM_QUESTIONS >= ChainOfVerification.DEFAULT_NUM_QUESTIONS


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestExecution:
    """Test suite for basic ChainOfVerification execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=factual_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CHAIN_OF_VERIFICATION
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_baseline_metadata(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == factual_input
        assert thought.metadata["phase"] == "baseline"
        assert "num_questions" in thought.metadata
        assert thought.metadata["reasoning_type"] == "chain_of_verification"
        assert "verification_questions" in thought.metadata
        assert "verification_answers" in thought.metadata

    @pytest.mark.asyncio
    async def test_execute_sets_initial_quality(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute sets initial quality score."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)

        assert thought.quality_score == 0.5
        assert thought.confidence == 0.5

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=factual_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.CHAIN_OF_VERIFICATION
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test execute with custom context."""
        context = {"num_questions": 5, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=factual_input, context=context
        )

        assert thought.metadata["num_questions"] == 5
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_num_questions_upper(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that num_questions is clamped to MAX_NUM_QUESTIONS."""
        thought = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": 20},
        )
        assert thought.metadata["num_questions"] == ChainOfVerification.MAX_NUM_QUESTIONS

    @pytest.mark.asyncio
    async def test_execute_clamps_num_questions_lower(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that num_questions is clamped to at least 1."""
        thought = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": 0},
        )
        assert thought.metadata["num_questions"] == 1

    @pytest.mark.asyncio
    async def test_execute_uses_default_num_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute uses default num_questions when not specified."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)
        assert thought.metadata["num_questions"] == ChainOfVerification.DEFAULT_NUM_QUESTIONS

    @pytest.mark.asyncio
    async def test_execute_initializes_empty_lists(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that execute initializes empty verification lists."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)
        assert thought.metadata["verification_questions"] == []
        assert thought.metadata["verification_answers"] == []


# ============================================================================
# Phase Transition Tests
# ============================================================================


class TestPhaseTransitions:
    """Test suite for phase transitions through the verification flow."""

    @pytest.mark.asyncio
    async def test_questions_phase_after_baseline(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that questions phase follows baseline."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert questions.type == ThoughtType.CONTINUATION
        assert questions.metadata["phase"] == "questions"
        assert questions.parent_id == baseline.id
        assert questions.step_number == 2

    @pytest.mark.asyncio
    async def test_answers_phase_after_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that answers phase follows questions."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        assert answers.type == ThoughtType.VERIFICATION
        assert answers.metadata["phase"] == "answers"
        assert answers.parent_id == questions.id
        assert answers.step_number == 3

    @pytest.mark.asyncio
    async def test_verified_phase_after_answers(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verified phase follows answers."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )

        assert verified.type == ThoughtType.CONCLUSION
        assert verified.metadata["phase"] == "verified"
        assert verified.parent_id == answers.id
        assert verified.step_number == 4

    @pytest.mark.asyncio
    async def test_full_verification_cycle(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test a complete verification cycle: baseline -> questions -> answers -> verified."""
        # Baseline
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        assert baseline.metadata["phase"] == "baseline"
        assert baseline.quality_score == 0.5

        # Questions
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.metadata["phase"] == "questions"
        assert questions.quality_score == 0.65

        # Answers
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers.metadata["phase"] == "answers"
        assert answers.quality_score == 0.8

        # Verified
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        assert verified.metadata["phase"] == "verified"
        assert verified.quality_score == 0.9

    @pytest.mark.asyncio
    async def test_phase_recorded_in_metadata(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that previous_phase is recorded in metadata."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert questions.metadata["previous_phase"] == "baseline"

    @pytest.mark.asyncio
    async def test_fallback_to_verified_for_unknown_phase(
        self, initialized_method: ChainOfVerification, session: Session
    ):
        """Test fallback to verified phase for unknown previous phase."""
        baseline = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        baseline.metadata["phase"] = "unknown_phase"

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert thought.type == ThoughtType.CONCLUSION
        assert thought.metadata["phase"] == "verified"


# ============================================================================
# Verification Questions Tests
# ============================================================================


class TestVerificationQuestions:
    """Test suite for verification question generation and handling."""

    @pytest.mark.asyncio
    async def test_questions_generated(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verification questions are generated."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert "verification_questions" in questions.metadata
        assert isinstance(questions.metadata["verification_questions"], list)
        assert len(questions.metadata["verification_questions"]) > 0

    @pytest.mark.asyncio
    async def test_num_questions_respected(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that the specified number of questions is generated."""
        num_questions = 5
        baseline = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": num_questions},
        )
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert len(questions.metadata["verification_questions"]) == num_questions

    @pytest.mark.asyncio
    async def test_questions_propagate_to_answers(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verification questions propagate to answers phase."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        assert (
            answers.metadata["verification_questions"]
            == questions.metadata["verification_questions"]
        )

    @pytest.mark.asyncio
    async def test_verification_answers_generated(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verification answers are generated."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        assert "verification_answers" in answers.metadata
        assert isinstance(answers.metadata["verification_answers"], list)
        assert len(answers.metadata["verification_answers"]) > 0

    @pytest.mark.asyncio
    async def test_answers_match_questions_count(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that number of answers matches number of questions."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        num_questions = len(answers.metadata["verification_questions"])
        num_answers = len(answers.metadata["verification_answers"])
        assert num_answers == num_questions

    @pytest.mark.asyncio
    async def test_answers_propagate_to_verified(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verification answers propagate to verified phase."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )

        assert verified.metadata["verification_answers"] == answers.metadata["verification_answers"]


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
            content="Test",
            metadata={"phase": "baseline"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        assert baseline.step_number == 1

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.step_number == 2

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)

        guidance_text = "Focus on geographic facts"
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline, guidance=guidance_text
        )

        assert "guidance" in questions.metadata
        assert questions.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test continue_reasoning with context parameter."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)

        context = {"additional_info": "test data"}
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline, context=context
        )

        assert "context" in questions.metadata
        assert questions.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        count_after_baseline = session.thought_count

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert session.thought_count == count_after_baseline + 1
        assert questions.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_continue_increments_depth(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that continue_reasoning increments depth."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        assert baseline.depth == 0

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.depth == 1

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers.depth == 2


# ============================================================================
# Quality Improvement Tests
# ============================================================================


class TestQualityImprovement:
    """Test suite for quality score improvement through phases."""

    @pytest.mark.asyncio
    async def test_quality_improves_through_phases(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that quality score improves with each phase."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        baseline_quality = baseline.quality_score

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        questions_quality = questions.quality_score

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        answers_quality = answers.quality_score

        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        verified_quality = verified.quality_score

        assert questions_quality > baseline_quality
        assert answers_quality > questions_quality
        assert verified_quality > answers_quality

    @pytest.mark.asyncio
    async def test_confidence_improves_through_phases(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that confidence improves with each phase."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        baseline_confidence = baseline.confidence

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        questions_confidence = questions.confidence

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        answers_confidence = answers.confidence

        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        verified_confidence = verified.confidence

        assert questions_confidence > baseline_confidence
        assert answers_confidence > questions_confidence
        assert verified_confidence > answers_confidence

    @pytest.mark.asyncio
    async def test_baseline_quality_score(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test baseline phase quality score."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        assert baseline.quality_score == 0.5

    @pytest.mark.asyncio
    async def test_questions_quality_score(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test questions phase quality score."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.quality_score == 0.65

    @pytest.mark.asyncio
    async def test_answers_quality_score(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test answers phase quality score."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers.quality_score == 0.8

    @pytest.mark.asyncio
    async def test_verified_quality_score(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test verified phase quality score."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        assert verified.quality_score == 0.9


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: ChainOfVerification, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: ChainOfVerification, session: Session):
        """Test handling of very long input."""
        long_input = "What are " + "the facts about " * 100 + "this topic?"
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=factual_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=factual_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_minimal_num_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test with minimal number of questions (1)."""
        baseline = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": 1},
        )
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert len(questions.metadata["verification_questions"]) == 1

    @pytest.mark.asyncio
    async def test_maximum_num_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test with maximum number of questions."""
        baseline = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": ChainOfVerification.MAX_NUM_QUESTIONS},
        )
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert (
            len(questions.metadata["verification_questions"])
            == ChainOfVerification.MAX_NUM_QUESTIONS
        )

    @pytest.mark.asyncio
    async def test_negative_num_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that negative num_questions is clamped to 1."""
        thought = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": -5},
        )
        assert thought.metadata["num_questions"] == 1

    @pytest.mark.asyncio
    async def test_missing_verification_questions_in_metadata(
        self, initialized_method: ChainOfVerification, session: Session
    ):
        """Test handling when verification_questions is missing from metadata."""
        baseline = await initialized_method.execute(session=session, input_text="Test")

        # Remove verification_questions from metadata
        del baseline.metadata["verification_questions"]

        # Should handle gracefully with empty list
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert isinstance(questions.metadata["verification_questions"], list)

    @pytest.mark.asyncio
    async def test_missing_num_questions_in_metadata(
        self, initialized_method: ChainOfVerification, session: Session
    ):
        """Test handling when num_questions is missing from metadata."""
        baseline = await initialized_method.execute(session=session, input_text="Test")

        # Remove num_questions from metadata
        del baseline.metadata["num_questions"]

        # Should use default
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.metadata["num_questions"] == ChainOfVerification.DEFAULT_NUM_QUESTIONS

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_execute(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that step counter resets on new execute."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=factual_input)
        assert thought1.step_number == 1

        # Continue
        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )
        assert thought2.step_number == 2

        # New execution should reset
        session2 = Session().start()
        thought3 = await initialized_method.execute(session=session2, input_text=factual_input)
        assert thought3.step_number == 1


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_baseline_content_structure(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that baseline response has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=factual_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Baseline Response" in content
        assert factual_input in content

    @pytest.mark.asyncio
    async def test_questions_content_structure(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that questions have expected content structure."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        content = questions.content
        assert isinstance(content, str)
        assert "Verification Questions" in content or "verification questions" in content.lower()

    @pytest.mark.asyncio
    async def test_answers_content_structure(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that answers have expected content structure."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        content = answers.content
        assert isinstance(content, str)
        assert "Verification Answers" in content or "verification answers" in content.lower()

    @pytest.mark.asyncio
    async def test_verified_content_structure(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that verified response has expected content structure."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )

        content = verified.content
        assert isinstance(content, str)
        assert "Verified Response" in content or "verified response" in content.lower()

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that guidance appears in metadata."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)

        guidance = "Focus on capital cities only"
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline, guidance=guidance
        )

        assert guidance in questions.metadata["guidance"]

    @pytest.mark.asyncio
    async def test_phase_indicators_in_content(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that phase indicators appear in content."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        # Should indicate phase progression
        assert "Phase" in questions.content or "phase" in questions.content


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test suite for integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self, initialized_method: ChainOfVerification, session: Session, knowledge_input: str
    ):
        """Test complete workflow from baseline to verified."""
        # Execute baseline
        baseline = await initialized_method.execute(session=session, input_text=knowledge_input)
        assert baseline is not None
        assert session.thought_count == 1

        # Generate questions
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions is not None
        assert session.thought_count == 2

        # Generate answers
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers is not None
        assert session.thought_count == 3

        # Generate verified response
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        assert verified is not None
        assert session.thought_count == 4
        assert verified.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_workflow_with_custom_num_questions(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test workflow with custom number of verification questions."""
        num_questions = 7

        baseline = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": num_questions},
        )

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )

        assert len(questions.metadata["verification_questions"]) == num_questions

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )

        assert len(answers.metadata["verification_answers"]) == num_questions

    @pytest.mark.asyncio
    async def test_thought_graph_structure(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that thought graph has correct structure."""
        baseline = await initialized_method.execute(session=session, input_text=factual_input)
        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )

        # Check graph structure
        assert session.graph.root_id == baseline.id
        assert baseline.id in session.graph.nodes
        assert questions.id in session.graph.nodes
        assert answers.id in session.graph.nodes
        assert verified.id in session.graph.nodes

        # Check parent-child relationships
        assert questions.parent_id == baseline.id
        assert answers.parent_id == questions.id
        assert verified.parent_id == answers.id

    @pytest.mark.asyncio
    async def test_metadata_propagation(
        self, initialized_method: ChainOfVerification, session: Session, factual_input: str
    ):
        """Test that metadata propagates correctly through phases."""
        num_questions = 4
        baseline = await initialized_method.execute(
            session=session,
            input_text=factual_input,
            context={"num_questions": num_questions},
        )

        questions = await initialized_method.continue_reasoning(
            session=session, previous_thought=baseline
        )
        assert questions.metadata["num_questions"] == num_questions

        answers = await initialized_method.continue_reasoning(
            session=session, previous_thought=questions
        )
        assert answers.metadata["num_questions"] == num_questions

        verified = await initialized_method.continue_reasoning(
            session=session, previous_thought=answers
        )
        assert verified.metadata["num_questions"] == num_questions
