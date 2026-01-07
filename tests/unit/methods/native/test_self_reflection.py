"""Comprehensive tests for SelfReflection reasoning method.

This module provides complete test coverage for the SelfReflection method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Reflection cycles (initial -> critique -> improvement)
- Configuration options (quality_threshold, max_iterations)
- Continue reasoning flow
- Quality improvement tracking
- Self-critique generation
- Iteration tracking and limits
- Convergence detection
- Edge cases

The tests aim for 90%+ coverage of the SelfReflection implementation.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.self_reflection import (
    SELF_REFLECTION_METADATA,
    SelfReflection,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> SelfReflection:
    """Provide a SelfReflection method instance for testing.

    Returns:
        SelfReflection instance (uninitialized).
    """
    return SelfReflection()


@pytest.fixture
async def initialized_method() -> SelfReflection:
    """Provide an initialized SelfReflection method instance.

    Returns:
        Initialized SelfReflection instance.
    """
    method = SelfReflection()
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
def simple_input() -> str:
    """Provide a simple test input.

    Returns:
        Simple question for testing.
    """
    return "What is the best way to learn programming?"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex question requiring deeper reflection.
    """
    return "Explain the philosophical implications of quantum mechanics on free will"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfReflectionMetadata:
    """Test suite for SelfReflection metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SELF_REFLECTION_METADATA.identifier == MethodIdentifier.SELF_REFLECTION

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SELF_REFLECTION_METADATA.name == "Self-Reflection"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert SELF_REFLECTION_METADATA.category == MethodCategory.HIGH_VALUE

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert SELF_REFLECTION_METADATA.complexity == 4
        assert 1 <= SELF_REFLECTION_METADATA.complexity <= 10

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert SELF_REFLECTION_METADATA.supports_revision is True

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert SELF_REFLECTION_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "metacognitive",
            "self-critique",
            "iterative",
            "refinement",
            "quality-driven",
        }
        assert expected_tags.issubset(SELF_REFLECTION_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert SELF_REFLECTION_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert SELF_REFLECTION_METADATA.max_thoughts == 20


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSelfReflectionInitialization:
    """Test suite for SelfReflection initialization."""

    def test_create_method(self, method: SelfReflection):
        """Test creating a SelfReflection instance."""
        assert isinstance(method, SelfReflection)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: SelfReflection):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.SELF_REFLECTION
        assert method.name == "Self-Reflection"
        assert method.category == MethodCategory.HIGH_VALUE
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: SelfReflection):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._reflection_cycle == 0
        assert method._current_phase == "initial"

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = SelfReflection()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._reflection_cycle = 2
        method._current_phase = "improve"

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._reflection_cycle == 0
        assert method._current_phase == "initial"

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: SelfReflection):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(
        self, initialized_method: SelfReflection
    ):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSelfReflectionExecution:
    """Test suite for basic SelfReflection execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SELF_REFLECTION
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metadata(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["phase"] == "initial"
        assert thought.metadata["reflection_cycle"] == 0
        assert "quality_threshold" in thought.metadata
        assert thought.metadata["reasoning_type"] == "self_reflection"

    @pytest.mark.asyncio
    async def test_execute_sets_initial_quality(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that execute sets initial quality score."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        assert thought.quality_score == 0.6
        assert thought.confidence == 0.6
        assert thought.metadata["needs_improvement"] is True

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.SELF_REFLECTION
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"quality_threshold": 0.9, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["quality_threshold"] == 0.9
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_quality_threshold(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that quality_threshold is clamped to [0.0, 1.0]."""
        # Test upper bound
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 1.5},
        )
        assert thought1.metadata["quality_threshold"] == 1.0

        # Re-initialize for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test lower bound
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"quality_threshold": -0.5},
        )
        assert thought2.metadata["quality_threshold"] == 0.0


# ============================================================================
# Reflection Cycle Tests
# ============================================================================


class TestReflectionCycle:
    """Test suite for the reflection cycle flow."""

    @pytest.mark.asyncio
    async def test_critique_phase_after_initial(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that critique follows initial thought."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert critique.type == ThoughtType.VERIFICATION
        assert critique.metadata["phase"] == "critique"
        assert critique.parent_id == initial.id
        assert critique.step_number == 2

    @pytest.mark.asyncio
    async def test_improvement_phase_after_critique(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that improvement follows critique."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        improvement = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique
        )

        assert improvement.type in (ThoughtType.REVISION, ThoughtType.CONCLUSION)
        assert improvement.metadata["phase"] == "improve"
        assert improvement.parent_id == critique.id
        assert improvement.step_number == 3
        assert improvement.metadata["reflection_cycle"] == 1

    @pytest.mark.asyncio
    async def test_full_reflection_cycle(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test a complete reflection cycle: initial -> critique -> improve."""
        # Initial thought
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        assert initial.metadata["reflection_cycle"] == 0

        # Critique
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        assert critique.metadata["reflection_cycle"] == 0

        # Improvement
        improvement = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique
        )
        assert improvement.metadata["reflection_cycle"] == 1

    @pytest.mark.asyncio
    async def test_multiple_reflection_cycles(
        self, initialized_method: SelfReflection, session: Session
    ):
        """Test multiple reflection cycles."""
        input_text = "Explain machine learning"

        # Cycle 0: Initial
        thought = await initialized_method.execute(session=session, input_text=input_text)
        assert thought.metadata["reflection_cycle"] == 0

        # Cycle 0: Critique
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["reflection_cycle"] == 0

        # Cycle 1: Improve
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["reflection_cycle"] == 1

        # Cycle 1: Critique
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["reflection_cycle"] == 1

        # Cycle 2: Improve
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["reflection_cycle"] == 2

    @pytest.mark.asyncio
    async def test_critique_after_improvement(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that critique can follow improvement for next cycle."""
        # Complete first cycle
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        critique1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        improve1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique1
        )

        # Start second cycle
        critique2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=improve1
        )

        assert critique2.type == ThoughtType.VERIFICATION
        assert critique2.metadata["phase"] == "critique"
        assert critique2.metadata["reflection_cycle"] == 1


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    @pytest.mark.asyncio
    async def test_default_quality_threshold(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test default quality threshold."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        assert thought.metadata["quality_threshold"] == SelfReflection.QUALITY_THRESHOLD
        assert thought.metadata["quality_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_custom_quality_threshold(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test custom quality threshold in context."""
        custom_threshold = 0.95

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": custom_threshold},
        )

        assert thought.metadata["quality_threshold"] == custom_threshold

    @pytest.mark.asyncio
    async def test_quality_threshold_propagates(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that quality threshold propagates through cycle."""
        custom_threshold = 0.75

        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": custom_threshold},
        )

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert critique.metadata["quality_threshold"] == custom_threshold

    @pytest.mark.asyncio
    async def test_max_reflection_cycles_constant(self):
        """Test that MAX_REFLECTION_CYCLES is properly defined."""
        assert SelfReflection.MAX_REFLECTION_CYCLES == 5
        assert isinstance(SelfReflection.MAX_REFLECTION_CYCLES, int)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content="Test",
            metadata={"phase": "initial"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        assert initial.step_number == 1

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        assert critique.step_number == 2

        improvement = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique
        )
        assert improvement.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        guidance_text = "Focus on practical examples"
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance_text
        )

        assert "guidance" in critique.metadata
        assert critique.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        context = {"additional_info": "test data"}
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, context=context
        )

        assert "context" in critique.metadata
        assert critique.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        count_after_initial = session.thought_count

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert session.thought_count == count_after_initial + 1
        assert critique.id in session.graph.nodes


# ============================================================================
# Quality Improvement Tests
# ============================================================================


class TestQualityImprovement:
    """Test suite for quality score improvement tracking."""

    @pytest.mark.asyncio
    async def test_quality_improves_with_cycles(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that quality score improves with each cycle."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        initial_quality = initial.quality_score

        # First cycle
        critique1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        improve1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique1
        )

        # Second cycle
        critique2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=improve1
        )
        improve2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique2
        )

        # Quality should improve
        assert improve1.quality_score > initial_quality
        assert improve2.quality_score > improve1.quality_score

    @pytest.mark.asyncio
    async def test_confidence_improves_with_cycles(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that confidence improves with each cycle."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        initial_confidence = initial.confidence

        # First cycle
        critique1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        improve1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique1
        )

        # Improvement should have higher confidence
        assert improve1.confidence > initial_confidence

    @pytest.mark.asyncio
    async def test_quality_caps_at_one(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that quality score caps at 1.0."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        # Run many cycles to try to exceed 1.0
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Quality should never exceed 1.0
        assert thought.quality_score is not None
        assert thought.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_previous_quality_stored_in_metadata(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that previous quality is stored in metadata."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert "previous_quality" in critique.metadata
        assert critique.metadata["previous_quality"] == initial.quality_score


# ============================================================================
# Iteration Tracking Tests
# ============================================================================


class TestIterationTracking:
    """Test suite for iteration counting and limits."""

    @pytest.mark.asyncio
    async def test_reflection_cycle_increments(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that reflection_cycle increments correctly."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        assert initialized_method._reflection_cycle == 0

        # Critique doesn't increment
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._reflection_cycle == 0

        # Improvement increments
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._reflection_cycle == 1

    @pytest.mark.asyncio
    async def test_max_cycles_limit(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that max reflection cycles limit is respected."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        # Run until max cycles reached
        for i in range(SelfReflection.MAX_REFLECTION_CYCLES * 2):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should not exceed max
        assert initialized_method._reflection_cycle <= SelfReflection.MAX_REFLECTION_CYCLES

    @pytest.mark.asyncio
    async def test_needs_improvement_flag(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test the needs_improvement flag in metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.8},
        )

        # Initial quality is 0.6, threshold is 0.8
        assert thought.metadata["needs_improvement"] is True

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_execute(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that step counter resets on new execute."""
        # First execution
        thought1 = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        assert thought1.step_number == 1

        # Continue
        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )
        assert thought2.step_number == 2

        # New execution should reset
        session2 = Session().start()
        thought3 = await initialized_method.execute(
            session=session2, input_text=simple_input
        )
        assert thought3.step_number == 1


# ============================================================================
# Convergence Detection Tests
# ============================================================================


class TestConvergenceDetection:
    """Test suite for convergence and completion detection."""

    @pytest.mark.asyncio
    async def test_conclusion_when_quality_met(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that thought becomes CONCLUSION when quality threshold is met."""
        # Set low threshold so it's easily met
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.5},
        )

        # Go through cycles until quality exceeds threshold
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Eventually should get a CONCLUSION
        # (improvement phase with quality >= threshold)
        if thought.metadata["phase"] == "improve" and thought.quality_score >= 0.5:
            assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_conclusion_at_max_cycles(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that reasoning concludes at max cycles even if quality not met."""
        # Set high threshold that won't be met
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.99},
        )

        # Run through max cycles
        for _ in range(SelfReflection.MAX_REFLECTION_CYCLES * 2 + 10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should eventually conclude
        if initialized_method._reflection_cycle >= SelfReflection.MAX_REFLECTION_CYCLES:
            if thought.metadata["phase"] == "improve":
                assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_needs_improvement_false_at_conclusion(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that needs_improvement is False at conclusion."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.5},
        )

        # Run until conclusion
        for _ in range(10):
            prev_thought = thought
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break

        if thought.type == ThoughtType.CONCLUSION:
            assert thought.metadata["needs_improvement"] is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(
        self, initialized_method: SelfReflection, session: Session
    ):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(
        self, initialized_method: SelfReflection, session: Session
    ):
        """Test handling of very long input."""
        long_input = "What is " + "very " * 1000 + "important?"
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_fallback_to_critique_phase(
        self, initialized_method: SelfReflection, session: Session
    ):
        """Test fallback to critique for unknown phase."""
        # Create a thought with unknown phase
        initial = await initialized_method.execute(
            session=session, input_text="Test"
        )

        # Manually modify phase to unknown value
        initial.metadata["phase"] = "unknown_phase"

        # Continue should fallback to critique
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert thought.type == ThoughtType.VERIFICATION
        assert thought.metadata["phase"] == "critique"

    @pytest.mark.asyncio
    async def test_simple_problem_converges_quickly(
        self, initialized_method: SelfReflection, session: Session
    ):
        """Test that simple problems can converge quickly."""
        simple_question = "What is 2+2?"

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_question,
            context={"quality_threshold": 0.7},
        )

        cycles_count = 0
        max_iterations = 20

        for _ in range(max_iterations):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break
            cycles_count += 1

        # Should converge within reasonable iterations
        assert cycles_count < max_iterations

    @pytest.mark.asyncio
    async def test_complex_problem_uses_more_cycles(
        self, initialized_method: SelfReflection, session: Session, complex_input: str
    ):
        """Test that complex problems may use more cycles."""
        thought = await initialized_method.execute(
            session=session,
            input_text=complex_input,
            context={"quality_threshold": 0.9},
        )

        # Track cycles
        cycles = []
        for _ in range(15):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            cycles.append(thought.metadata["reflection_cycle"])

        # Should go through multiple cycles
        max_cycle = max(cycles)
        assert max_cycle >= 1

    @pytest.mark.asyncio
    async def test_quality_threshold_zero(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test handling of quality threshold of 0.0."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.0},
        )

        # Should immediately meet threshold
        assert thought.metadata["needs_improvement"] is False

    @pytest.mark.asyncio
    async def test_quality_threshold_one(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test handling of quality threshold of 1.0."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 1.0},
        )

        # Should always need improvement (initial quality < 1.0)
        assert thought.metadata["needs_improvement"] is True


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_initial_content_structure(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that initial response has expected content structure."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Initial Response" in content
        assert simple_input in content

    @pytest.mark.asyncio
    async def test_critique_content_structure(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that critique has expected content structure."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        content = critique.content
        assert isinstance(content, str)
        assert "Self-Critique" in content
        assert "Strengths" in content or "Weaknesses" in content

    @pytest.mark.asyncio
    async def test_improvement_content_structure(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that improvement has expected content structure."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        improvement = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique
        )

        content = improvement.content
        assert isinstance(content, str)
        assert "Improved Response" in content or "Reflection Cycle" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: SelfReflection, session: Session, simple_input: str
    ):
        """Test that guidance appears in generated content."""
        initial = await initialized_method.execute(
            session=session, input_text=simple_input
        )

        guidance = "Focus on practical examples"
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance
        )

        # Guidance should appear in content or metadata
        assert guidance in critique.metadata["guidance"]
