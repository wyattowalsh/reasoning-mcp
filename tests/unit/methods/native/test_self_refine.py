"""Comprehensive tests for SelfRefine reasoning method.

This module provides complete test coverage for the SelfRefine method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Refinement cycles (generate -> feedback -> refine)
- Configuration options (max_iterations)
- Continue reasoning flow
- Feedback generation and tracking
- Refinement tracking
- Iteration tracking and limits
- Edge cases

The tests aim for 90%+ coverage of the SelfRefine implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.self_refine import (
    SELF_REFINE_METADATA,
    SelfRefine,
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
def method() -> SelfRefine:
    """Provide a SelfRefine method instance for testing.

    Returns:
        SelfRefine instance (uninitialized).
    """
    return SelfRefine()


@pytest.fixture
async def initialized_method() -> SelfRefine:
    """Provide an initialized SelfRefine method instance.

    Returns:
        Initialized SelfRefine instance.
    """
    method = SelfRefine()
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
    return "Write a clear explanation of recursion"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex question requiring deeper refinement.
    """
    return "Write a comprehensive guide to machine learning for beginners"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfRefineMetadata:
    """Test suite for SelfRefine metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SELF_REFINE_METADATA.identifier == MethodIdentifier.SELF_REFINE

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SELF_REFINE_METADATA.name == "Self-Refine"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert SELF_REFINE_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert SELF_REFINE_METADATA.complexity == 4
        assert 1 <= SELF_REFINE_METADATA.complexity <= 10

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert SELF_REFINE_METADATA.supports_revision is True

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert SELF_REFINE_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "iterative",
            "self-improvement",
            "feedback",
            "refinement",
            "incremental",
        }
        assert expected_tags.issubset(SELF_REFINE_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert SELF_REFINE_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert SELF_REFINE_METADATA.max_thoughts == 12

    def test_metadata_description(self):
        """Test that metadata has meaningful description."""
        assert "feedback" in SELF_REFINE_METADATA.description.lower()
        assert "refine" in SELF_REFINE_METADATA.description.lower()


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSelfRefineInitialization:
    """Test suite for SelfRefine initialization."""

    def test_create_method(self, method: SelfRefine):
        """Test creating a SelfRefine instance."""
        assert isinstance(method, SelfRefine)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: SelfRefine):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.SELF_REFINE
        assert method.name == "Self-Refine"
        assert method.category == MethodCategory.SPECIALIZED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: SelfRefine):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._iteration_count == 0
        assert method._current_phase == "generate"

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = SelfRefine()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._iteration_count = 2
        method._current_phase = "refine"

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._iteration_count == 0
        assert method._current_phase == "generate"

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: SelfRefine):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: SelfRefine):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSelfRefineExecution:
    """Test suite for basic SelfRefine execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SELF_REFINE
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metadata(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["phase"] == "generate"
        assert thought.metadata["iteration_count"] == 0
        assert "max_iterations" in thought.metadata
        assert thought.metadata["reasoning_type"] == "self_refine"

    @pytest.mark.asyncio
    async def test_execute_sets_initial_confidence(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute sets initial confidence."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert thought.confidence == 0.6

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.SELF_REFINE
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"max_iterations": 3, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["max_iterations"] == 3
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_max_iterations(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that max_iterations is clamped to [1, 10]."""
        # Test upper bound
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": 20},
        )
        assert thought1.metadata["max_iterations"] == 10

        # Re-initialize for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test lower bound
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"max_iterations": 0},
        )
        assert thought2.metadata["max_iterations"] == 1

    @pytest.mark.asyncio
    async def test_execute_initializes_feedback_items(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that execute initializes empty feedback_items list."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "feedback_items" in thought.metadata
        assert thought.metadata["feedback_items"] == []


# ============================================================================
# Refinement Cycle Tests
# ============================================================================


class TestRefinementCycle:
    """Test suite for the refinement cycle flow."""

    @pytest.mark.asyncio
    async def test_feedback_phase_after_generate(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback follows generate thought."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert feedback.type == ThoughtType.VERIFICATION
        assert feedback.metadata["phase"] == "feedback"
        assert feedback.parent_id == initial.id
        assert feedback.step_number == 2

    @pytest.mark.asyncio
    async def test_refine_phase_after_feedback(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement follows feedback."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        assert refinement.type in (ThoughtType.REVISION, ThoughtType.CONCLUSION)
        assert refinement.metadata["phase"] == "refine"
        assert refinement.parent_id == feedback.id
        assert refinement.step_number == 3
        assert refinement.metadata["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_full_refinement_cycle(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test a complete refinement cycle: generate -> feedback -> refine."""
        # Initial generation
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        assert initial.metadata["iteration_count"] == 0

        # Feedback
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        assert feedback.metadata["iteration_count"] == 0

        # Refinement
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )
        assert refinement.metadata["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_refinement_cycles(
        self, initialized_method: SelfRefine, session: Session
    ):
        """Test multiple refinement cycles."""
        input_text = "Write about artificial intelligence"

        # Cycle 0: Generate
        thought = await initialized_method.execute(session=session, input_text=input_text)
        assert thought.metadata["iteration_count"] == 0

        # Cycle 0: Feedback
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["iteration_count"] == 0

        # Cycle 1: Refine
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["iteration_count"] == 1

        # Cycle 1: Feedback
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["iteration_count"] == 1

        # Cycle 2: Refine
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["iteration_count"] == 2

    @pytest.mark.asyncio
    async def test_feedback_after_refinement(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback can follow refinement for next cycle."""
        # Complete first cycle
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refine1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback1
        )

        # Start second cycle
        feedback2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=refine1
        )

        assert feedback2.type == ThoughtType.VERIFICATION
        assert feedback2.metadata["phase"] == "feedback"
        assert feedback2.metadata["iteration_count"] == 1


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    @pytest.mark.asyncio
    async def test_default_max_iterations(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test default max iterations."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["max_iterations"] == SelfRefine.MAX_ITERATIONS
        assert thought.metadata["max_iterations"] == 4

    @pytest.mark.asyncio
    async def test_custom_max_iterations(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test custom max iterations in context."""
        custom_iterations = 6

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": custom_iterations},
        )

        assert thought.metadata["max_iterations"] == custom_iterations

    @pytest.mark.asyncio
    async def test_max_iterations_propagates(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that max_iterations propagates through cycle."""
        custom_iterations = 5

        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": custom_iterations},
        )

        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert feedback.metadata["max_iterations"] == custom_iterations

    @pytest.mark.asyncio
    async def test_max_iterations_constant(self):
        """Test that MAX_ITERATIONS is properly defined."""
        assert SelfRefine.MAX_ITERATIONS == 4
        assert isinstance(SelfRefine.MAX_ITERATIONS, int)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_REFINE,
            content="Test",
            metadata={"phase": "generate"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        assert initial.step_number == 1

        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        assert feedback.step_number == 2

        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )
        assert refinement.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Focus on clarity"
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance_text
        )

        assert "guidance" in feedback.metadata
        assert feedback.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        context = {"additional_info": "test data"}
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, context=context
        )

        assert "context" in feedback.metadata
        assert feedback.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        count_after_initial = session.thought_count

        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert session.thought_count == count_after_initial + 1
        assert feedback.id in session.graph.nodes


# ============================================================================
# Feedback Generation Tests
# ============================================================================


class TestFeedbackGeneration:
    """Test suite for feedback generation."""

    @pytest.mark.asyncio
    async def test_feedback_contains_items(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback contains feedback items."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert "feedback_items" in feedback.metadata
        assert isinstance(feedback.metadata["feedback_items"], list)
        assert len(feedback.metadata["feedback_items"]) > 0

    @pytest.mark.asyncio
    async def test_feedback_count_matches_items(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback_count matches length of feedback_items."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        feedback_items = feedback.metadata["feedback_items"]
        feedback_count = feedback.metadata["feedback_count"]

        assert feedback_count == len(feedback_items)

    @pytest.mark.asyncio
    async def test_feedback_content_structure(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback has expected content structure."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        content = feedback.content
        assert isinstance(content, str)
        assert "Feedback" in content
        assert "Iteration" in content

    @pytest.mark.asyncio
    async def test_feedback_references_previous_step(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback references the previous step."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert f"Step {initial.step_number}" in feedback.content


# ============================================================================
# Refinement Tests
# ============================================================================


class TestRefinement:
    """Test suite for refinement generation."""

    @pytest.mark.asyncio
    async def test_refinement_addresses_feedback(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement addresses feedback items."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        assert "addressed_items" in refinement.metadata
        assert refinement.metadata["addressed_items"] > 0

    @pytest.mark.asyncio
    async def test_refinement_increments_iteration(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement increments iteration count."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        assert refinement.metadata["iteration_count"] == 1
        assert refinement.metadata["refinement_iteration"] == 1

    @pytest.mark.asyncio
    async def test_refinement_content_structure(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement has expected content structure."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        content = refinement.content
        assert isinstance(content, str)
        assert "Refined" in content or "Iteration" in content

    @pytest.mark.asyncio
    async def test_refinement_stores_feedback_items(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement stores feedback items in metadata."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        assert "feedback_items" in refinement.metadata
        feedback_items = feedback.metadata["feedback_items"]
        assert refinement.metadata["feedback_items"] == feedback_items


# ============================================================================
# Iteration Tracking Tests
# ============================================================================


class TestIterationTracking:
    """Test suite for iteration counting and limits."""

    @pytest.mark.asyncio
    async def test_iteration_count_increments(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that iteration_count increments correctly."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert initialized_method._iteration_count == 0

        # Feedback doesn't increment
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._iteration_count == 0

        # Refinement increments
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._iteration_count == 1

    @pytest.mark.asyncio
    async def test_max_iterations_limit(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that max iterations limit is respected."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Run until max iterations reached
        for _i in range(SelfRefine.MAX_ITERATIONS * 2 + 2):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should not exceed max
        assert initialized_method._iteration_count <= SelfRefine.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_should_continue_flag(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test the should_continue flag in metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": 2},
        )

        # Generate feedback and refinement
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        # At iteration 1, should continue
        assert thought.metadata["should_continue"] is True

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_execute(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that step counter resets on new execute."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought1.step_number == 1

        # Continue
        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )
        assert thought2.step_number == 2

        # New execution should reset
        session2 = Session().start()
        thought3 = await initialized_method.execute(session=session2, input_text=simple_input)
        assert thought3.step_number == 1

    @pytest.mark.asyncio
    async def test_iteration_count_resets_on_execute(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that iteration count resets on new execute."""
        # First execution and cycle
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._iteration_count == 1

        # New execution should reset
        session2 = Session().start()
        await initialized_method.execute(session=session2, input_text=simple_input)
        assert initialized_method._iteration_count == 0


# ============================================================================
# Confidence Tests
# ============================================================================


class TestConfidenceImprovement:
    """Test suite for confidence improvement tracking."""

    @pytest.mark.asyncio
    async def test_confidence_improves_with_iterations(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that confidence improves with each refinement iteration."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        initial_confidence = initial.confidence

        # First cycle
        feedback1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refine1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback1
        )

        # Second cycle
        feedback2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=refine1
        )
        refine2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback2
        )

        # Confidence should improve
        assert refine1.confidence > initial_confidence
        assert refine2.confidence > refine1.confidence

    @pytest.mark.asyncio
    async def test_confidence_caps_at_limit(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that confidence caps at 0.95."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Run many cycles to try to exceed 0.95
        for _ in range(20):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Confidence should never exceed 0.95
        assert thought.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_feedback_has_moderate_confidence(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback phase has moderate confidence."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert feedback.confidence == 0.7


# ============================================================================
# Conclusion Tests
# ============================================================================


class TestConclusion:
    """Test suite for conclusion detection."""

    @pytest.mark.asyncio
    async def test_conclusion_at_max_iterations(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that reasoning concludes at max iterations."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": 2},
        )

        # Run through max iterations
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should eventually conclude
        if initialized_method._iteration_count >= 2:
            if thought.metadata["phase"] == "refine":
                assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_should_continue_false_at_conclusion(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that should_continue is False at conclusion."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": 1},
        )

        # Run until conclusion
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break

        if thought.type == ThoughtType.CONCLUSION:
            assert thought.metadata["should_continue"] is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: SelfRefine, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: SelfRefine, session: Session):
        """Test handling of very long input."""
        long_input = "Write about " + "very " * 1000 + "important topic"
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_fallback_to_feedback_phase(
        self, initialized_method: SelfRefine, session: Session
    ):
        """Test fallback to feedback for unknown phase."""
        # Create a thought with unknown phase
        initial = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        initial.metadata["phase"] = "unknown_phase"

        # Continue should fallback to feedback
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert thought.type == ThoughtType.VERIFICATION
        assert thought.metadata["phase"] == "feedback"

    @pytest.mark.asyncio
    async def test_simple_task_completes(self, initialized_method: SelfRefine, session: Session):
        """Test that simple tasks complete within max iterations."""
        simple_question = "Explain addition"

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_question,
            context={"max_iterations": 2},
        )

        cycles_count = 0
        max_steps = 20

        for _ in range(max_steps):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break
            cycles_count += 1

        # Should complete within reasonable steps
        assert cycles_count < max_steps

    @pytest.mark.asyncio
    async def test_complex_task_uses_more_cycles(
        self, initialized_method: SelfRefine, session: Session, complex_input: str
    ):
        """Test that complex tasks may use more cycles."""
        thought = await initialized_method.execute(
            session=session,
            input_text=complex_input,
            context={"max_iterations": 4},
        )

        # Track iterations
        iterations = []
        for _ in range(15):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            iterations.append(thought.metadata["iteration_count"])

        # Should go through multiple iterations
        max_iteration = max(iterations)
        assert max_iteration >= 1

    @pytest.mark.asyncio
    async def test_max_iterations_one(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test handling of max_iterations = 1."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_iterations": 1},
        )

        assert thought.metadata["max_iterations"] == 1

        # After one refinement, should conclude
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )  # feedback
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )  # refinement (iteration 1)

        assert thought.metadata["iteration_count"] == 1
        assert thought.type == ThoughtType.CONCLUSION


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_initial_content_structure(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that initial generation has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Initial Generation" in content
        assert simple_input in content

    @pytest.mark.asyncio
    async def test_feedback_content_references_task(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that feedback references the task."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        content = feedback.content
        assert isinstance(content, str)
        assert "Feedback" in content

    @pytest.mark.asyncio
    async def test_refinement_content_shows_improvements(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that refinement shows improvements."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        refinement = await initialized_method.continue_reasoning(
            session=session, previous_thought=feedback
        )

        content = refinement.content
        assert isinstance(content, str)
        assert "Refined" in content or "Iteration" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: SelfRefine, session: Session, simple_input: str
    ):
        """Test that guidance appears in generated content or metadata."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        guidance = "Be more specific"
        feedback = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance
        )

        # Guidance should appear in metadata
        assert feedback.metadata["guidance"] == guidance
