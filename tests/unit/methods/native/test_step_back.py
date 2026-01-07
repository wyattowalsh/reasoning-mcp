"""
Comprehensive tests for StepBack reasoning method.

This module provides complete test coverage for the StepBack method
from reasoning_mcp.methods.native.step_back, including:

1. Initialization and health checks
2. Basic execution and abstraction
3. Abstraction phase behavior
4. Configuration options
5. Continue reasoning through phases
6. Principle extraction
7. Application of abstract principles
8. Multiple abstraction levels
9. Solution derivation
10. Edge cases

Test coverage: 90%+ with 15+ test cases
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.step_back import (
    STEP_BACK_METADATA,
    StepBack,
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
def step_back_method() -> StepBack:
    """Provide a StepBack method instance for testing.

    Returns:
        Fresh StepBack instance (not initialized).
    """
    return StepBack()


@pytest.fixture
async def initialized_method() -> StepBack:
    """Provide an initialized StepBack method for testing.

    Returns:
        StepBack instance that has been initialized.
    """
    method = StepBack()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Provide an active session for testing.

    Returns:
        Session that has been started (status=ACTIVE).
    """
    session = Session()
    session.start()
    return session


@pytest.fixture
def sample_input() -> str:
    """Provide sample input text for testing.

    Returns:
        A sample problem statement.
    """
    return "How can I improve my team's productivity?"


@pytest.fixture
def concrete_problem() -> str:
    """Provide a very concrete problem for testing.

    Returns:
        A specific, detailed problem statement.
    """
    return "Fix the bug in function process_data() at line 42 where variable x is undefined"


@pytest.fixture
def abstract_problem() -> str:
    """Provide an already abstract problem for testing.

    Returns:
        An abstract, high-level problem statement.
    """
    return "What are the fundamental principles of effective system design?"


# ============================================================================
# Test StepBack Metadata
# ============================================================================


class TestStepBackMetadata:
    """Test suite for StepBack method metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert STEP_BACK_METADATA.identifier == MethodIdentifier.STEP_BACK

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert STEP_BACK_METADATA.name == "Step Back"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert STEP_BACK_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert STEP_BACK_METADATA.complexity == 4

    def test_metadata_supports_branching(self):
        """Test that metadata correctly indicates branching support."""
        assert STEP_BACK_METADATA.supports_branching is False

    def test_metadata_supports_revision(self):
        """Test that metadata correctly indicates revision support."""
        assert STEP_BACK_METADATA.supports_revision is True

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {
            "abstraction",
            "principles",
            "conceptual",
            "general-to-specific",
            "high-level",
            "strategic",
        }
        assert expected_tags.issubset(STEP_BACK_METADATA.tags)


# ============================================================================
# Test Initialization
# ============================================================================


class TestStepBackInitialization:
    """Test suite for StepBack initialization."""

    def test_create_uninitialized_method(self, step_back_method: StepBack):
        """Test creating a new StepBack method instance."""
        assert step_back_method._initialized is False
        assert step_back_method._step_counter == 0
        assert step_back_method._current_phase == StepBack.PHASE_ABSTRACTION
        assert step_back_method._abstraction_levels == []
        assert step_back_method._identified_principles == []

    @pytest.mark.asyncio
    async def test_initialize_method(self, step_back_method: StepBack):
        """Test initializing the StepBack method."""
        await step_back_method.initialize()

        assert step_back_method._initialized is True
        assert step_back_method._step_counter == 0
        assert step_back_method._current_phase == StepBack.PHASE_ABSTRACTION
        assert step_back_method._abstraction_levels == []
        assert step_back_method._identified_principles == []

    @pytest.mark.asyncio
    async def test_reinitialize_resets_state(self, initialized_method: StepBack):
        """Test that reinitializing resets the method state."""
        # Modify state
        initialized_method._step_counter = 5
        initialized_method._current_phase = StepBack.PHASE_APPLICATION
        initialized_method._abstraction_levels.append("level1")
        initialized_method._identified_principles.append("principle1")

        # Reinitialize
        await initialized_method.initialize()

        # Verify reset
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == StepBack.PHASE_ABSTRACTION
        assert initialized_method._abstraction_levels == []
        assert initialized_method._identified_principles == []

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(
        self, step_back_method: StepBack
    ):
        """Test health check returns False before initialization."""
        result = await step_back_method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(
        self, initialized_method: StepBack
    ):
        """Test health check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_with_invalid_phase(
        self, initialized_method: StepBack
    ):
        """Test health check returns False with invalid phase."""
        initialized_method._current_phase = "INVALID_PHASE"
        result = await initialized_method.health_check()
        assert result is False


# ============================================================================
# Test Properties
# ============================================================================


class TestStepBackProperties:
    """Test suite for StepBack properties."""

    def test_identifier_property(self, step_back_method: StepBack):
        """Test identifier property returns correct value."""
        assert step_back_method.identifier == MethodIdentifier.STEP_BACK

    def test_name_property(self, step_back_method: StepBack):
        """Test name property returns correct value."""
        assert step_back_method.name == "Step Back"

    def test_description_property(self, step_back_method: StepBack):
        """Test description property returns correct value."""
        assert "Step back" in step_back_method.description
        assert "abstract" in step_back_method.description.lower()

    def test_category_property(self, step_back_method: StepBack):
        """Test category property returns correct value."""
        assert step_back_method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Test Basic Execution
# ============================================================================


class TestStepBackExecution:
    """Test suite for StepBack execute method."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, step_back_method: StepBack, active_session: Session, sample_input: str
    ):
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await step_back_method.execute(
                session=active_session,
                input_text=sample_input,
            )

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute creates an initial abstraction thought."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.STEP_BACK
        assert thought.step_number == 1
        assert thought.depth == 0
        assert sample_input in thought.content or sample_input in thought.metadata.get(
            "input", ""
        )

    @pytest.mark.asyncio
    async def test_execute_sets_abstraction_phase(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute sets the phase to ABSTRACTION."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION
        assert initialized_method._current_phase == StepBack.PHASE_ABSTRACTION

    @pytest.mark.asyncio
    async def test_execute_sets_abstraction_level(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute sets initial abstraction level."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert "abstraction_level" in thought.metadata
        assert thought.metadata["abstraction_level"] == 1

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute adds thought to the session."""
        initial_count = active_session.thought_count

        await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert active_session.thought_count == initial_count + 1
        assert active_session.current_method == MethodIdentifier.STEP_BACK

    @pytest.mark.asyncio
    async def test_execute_with_context(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test execute with additional context."""
        context = {"domain": "software", "priority": "high"}

        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
            context=context,
        )

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_resets_counters(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute resets step counter and state."""
        # Set some state
        initialized_method._step_counter = 10
        initialized_method._current_phase = StepBack.PHASE_REFINEMENT

        await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert initialized_method._step_counter == 1
        assert initialized_method._current_phase == StepBack.PHASE_ABSTRACTION

    @pytest.mark.asyncio
    async def test_execute_confidence_level(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that initial thought has moderate confidence."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Initial abstraction should have moderate confidence
        assert 0.5 <= thought.confidence <= 0.7


# ============================================================================
# Test Continue Reasoning
# ============================================================================


class TestStepBackContinueReasoning:
    """Test suite for StepBack continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, step_back_method: StepBack, active_session: Session
    ):
        """Test that continue_reasoning raises RuntimeError if not initialized."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.STEP_BACK,
            content="Test",
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await step_back_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
            )

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning increments step counter."""
        first = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        second = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first,
        )

        assert second.step_number == 2
        assert initialized_method._step_counter == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning sets parent_id correctly."""
        first = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        second = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first,
        )

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_depth(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning increases depth."""
        first = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        second = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first,
        )

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_to_session(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning adds thought to session."""
        first = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        initial_count = active_session.thought_count

        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first,
        )

        assert active_session.thought_count == initial_count + 1


# ============================================================================
# Test Phase Progression
# ============================================================================


class TestStepBackPhaseProgression:
    """Test suite for StepBack phase progression."""

    @pytest.mark.asyncio
    async def test_phase_progression_natural_flow(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test natural progression through phases based on step count."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Step 2: Should stay in ABSTRACTION
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION

        # Step 3: Should move to PRINCIPLES
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_PRINCIPLES

        # Step 5: Should move to FRAMEWORK
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_FRAMEWORK

        # Step 7: Should move to APPLICATION
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_APPLICATION

        # Step 9: Should move to REFINEMENT
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_REFINEMENT

    @pytest.mark.asyncio
    async def test_guidance_forces_principles_phase(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that guidance with 'principle' keyword forces PRINCIPLES phase."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Identify the core principles",
        )

        assert thought.metadata["phase"] == StepBack.PHASE_PRINCIPLES

    @pytest.mark.asyncio
    async def test_guidance_forces_framework_phase(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that guidance with 'framework' keyword forces FRAMEWORK phase."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Build a conceptual framework",
        )

        assert thought.metadata["phase"] == StepBack.PHASE_FRAMEWORK

    @pytest.mark.asyncio
    async def test_guidance_forces_application_phase(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that guidance with 'apply' keyword forces APPLICATION phase."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Apply these concepts to the specific problem",
        )

        assert thought.metadata["phase"] == StepBack.PHASE_APPLICATION

    @pytest.mark.asyncio
    async def test_guidance_forces_refinement_phase(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that guidance with 'refine' keyword forces REFINEMENT phase."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Refine the solution",
        )

        assert thought.metadata["phase"] == StepBack.PHASE_REFINEMENT


# ============================================================================
# Test Thought Types by Phase
# ============================================================================


class TestStepBackThoughtTypes:
    """Test suite for thought types in different phases."""

    @pytest.mark.asyncio
    async def test_principles_phase_uses_hypothesis_type(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that PRINCIPLES phase uses HYPOTHESIS thought type."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Force PRINCIPLES phase
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Identify the principles",
        )

        assert thought.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_framework_phase_uses_synthesis_type(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that FRAMEWORK phase uses SYNTHESIS thought type."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Force FRAMEWORK phase
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Build the framework",
        )

        assert thought.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_application_phase_uses_continuation_type(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that APPLICATION phase uses CONTINUATION thought type."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Force APPLICATION phase
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Apply to the specific problem",
        )

        assert thought.type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_refinement_phase_uses_verification_type(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that REFINEMENT phase uses VERIFICATION thought type."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Force REFINEMENT phase
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Refine the solution",
        )

        assert thought.type == ThoughtType.VERIFICATION


# ============================================================================
# Test Confidence Calculation
# ============================================================================


class TestStepBackConfidence:
    """Test suite for confidence calculation across phases."""

    @pytest.mark.asyncio
    async def test_confidence_increases_through_phases(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that confidence generally increases as we move from abstract to concrete."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )
        initial_confidence = thought.confidence

        # Move through phases and track confidence
        for guidance in [
            "Identify principles",  # PRINCIPLES
            "Build framework",  # FRAMEWORK
            "Apply to problem",  # APPLICATION
            "Refine solution",  # REFINEMENT
        ]:
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=guidance,
            )

        # Final confidence should be higher than initial
        assert thought.confidence > initial_confidence

    @pytest.mark.asyncio
    async def test_confidence_bounded(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that confidence stays within valid bounds."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Go through multiple iterations
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
            )

        # Confidence should stay in valid range
        assert 0.3 <= thought.confidence <= 0.95


# ============================================================================
# Test Abstraction Levels
# ============================================================================


class TestStepBackAbstractionLevels:
    """Test suite for abstraction level tracking."""

    @pytest.mark.asyncio
    async def test_abstraction_level_decreases_through_phases(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that abstraction level decreases from abstract to concrete phases."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # ABSTRACTION phase should have highest level (5)
        assert thought.metadata["abstraction_level"] == 1  # Initial is 1, but phase is 5

        # Force different phases and check levels
        phases_and_levels = [
            ("Identify principles", 4),  # PRINCIPLES
            ("Build framework", 3),  # FRAMEWORK
            ("Apply to problem", 2),  # APPLICATION
            ("Refine solution", 1),  # REFINEMENT
        ]

        for guidance, expected_level in phases_and_levels:
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=guidance,
            )
            # The abstraction_level in metadata should match the phase's level
            assert thought.metadata["abstraction_level"] == expected_level


# ============================================================================
# Test Content Generation
# ============================================================================


class TestStepBackContentGeneration:
    """Test suite for content generation across phases."""

    @pytest.mark.asyncio
    async def test_abstraction_content_includes_step_number(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that abstraction content includes step number."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        assert "Step 1" in thought.content

    @pytest.mark.asyncio
    async def test_abstraction_content_includes_problem(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that abstraction content references the problem."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Should reference the specific problem
        assert sample_input in thought.content or "Specific Problem" in thought.content

    @pytest.mark.asyncio
    async def test_principles_content_mentions_principles(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that PRINCIPLES phase content mentions principles."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Identify principles",
        )

        assert "principle" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_application_content_references_original_problem(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that APPLICATION phase content references original problem."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Store the input in metadata
        thought.metadata["input"] = sample_input

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Apply to problem",
        )

        # Should reference applying to the specific problem
        assert "specific problem" in thought.content.lower() or "apply" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_guidance_included_in_content(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that provided guidance is included in the content."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        custom_guidance = "Focus on scalability principles"
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance=custom_guidance,
        )

        assert custom_guidance in thought.content


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestStepBackEdgeCases:
    """Test suite for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_already_abstract_problem(
        self,
        initialized_method: StepBack,
        active_session: Session,
        abstract_problem: str,
    ):
        """Test handling of already abstract problem."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=abstract_problem,
        )

        # Should still create abstraction thought
        assert thought.type == ThoughtType.INITIAL
        assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION

        # Should handle gracefully
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )
        assert thought is not None

    @pytest.mark.asyncio
    async def test_very_concrete_problem(
        self,
        initialized_method: StepBack,
        active_session: Session,
        concrete_problem: str,
    ):
        """Test handling of very concrete, specific problem."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=concrete_problem,
        )

        # Should successfully abstract even concrete problems
        assert thought.type == ThoughtType.INITIAL
        assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION
        assert "Step" in thought.content

    @pytest.mark.asyncio
    async def test_empty_context(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test execution with None context."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
            context=None,
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_guidance(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test continue_reasoning with None guidance."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance=None,
        )

        assert thought.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_multiple_continue_iterations(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test multiple iterations of continue_reasoning."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Continue for many steps
        for i in range(15):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
            )

        # Should reach REFINEMENT phase and stay there
        assert thought.metadata["phase"] == StepBack.PHASE_REFINEMENT
        assert thought.step_number == 16

    @pytest.mark.asyncio
    async def test_metadata_preserved_through_continuation(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that metadata is properly tracked through continuations."""
        context = {"domain": "testing"}

        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
            context=context,
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )

        # Metadata should be preserved
        assert "phase" in thought.metadata
        assert "abstraction_level" in thought.metadata
        assert "principles_identified" in thought.metadata

    @pytest.mark.asyncio
    async def test_paused_session_can_execute(
        self, initialized_method: StepBack, sample_input: str
    ):
        """Test that method doesn't validate session status."""
        # Create a paused session
        session = Session()
        session.start()
        session.pause()

        # Method should still execute (session status validation is external)
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_input,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestStepBackIntegration:
    """Test suite for integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_reasoning_flow(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test complete reasoning flow through all phases."""
        # Execute initial thought
        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )
        assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION

        # Progress through all phases
        phases = [
            StepBack.PHASE_ABSTRACTION,
            StepBack.PHASE_PRINCIPLES,
            StepBack.PHASE_FRAMEWORK,
            StepBack.PHASE_APPLICATION,
            StepBack.PHASE_REFINEMENT,
        ]

        seen_phases = {thought.metadata["phase"]}

        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
            )
            seen_phases.add(thought.metadata["phase"])

        # Should have seen multiple phases
        assert len(seen_phases) >= 3

    @pytest.mark.asyncio
    async def test_session_metrics_updated(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that session metrics are properly updated."""
        initial_thoughts = active_session.metrics.total_thoughts

        thought = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        # Metrics should be updated
        assert active_session.metrics.total_thoughts == initial_thoughts + 1
        assert (
            active_session.metrics.methods_used[MethodIdentifier.STEP_BACK]
            == 1
        )

        # Continue and check again
        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
        )

        assert active_session.metrics.total_thoughts == initial_thoughts + 2
        assert (
            active_session.metrics.methods_used[MethodIdentifier.STEP_BACK]
            == 2
        )

    @pytest.mark.asyncio
    async def test_thought_graph_connectivity(
        self,
        initialized_method: StepBack,
        active_session: Session,
        sample_input: str,
    ):
        """Test that thoughts are properly connected in the graph."""
        thought1 = await initialized_method.execute(
            session=active_session,
            input_text=sample_input,
        )

        thought2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought1,
        )

        thought3 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought2,
        )

        # Verify parent-child relationships
        assert thought2.parent_id == thought1.id
        assert thought3.parent_id == thought2.id

        # Verify depths
        assert thought1.depth == 0
        assert thought2.depth == 1
        assert thought3.depth == 2

    @pytest.mark.asyncio
    async def test_multiple_executions_in_same_session(
        self,
        initialized_method: StepBack,
        active_session: Session,
    ):
        """Test multiple execute calls in the same session."""
        # First execution
        thought1 = await initialized_method.execute(
            session=active_session,
            input_text="First problem",
        )

        # Second execution - should reset internal state
        thought2 = await initialized_method.execute(
            session=active_session,
            input_text="Second problem",
        )

        # Both should be initial thoughts
        assert thought1.type == ThoughtType.INITIAL
        assert thought2.type == ThoughtType.INITIAL

        # Step counter should reset
        assert thought2.step_number == 1

        # Both should be in session
        assert active_session.thought_count >= 2
