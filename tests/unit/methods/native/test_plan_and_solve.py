"""Comprehensive tests for PlanAndSolve reasoning method.

This module provides complete test coverage for the PlanAndSolve method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Phase progression (understand → plan → execute → synthesize)
- Plan generation and step extraction
- Step execution tracking
- Synthesis and completion
- Configuration options
- Continue reasoning flow
- Edge cases

The tests aim for 90%+ coverage of the PlanAndSolve implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.plan_and_solve import (
    PLAN_AND_SOLVE_METADATA,
    PlanAndSolve,
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
def method() -> PlanAndSolve:
    """Provide a PlanAndSolve method instance for testing.

    Returns:
        PlanAndSolve instance (uninitialized).
    """
    return PlanAndSolve()


@pytest.fixture
async def initialized_method() -> PlanAndSolve:
    """Provide an initialized PlanAndSolve method instance.

    Returns:
        Initialized PlanAndSolve instance.
    """
    method = PlanAndSolve()
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
        Simple problem for testing.
    """
    return "Solve the equation: 2x + 5 = 13"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex problem requiring detailed planning.
    """
    return "Analyze the impact of climate change on global agricultural systems"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMetadata:
    """Test suite for PlanAndSolve metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert PLAN_AND_SOLVE_METADATA.identifier == MethodIdentifier.PLAN_AND_SOLVE

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert PLAN_AND_SOLVE_METADATA.name == "Plan-and-Solve"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert PLAN_AND_SOLVE_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert PLAN_AND_SOLVE_METADATA.complexity == 4
        assert 1 <= PLAN_AND_SOLVE_METADATA.complexity <= 10

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert PLAN_AND_SOLVE_METADATA.supports_branching is False

    def test_metadata_no_revision(self):
        """Test that metadata indicates no revision support."""
        assert PLAN_AND_SOLVE_METADATA.supports_revision is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "planning",
            "structured",
            "step-by-step",
            "decomposition",
            "sequential",
        }
        assert expected_tags.issubset(PLAN_AND_SOLVE_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert PLAN_AND_SOLVE_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert PLAN_AND_SOLVE_METADATA.max_thoughts == 15


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test suite for PlanAndSolve initialization."""

    def test_create_method(self, method: PlanAndSolve):
        """Test creating a PlanAndSolve instance."""
        assert isinstance(method, PlanAndSolve)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: PlanAndSolve):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.PLAN_AND_SOLVE
        assert method.name == "Plan-and-Solve"
        assert method.category == MethodCategory.SPECIALIZED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: PlanAndSolve):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "understand"
        assert method._plan_steps == []
        assert method._current_step_index == -1

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = PlanAndSolve()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "execute"
        method._plan_steps = ["step1", "step2"]
        method._current_step_index = 1

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._current_phase == "understand"
        assert method._plan_steps == []
        assert method._current_step_index == -1

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: PlanAndSolve):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: PlanAndSolve):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestExecution:
    """Test suite for basic PlanAndSolve execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.PLAN_AND_SOLVE
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_understand_phase(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute sets understand phase."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert thought.metadata["phase"] == "understand"
        assert thought.metadata["reasoning_type"] == "plan_and_solve"
        assert thought.metadata["input"] == simple_input

    @pytest.mark.asyncio
    async def test_execute_initializes_metadata(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["phase"] == "understand"
        assert thought.metadata["plan_steps"] == []
        assert thought.metadata["current_step_index"] == -1
        assert thought.metadata["reasoning_type"] == "plan_and_solve"

    @pytest.mark.asyncio
    async def test_execute_sets_initial_scores(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute sets initial confidence and quality."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert thought.confidence == 0.7
        assert thought.quality_score == 0.6

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.PLAN_AND_SOLVE
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"domain": "mathematics", "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["context"]["domain"] == "mathematics"
        assert thought.metadata["context"]["custom_key"] == "custom_value"


# ============================================================================
# Plan Generation Tests
# ============================================================================


class TestPlanGeneration:
    """Test suite for plan generation."""

    @pytest.mark.asyncio
    async def test_plan_phase_after_understand(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan follows understand phase."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        assert plan.type == ThoughtType.CONTINUATION
        assert plan.metadata["phase"] == "plan"
        assert plan.parent_id == understand.id
        assert plan.step_number == 2

    @pytest.mark.asyncio
    async def test_plan_creates_plan_steps(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan phase creates plan steps."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        assert "plan_steps" in plan.metadata
        assert len(plan.metadata["plan_steps"]) > 0
        assert isinstance(plan.metadata["plan_steps"], list)

    @pytest.mark.asyncio
    async def test_plan_has_default_steps(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan has expected default steps."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        steps = plan.metadata["plan_steps"]
        assert len(steps) >= 3  # At least 3 steps
        assert all(isinstance(step, str) for step in steps)

    @pytest.mark.asyncio
    async def test_plan_with_guidance(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test plan generation with guidance."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Create a detailed mathematical plan"
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand, guidance=guidance_text
        )

        assert "guidance" in plan.metadata
        assert plan.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_plan_depth_increases(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan depth is one more than understand."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        assert plan.depth == understand.depth + 1


# ============================================================================
# Step Execution Tests
# ============================================================================


class TestStepExecution:
    """Test suite for plan step execution."""

    @pytest.mark.asyncio
    async def test_execute_phase_after_plan(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute follows plan phase."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        execute = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )

        assert execute.type == ThoughtType.CONTINUATION
        assert execute.metadata["phase"] == "execute"
        assert execute.parent_id == plan.id
        assert execute.step_number == 3

    @pytest.mark.asyncio
    async def test_execute_starts_at_step_zero(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execution starts at step 0."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        execute = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )

        assert execute.metadata["current_step_index"] == 0

    @pytest.mark.asyncio
    async def test_execute_carries_plan_steps(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execution carries forward plan steps."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        execute = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )

        assert execute.metadata["plan_steps"] == plan.metadata["plan_steps"]

    @pytest.mark.asyncio
    async def test_execute_multiple_steps(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test executing multiple plan steps."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        # Execute step 1
        execute1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )
        assert execute1.metadata["current_step_index"] == 0

        # Execute step 2
        execute2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=execute1
        )
        assert execute2.metadata["current_step_index"] == 1

        # Execute step 3
        execute3 = await initialized_method.continue_reasoning(
            session=session, previous_thought=execute2
        )
        assert execute3.metadata["current_step_index"] == 2

    @pytest.mark.asyncio
    async def test_execute_step_increments(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that step index increments correctly."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )
        execute1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )

        execute2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=execute1
        )

        assert (
            execute2.metadata["current_step_index"] == execute1.metadata["current_step_index"] + 1
        )


# ============================================================================
# Synthesis Tests
# ============================================================================


class TestSynthesis:
    """Test suite for synthesis phase."""

    @pytest.mark.asyncio
    async def test_synthesis_after_all_steps(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that synthesis occurs after all steps executed."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        # Execute all steps
        thought = plan
        num_steps = len(plan.metadata["plan_steps"])
        for _ in range(num_steps):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Next should be synthesis
        synthesis = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert synthesis.metadata["phase"] == "synthesize"
        assert synthesis.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_synthesis_is_conclusion(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that synthesis phase creates CONCLUSION thought."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        # Execute all steps
        thought = plan
        num_steps = len(plan.metadata["plan_steps"])
        for _ in range(num_steps + 1):  # +1 to reach synthesis
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        if thought.metadata["phase"] == "synthesize":
            assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_synthesis_has_high_confidence(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that synthesis has higher confidence than earlier phases."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        # Execute all steps to synthesis
        thought = plan
        num_steps = len(plan.metadata["plan_steps"])
        for _ in range(num_steps + 1):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        if thought.metadata["phase"] == "synthesize":
            assert thought.confidence >= 0.85
            assert thought.quality_score >= 0.8

    @pytest.mark.asyncio
    async def test_synthesis_with_guidance(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test synthesis with guidance."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        # Execute to synthesis
        thought = plan
        num_steps = len(plan.metadata["plan_steps"])
        for i in range(num_steps + 1):
            guidance = "Provide detailed synthesis" if i == num_steps else None
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought, guidance=guidance
            )

        if thought.metadata["phase"] == "synthesize":
            assert "guidance" in thought.metadata


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.PLAN_AND_SOLVE,
            content="Test",
            metadata={"phase": "understand"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        assert understand.step_number == 1

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )
        assert plan.step_number == 2

        execute = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )
        assert execute.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Focus on mathematical approach"
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand, guidance=guidance_text
        )

        assert "guidance" in plan.metadata
        assert plan.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        context = {"additional_info": "test data"}
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand, context=context
        )

        assert "context" in plan.metadata
        assert plan.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        count_after_understand = session.thought_count

        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        assert session.thought_count == count_after_understand + 1
        assert plan.id in session.graph.nodes


# ============================================================================
# Phase Progression Tests
# ============================================================================


class TestPhaseProgression:
    """Test suite for phase progression logic."""

    @pytest.mark.asyncio
    async def test_full_phase_progression(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test complete phase progression: understand → plan → execute → synthesize."""
        # Understand
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["phase"] == "understand"

        # Plan
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["phase"] == "plan"

        # Execute (all steps)
        num_steps = len(thought.metadata["plan_steps"])
        for i in range(num_steps):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            assert thought.metadata["phase"] == "execute"
            assert thought.metadata["current_step_index"] == i

        # Synthesize
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["phase"] == "synthesize"

    @pytest.mark.asyncio
    async def test_phase_transitions_are_linear(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that phases progress in expected linear order."""
        phases = []

        thought = await initialized_method.execute(session=session, input_text=simple_input)
        phases.append(thought.metadata["phase"])

        # Continue through all phases
        for _ in range(10):  # More than enough to reach synthesize
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            phases.append(thought.metadata["phase"])
            if thought.metadata["phase"] == "synthesize":
                break

        # Check expected progression
        assert phases[0] == "understand"
        assert phases[1] == "plan"
        # Then multiple "execute" phases
        assert "execute" in phases[2:-1]
        assert phases[-1] == "synthesize"

    @pytest.mark.asyncio
    async def test_depth_increases_each_phase(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that depth increases with each phase."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        prev_depth = thought.depth

        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.metadata["phase"] != "synthesize":  # synthesize might not increase
                assert thought.depth >= prev_depth
            prev_depth = thought.depth


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: PlanAndSolve, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: PlanAndSolve, session: Session):
        """Test handling of very long input."""
        long_input = "Solve " + "x + 1 = 2, " * 500
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_continue_after_synthesize(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that continuing after synthesize handles gracefully."""
        # Get to synthesize phase
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Progress through all phases
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.metadata["phase"] == "synthesize":
                break

        # Try to continue after synthesize
        final = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert final.metadata["phase"] == "synthesize"
        assert final.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_fallback_to_plan_unknown_phase(
        self, initialized_method: PlanAndSolve, session: Session
    ):
        """Test fallback to plan for unknown phase."""
        # Create a thought with unknown phase
        understand = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        understand.metadata["phase"] = "unknown_phase"

        # Continue should fallback to plan
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        assert thought.type == ThoughtType.CONTINUATION
        assert thought.metadata["phase"] == "plan"

    @pytest.mark.asyncio
    async def test_plan_steps_persist_through_execution(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan steps persist through all execution steps."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        original_steps = plan.metadata["plan_steps"]

        # Execute through all steps
        thought = plan
        for _ in range(len(original_steps)):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            assert thought.metadata["plan_steps"] == original_steps

    @pytest.mark.asyncio
    async def test_execute_resets_on_new_execution(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute resets state for new execution."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=simple_input)
        await initialized_method.continue_reasoning(session=session, previous_thought=thought1)

        # New execution should reset
        session2 = Session().start()
        thought2 = await initialized_method.execute(
            session=session2, input_text="Different problem"
        )

        assert thought2.step_number == 1
        assert thought2.metadata["phase"] == "understand"
        assert thought2.metadata["plan_steps"] == []
        assert thought2.metadata["current_step_index"] == -1


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_understand_content_structure(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that understand phase has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Understanding the Problem" in content
        assert simple_input in content

    @pytest.mark.asyncio
    async def test_plan_content_structure(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that plan phase has expected content structure."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )

        content = plan.content
        assert isinstance(content, str)
        assert "Step-by-Step Plan" in content or "Plan" in content
        assert "STEP" in content  # Should have step markers

    @pytest.mark.asyncio
    async def test_execute_content_structure(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that execute phase has expected content structure."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand
        )
        execute = await initialized_method.continue_reasoning(
            session=session, previous_thought=plan
        )

        content = execute.content
        assert isinstance(content, str)
        assert "Executing Plan Step" in content or "Execute" in content

    @pytest.mark.asyncio
    async def test_synthesize_content_structure(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that synthesize phase has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Progress to synthesize
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.metadata["phase"] == "synthesize":
                break

        if thought.metadata["phase"] == "synthesize":
            content = thought.content
            assert isinstance(content, str)
            assert "Synthesizing" in content or "Final Answer" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that guidance appears in generated content or metadata."""
        understand = await initialized_method.execute(session=session, input_text=simple_input)

        guidance = "Focus on step-by-step breakdown"
        plan = await initialized_method.continue_reasoning(
            session=session, previous_thought=understand, guidance=guidance
        )

        # Guidance should appear in metadata
        assert guidance in plan.metadata["guidance"]


# ============================================================================
# Confidence and Quality Tests
# ============================================================================


class TestConfidenceAndQuality:
    """Test suite for confidence and quality scores."""

    @pytest.mark.asyncio
    async def test_confidence_increases_through_phases(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that confidence generally increases through phases."""
        thoughts = []

        thought = await initialized_method.execute(session=session, input_text=simple_input)
        thoughts.append(thought)

        # Progress through phases
        for _ in range(6):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            thoughts.append(thought)
            if thought.metadata["phase"] == "synthesize":
                break

        # Check that later phases generally have higher confidence
        understand_conf = thoughts[0].confidence
        synthesize_conf = (
            thoughts[-1].confidence
            if thoughts[-1].metadata["phase"] == "synthesize"
            else thoughts[-1].confidence
        )

        assert synthesize_conf >= understand_conf

    @pytest.mark.asyncio
    async def test_quality_increases_through_phases(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that quality score increases through phases."""
        thoughts = []

        thought = await initialized_method.execute(session=session, input_text=simple_input)
        thoughts.append(thought)

        # Progress through phases
        for _ in range(6):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            thoughts.append(thought)
            if thought.metadata["phase"] == "synthesize":
                break

        # Check quality progression
        initial_quality = thoughts[0].quality_score
        final_quality = thoughts[-1].quality_score

        assert final_quality >= initial_quality

    @pytest.mark.asyncio
    async def test_synthesize_has_highest_scores(
        self, initialized_method: PlanAndSolve, session: Session, simple_input: str
    ):
        """Test that synthesize phase has highest confidence and quality."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Progress to synthesize
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.metadata["phase"] == "synthesize":
                break

        if thought.metadata["phase"] == "synthesize":
            assert thought.confidence >= 0.85
            assert thought.quality_score >= 0.8


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test suite for integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_problem_solving_flow(
        self, initialized_method: PlanAndSolve, session: Session
    ):
        """Test complete problem-solving flow from start to finish."""
        problem = "Calculate the area of a circle with radius 5"

        # Execute
        thought = await initialized_method.execute(session=session, input_text=problem)
        assert thought.metadata["phase"] == "understand"
        assert session.thought_count == 1

        # Plan
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["phase"] == "plan"
        assert len(thought.metadata["plan_steps"]) > 0
        assert session.thought_count == 2

        # Execute all steps
        num_steps = len(thought.metadata["plan_steps"])
        for i in range(num_steps):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            assert thought.metadata["phase"] == "execute"
            assert thought.metadata["current_step_index"] == i

        # Synthesize
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["phase"] == "synthesize"
        assert thought.type == ThoughtType.CONCLUSION

        # Verify session state
        assert session.thought_count >= 6  # At least understand + plan + 4 steps
        assert session.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_multiple_problems_in_sequence(self, initialized_method: PlanAndSolve):
        """Test solving multiple problems in sequence."""
        problems = [
            "Solve: x + 5 = 10",
            "Calculate: 15 * 3",
            "Find the square root of 144",
        ]

        for problem in problems:
            session = Session().start()
            thought = await initialized_method.execute(session=session, input_text=problem)

            # Complete the reasoning
            for _ in range(10):
                thought = await initialized_method.continue_reasoning(
                    session=session, previous_thought=thought
                )
                if thought.metadata["phase"] == "synthesize":
                    break

            assert thought.metadata["phase"] == "synthesize"

    @pytest.mark.asyncio
    async def test_complex_problem_with_many_steps(
        self, initialized_method: PlanAndSolve, session: Session, complex_input: str
    ):
        """Test handling complex problem with many execution steps."""
        thought = await initialized_method.execute(session=session, input_text=complex_input)

        # Count total thoughts generated
        thought_count = 1

        # Progress to synthesize
        for _ in range(20):  # Allow for many steps
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            thought_count += 1
            if thought.metadata["phase"] == "synthesize":
                break

        # Should have gone through understand, plan, multiple executes, and synthesize
        assert thought_count >= 6
        assert thought.metadata["phase"] == "synthesize"
