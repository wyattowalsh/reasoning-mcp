"""Comprehensive unit tests for MetacognitiveMethod.

This module tests the MetacognitiveMethod class from reasoning_mcp.methods.native.metacognitive,
covering initialization, execution, all four metacognitive phases, configuration, strategy
adaptation, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.metacognitive import (
    METACOGNITIVE_METADATA,
    MetacognitiveMethod,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestMetacognitiveInitialization:
    """Tests for MetacognitiveMethod initialization and health checks."""

    @pytest.mark.asyncio
    async def test_initialization_sets_state(self):
        """Test that initialize() sets proper initial state."""
        method = MetacognitiveMethod()
        assert method._initialized is False

        await method.initialize()

        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "planning"
        assert method._strategy_adjustments == 0
        assert method._metacognitive_cycle == 0

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self):
        """Test health_check() returns False before initialization."""
        method = MetacognitiveMethod()
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self):
        """Test health_check() returns True after initialization."""
        method = MetacognitiveMethod()
        await method.initialize()
        result = await method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_reinitialization_resets_state(self):
        """Test that re-initializing resets state."""
        method = MetacognitiveMethod()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "evaluating"
        method._strategy_adjustments = 3
        method._metacognitive_cycle = 2

        # Re-initialize
        await method.initialize()

        assert method._step_counter == 0
        assert method._current_phase == "planning"
        assert method._strategy_adjustments == 0
        assert method._metacognitive_cycle == 0

    def test_identifier_property(self):
        """Test identifier property returns correct value."""
        method = MetacognitiveMethod()
        assert method.identifier == MethodIdentifier.METACOGNITIVE

    def test_name_property(self):
        """Test name property returns correct value."""
        method = MetacognitiveMethod()
        assert method.name == "Metacognitive Reasoning"

    def test_description_property(self):
        """Test description property returns correct value."""
        method = MetacognitiveMethod()
        assert "thinking about thinking" in method.description.lower()
        assert "monitors and regulates" in method.description.lower()

    def test_category_property(self):
        """Test category property returns correct value."""
        method = MetacognitiveMethod()
        assert method.category == MethodCategory.ADVANCED


class TestMetacognitiveBasicExecution:
    """Tests for basic execute() functionality."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(self):
        """Test execute() raises error if not initialized."""
        method = MetacognitiveMethod()
        session = Session().start()

        with pytest.raises(RuntimeError, match="must be initialized before execution"):
            await method.execute(session=session, input_text="Test problem")

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self):
        """Test execute() creates initial planning thought."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="How to solve complex optimization problem?"
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.METACOGNITIVE
        assert thought.step_number == 1
        assert thought.depth == 0
        assert "PLANNING" in thought.content
        assert "How to solve complex optimization problem?" in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_planning_phase(self):
        """Test execute() sets phase to planning."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert thought.metadata["phase"] == "planning"
        assert method._current_phase == "planning"

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self):
        """Test execute() adds thought to session."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        assert session.thought_count == 0

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert session.thought_count == 1
        assert session.current_method == MethodIdentifier.METACOGNITIVE
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test execute() handles context parameter."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()
        context = {"domain": "mathematics", "difficulty": "high"}

        thought = await method.execute(
            session=session,
            input_text="Solve equation",
            context=context
        )

        assert thought.metadata["context"] == context
        assert "mathematics" in thought.content or "high" in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metrics(self):
        """Test execute() sets proper metacognitive metrics."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert thought.metadata["reasoning_type"] == "metacognitive"
        assert thought.metadata["metacognitive_cycle"] == 0
        assert thought.metadata["strategy_awareness"] == 0.8
        assert thought.metadata["progress_monitoring"] == 0.0
        assert thought.metadata["self_evaluation_quality"] == 0.0
        assert thought.metadata["adaptive_adjustments"] == 0
        assert thought.confidence == 0.7


class TestMetacognitiveFourPhases:
    """Tests for the four metacognitive phases: PLANNING, MONITORING, EVALUATING, REGULATING."""

    @pytest.mark.asyncio
    async def test_phase_progression_planning_to_monitoring(self):
        """Test phase progression from PLANNING to MONITORING."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        # Execute: PLANNING
        plan_thought = await method.execute(session=session, input_text="Test problem")
        assert plan_thought.metadata["phase"] == "planning"

        # Continue: MONITORING
        monitor_thought = await method.continue_reasoning(
            session=session,
            previous_thought=plan_thought
        )

        assert monitor_thought.metadata["phase"] == "monitoring"
        assert monitor_thought.type == ThoughtType.VERIFICATION
        assert monitor_thought.parent_id == plan_thought.id
        assert "MONITORING" in monitor_thought.content
        assert "Tracking Reasoning Progress" in monitor_thought.content

    @pytest.mark.asyncio
    async def test_phase_progression_monitoring_to_evaluating(self):
        """Test phase progression from MONITORING to EVALUATING."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)

        # Continue: EVALUATING
        evaluate_thought = await method.continue_reasoning(
            session=session,
            previous_thought=monitor
        )

        assert evaluate_thought.metadata["phase"] == "evaluating"
        assert evaluate_thought.type == ThoughtType.VERIFICATION
        assert "EVALUATING" in evaluate_thought.content
        assert "Assessing Reasoning Quality" in evaluate_thought.content

    @pytest.mark.asyncio
    async def test_phase_progression_evaluating_to_regulating(self):
        """Test phase progression from EVALUATING to REGULATING."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)

        # Continue: REGULATING
        regulate_thought = await method.continue_reasoning(
            session=session,
            previous_thought=evaluate
        )

        assert regulate_thought.metadata["phase"] == "regulating"
        assert regulate_thought.type == ThoughtType.REVISION
        assert "REGULATING" in regulate_thought.content
        assert "Adapting Strategy" in regulate_thought.content
        assert method._strategy_adjustments == 1

    @pytest.mark.asyncio
    async def test_phase_progression_regulating_to_monitoring_new_cycle(self):
        """Test phase progression from REGULATING back to MONITORING (new cycle)."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        regulate = await method.continue_reasoning(session=session, previous_thought=evaluate)

        # Continue: Back to MONITORING (new cycle)
        monitor2_thought = await method.continue_reasoning(
            session=session,
            previous_thought=regulate
        )

        assert monitor2_thought.metadata["phase"] == "monitoring"
        assert monitor2_thought.type == ThoughtType.VERIFICATION
        assert monitor2_thought.metadata["metacognitive_cycle"] == 1
        assert method._metacognitive_cycle == 1

    @pytest.mark.asyncio
    async def test_planning_phase_content_structure(self):
        """Test PLANNING phase has correct content structure."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Complex problem")

        content = thought.content
        assert "METACOGNITIVE PLANNING:" in content
        assert "PROBLEM ANALYSIS" in content
        assert "STRATEGY SELECTION" in content
        assert "RESOURCE IDENTIFICATION" in content
        assert "GOAL SETTING" in content
        assert "ANTICIPATING OBSTACLES" in content

    @pytest.mark.asyncio
    async def test_monitoring_phase_content_structure(self):
        """Test MONITORING phase has correct content structure."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)

        content = monitor.content
        assert "PROGRESS MONITORING:" in content
        assert "COMPREHENSION CHECK" in content
        assert "PROGRESS ASSESSMENT" in content
        assert "ATTENTION MANAGEMENT" in content
        assert "PACING CHECK" in content
        assert "ERROR DETECTION" in content

    @pytest.mark.asyncio
    async def test_evaluating_phase_content_structure(self):
        """Test EVALUATING phase has correct content structure."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)

        content = evaluate.content
        assert "QUALITY EVALUATION:" in content
        assert "EFFECTIVENESS ASSESSMENT" in content
        assert "EFFICIENCY ANALYSIS" in content
        assert "STRENGTHS IDENTIFICATION" in content
        assert "WEAKNESSES IDENTIFICATION" in content
        assert "LEARNING OPPORTUNITIES" in content

    @pytest.mark.asyncio
    async def test_regulating_phase_content_structure(self):
        """Test REGULATING phase has correct content structure."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        regulate = await method.continue_reasoning(session=session, previous_thought=evaluate)

        content = regulate.content
        assert "ADAPTIVE REGULATION:" in content
        assert "STRATEGY ADJUSTMENT" in content
        assert "ALTERNATIVE APPROACHES" in content
        assert "RESOURCE REALLOCATION" in content
        assert "GOAL REVISION" in content
        assert "IMPLEMENTATION PLAN" in content
        assert "Adjustment #1" in content


class TestMetacognitiveContinueReasoning:
    """Tests for continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(self):
        """Test continue_reasoning() raises error if not initialized."""
        method = MetacognitiveMethod()
        session = Session().start()
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.METACOGNITIVE,
            content="Test",
            metadata={"phase": "planning"}
        )

        with pytest.raises(RuntimeError, match="must be initialized before continuation"):
            await method.continue_reasoning(
                session=session,
                previous_thought=thought
            )

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_counter(self):
        """Test continue_reasoning() increments step counter."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        assert method._step_counter == 1

        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        assert method._step_counter == 2
        assert monitor.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance_parameter(self):
        """Test continue_reasoning() respects guidance parameter."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="Check if approach is working"
        )

        assert monitor.metadata["guidance"] == "Check if approach is working"
        assert "Check if approach is working" in monitor.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context_parameter(self):
        """Test continue_reasoning() handles context parameter."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()
        context = {"additional_info": "important details"}

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            context=context
        )

        assert monitor.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_continue_reasoning_updates_session(self):
        """Test continue_reasoning() updates session properly."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        initial_count = session.thought_count

        monitor = await method.continue_reasoning(session=session, previous_thought=plan)

        assert session.thought_count == initial_count + 1
        assert monitor.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_continue_reasoning_continues_metacognitive_cycle(self):
        """Test complete metacognitive cycle continuation."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        # Complete one full cycle
        plan = await method.execute(session=session, input_text="Test")
        monitor1 = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate1 = await method.continue_reasoning(session=session, previous_thought=monitor1)
        regulate1 = await method.continue_reasoning(session=session, previous_thought=evaluate1)
        monitor2 = await method.continue_reasoning(session=session, previous_thought=regulate1)

        # Verify cycle incremented
        assert plan.metadata["metacognitive_cycle"] == 0
        assert monitor1.metadata["metacognitive_cycle"] == 0
        assert monitor2.metadata["metacognitive_cycle"] == 1


class TestMetacognitiveStrategySelection:
    """Tests for strategy selection and guidance-based phase control."""

    @pytest.mark.asyncio
    async def test_guidance_override_to_planning_phase(self):
        """Test guidance can override to PLANNING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)

        # Force back to planning with guidance
        new_plan = await method.continue_reasoning(
            session=session,
            previous_thought=monitor,
            guidance="plan new strategy"
        )

        assert new_plan.metadata["phase"] == "planning"
        assert new_plan.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_guidance_override_to_monitoring_phase(self):
        """Test guidance can override to MONITORING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor1 = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor1)

        # Force back to monitoring
        monitor2 = await method.continue_reasoning(
            session=session,
            previous_thought=evaluate,
            guidance="monitor current progress"
        )

        assert monitor2.metadata["phase"] == "monitoring"
        assert monitor2.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_guidance_override_to_evaluating_phase(self):
        """Test guidance can override to EVALUATING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")

        # Force to evaluating immediately
        evaluate = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="evaluate effectiveness"
        )

        assert evaluate.metadata["phase"] == "evaluating"
        assert evaluate.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_guidance_override_to_regulating_phase(self):
        """Test guidance can override to REGULATING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")

        # Force to regulating immediately
        regulate = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="adjust strategy now"
        )

        assert regulate.metadata["phase"] == "regulating"
        assert regulate.type == ThoughtType.REVISION

    @pytest.mark.asyncio
    async def test_guidance_alternative_creates_branch(self):
        """Test guidance with 'alternative' creates BRANCH thought type."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")

        # Force to alternative/branch
        branch = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="explore alternative approach"
        )

        assert branch.metadata["phase"] == "regulating"
        assert branch.type == ThoughtType.BRANCH


class TestMetacognitiveProgressMonitoring:
    """Tests for tracking reasoning progress."""

    @pytest.mark.asyncio
    async def test_progress_monitoring_metrics_in_monitoring_phase(self):
        """Test progress_monitoring metric is high in MONITORING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)

        assert monitor.metadata["progress_monitoring"] == 0.9
        assert monitor.metadata["strategy_awareness"] == 0.8

    @pytest.mark.asyncio
    async def test_step_counter_tracks_progress(self):
        """Test step counter correctly tracks reasoning progress."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thoughts = []
        thoughts.append(await method.execute(session=session, input_text="Test"))

        for _ in range(5):
            thoughts.append(await method.continue_reasoning(
                session=session,
                previous_thought=thoughts[-1]
            ))

        # Verify step numbers
        for i, thought in enumerate(thoughts):
            assert thought.step_number == i + 1

    @pytest.mark.asyncio
    async def test_depth_increases_with_continuation(self):
        """Test depth increases as reasoning continues."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        assert plan.depth == 0

        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        assert monitor.depth == plan.depth + 1

        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        assert evaluate.depth == monitor.depth + 1


class TestMetacognitiveStrategyAdaptation:
    """Tests for adapting strategy when not working."""

    @pytest.mark.asyncio
    async def test_strategy_adjustments_counter_increments(self):
        """Test strategy_adjustments counter increments in REGULATING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        assert method._strategy_adjustments == 0

        # Navigate to regulating phase
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        regulate = await method.continue_reasoning(session=session, previous_thought=evaluate)

        assert method._strategy_adjustments == 1
        assert regulate.metadata["adaptive_adjustments"] == 1

    @pytest.mark.asyncio
    async def test_multiple_strategy_adjustments(self):
        """Test multiple strategy adjustments increment properly."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test")

        # Force multiple regulations
        for i in range(3):
            thought = await method.continue_reasoning(
                session=session,
                previous_thought=thought,
                guidance="adjust strategy"
            )
            assert thought.metadata["phase"] == "regulating"
            assert method._strategy_adjustments == i + 1
            assert f"Adjustment #{i + 1}" in thought.content

    @pytest.mark.asyncio
    async def test_regulating_phase_modifies_confidence(self):
        """Test REGULATING phase sets appropriate confidence level."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        regulate = await method.continue_reasoning(session=session, previous_thought=evaluate)

        # Regulating introduces uncertainty
        assert regulate.confidence == 0.75
        assert regulate.metadata["strategy_awareness"] == 0.95


class TestMetacognitiveSelfEvaluation:
    """Tests for evaluating own reasoning quality."""

    @pytest.mark.asyncio
    async def test_self_evaluation_metric_in_evaluating_phase(self):
        """Test self_evaluation_quality is high in EVALUATING phase."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)

        assert evaluate.metadata["self_evaluation_quality"] == 0.95
        assert evaluate.metadata["progress_monitoring"] == 0.7

    @pytest.mark.asyncio
    async def test_confidence_increases_through_phases(self):
        """Test confidence increases as metacognitive process progresses."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        assert plan.confidence == 0.7

        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        assert monitor.confidence >= plan.confidence

        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)
        assert evaluate.confidence >= monitor.confidence

    @pytest.mark.asyncio
    async def test_evaluating_phase_references_previous_step(self):
        """Test EVALUATING phase references previous step number."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)

        assert f"step {monitor.step_number}" in evaluate.content
        assert evaluate.metadata["previous_step"] == monitor.step_number


class TestMetacognitiveEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_simple_problem_handling(self):
        """Test metacognitive method handles simple problems."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="What is 2 + 2?"
        )

        assert thought is not None
        assert thought.metadata["phase"] == "planning"
        assert "What is 2 + 2?" in thought.content

    @pytest.mark.asyncio
    async def test_stuck_problem_multiple_adjustments(self):
        """Test handling of stuck problem requiring multiple strategy adjustments."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Unsolvable riddle with no clear answer"
        )

        # Simulate getting stuck and adjusting multiple times
        for _ in range(5):
            thought = await method.continue_reasoning(
                session=session,
                previous_thought=thought,
                guidance="adjust strategy"
            )

        assert method._strategy_adjustments == 5
        assert "Adjustment #5" in thought.content

    @pytest.mark.asyncio
    async def test_strategy_switching_via_guidance(self):
        """Test switching strategies via guidance keywords."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")

        # Switch to different phases rapidly
        monitor = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="track progress closely"
        )
        assert monitor.metadata["phase"] == "monitoring"

        evaluate = await method.continue_reasoning(
            session=session,
            previous_thought=monitor,
            guidance="assess quality now"
        )
        assert evaluate.metadata["phase"] == "evaluating"

        regulate = await method.continue_reasoning(
            session=session,
            previous_thought=evaluate,
            guidance="regulate and adjust"
        )
        assert regulate.metadata["phase"] == "regulating"

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test handling of empty input text."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="")

        assert thought is not None
        assert thought.metadata["phase"] == "planning"

    @pytest.mark.asyncio
    async def test_none_context_handling(self):
        """Test that None context is handled properly."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test",
            context=None
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context_handling(self):
        """Test that empty context dict is handled properly."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test",
            context={}
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_metacognitive_cycle_counter(self):
        """Test metacognitive cycle counter increments properly."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        # First cycle
        plan1 = await method.execute(session=session, input_text="Test")
        assert plan1.metadata["metacognitive_cycle"] == 0

        monitor1 = await method.continue_reasoning(session=session, previous_thought=plan1)
        evaluate1 = await method.continue_reasoning(session=session, previous_thought=monitor1)
        regulate1 = await method.continue_reasoning(session=session, previous_thought=evaluate1)

        # Second cycle starts
        monitor2 = await method.continue_reasoning(session=session, previous_thought=regulate1)
        assert monitor2.metadata["metacognitive_cycle"] == 1
        assert method._metacognitive_cycle == 1

    @pytest.mark.asyncio
    async def test_complex_guidance_with_multiple_keywords(self):
        """Test guidance with multiple keywords prioritizes actionable keywords."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")

        # Guidance with multiple keywords - actionable keywords (evaluate) take priority
        thought = await method.continue_reasoning(
            session=session,
            previous_thought=plan,
            guidance="plan to monitor and evaluate strategy"
        )

        # Actionable keywords (evaluate) prioritized over general planning keywords
        assert thought.metadata["phase"] == "evaluating"

    @pytest.mark.asyncio
    async def test_parent_child_relationship_maintained(self):
        """Test parent-child relationships are maintained throughout reasoning."""
        method = MetacognitiveMethod()
        await method.initialize()
        session = Session().start()

        plan = await method.execute(session=session, input_text="Test")
        monitor = await method.continue_reasoning(session=session, previous_thought=plan)
        evaluate = await method.continue_reasoning(session=session, previous_thought=monitor)

        assert monitor.parent_id == plan.id
        assert evaluate.parent_id == monitor.id

        # Verify in graph
        assert plan.id in session.graph.nodes
        assert monitor.id in session.graph.nodes
        assert evaluate.id in session.graph.nodes


class TestMetacognitiveMetadata:
    """Tests for METACOGNITIVE_METADATA configuration."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert METACOGNITIVE_METADATA.identifier == MethodIdentifier.METACOGNITIVE

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert METACOGNITIVE_METADATA.name == "Metacognitive Reasoning"

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert METACOGNITIVE_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self):
        """Test metadata has high complexity."""
        assert METACOGNITIVE_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert METACOGNITIVE_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert METACOGNITIVE_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context not required."""
        assert METACOGNITIVE_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test metadata has minimum 4 thoughts (4 phases)."""
        assert METACOGNITIVE_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test metadata has no max thoughts limit."""
        assert METACOGNITIVE_METADATA.max_thoughts == 0

    def test_metadata_tags(self):
        """Test metadata has appropriate tags."""
        expected_tags = {
            "metacognitive",
            "self-monitoring",
            "learning",
            "adaptive",
            "strategy",
            "self-awareness",
            "regulation",
        }
        assert METACOGNITIVE_METADATA.tags == frozenset(expected_tags)

    def test_metadata_best_for(self):
        """Test metadata lists appropriate use cases."""
        assert "learning optimization" in METACOGNITIVE_METADATA.best_for
        assert "strategy selection" in METACOGNITIVE_METADATA.best_for
        assert "adaptive reasoning" in METACOGNITIVE_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata lists inappropriate use cases."""
        assert "simple factual questions" in METACOGNITIVE_METADATA.not_recommended_for
        assert "time-critical tasks" in METACOGNITIVE_METADATA.not_recommended_for
