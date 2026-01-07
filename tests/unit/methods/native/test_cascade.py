"""Unit tests for CascadeThinking reasoning method.

This module provides comprehensive tests for the CascadeThinking method implementation,
covering initialization, execution, cascade levels, configuration, continuation,
refinement, level transitions, feedback loops, synthesis, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.cascade import (
    CASCADE_THINKING_METADATA,
    CascadeThinkingMethod,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


# Fixtures


@pytest.fixture
def cascade_method() -> CascadeThinkingMethod:
    """Create a CascadeThinkingMethod instance for testing.

    Returns:
        A fresh CascadeThinkingMethod instance
    """
    return CascadeThinkingMethod()


@pytest.fixture
def active_session() -> Session:
    """Create an active session for testing.

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
    return "Design a distributed microservices architecture for an e-commerce platform"


@pytest.fixture
def complex_problem() -> str:
    """Provide a complex problem for testing.

    Returns:
        A complex problem string
    """
    return (
        "Develop a comprehensive climate change mitigation strategy that addresses "
        "emissions reduction, renewable energy adoption, carbon sequestration, "
        "policy frameworks, and international cooperation"
    )


# Test Metadata


class TestMetadata:
    """Tests for CascadeThinking metadata."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert CASCADE_THINKING_METADATA.identifier == MethodIdentifier.CASCADE_THINKING

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert CASCADE_THINKING_METADATA.name == "Cascade Thinking"

    def test_metadata_description(self):
        """Test metadata has descriptive text."""
        assert len(CASCADE_THINKING_METADATA.description) > 0
        assert "hierarchical" in CASCADE_THINKING_METADATA.description.lower()
        assert "cascad" in CASCADE_THINKING_METADATA.description.lower()  # cascade, cascading

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert CASCADE_THINKING_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self):
        """Test metadata contains expected tags."""
        expected_tags = {
            "hierarchical",
            "cascade",
            "refinement",
            "strategic",
            "tactical",
            "operational",
        }
        assert expected_tags.issubset(CASCADE_THINKING_METADATA.tags)

    def test_metadata_complexity(self):
        """Test metadata has reasonable complexity rating."""
        assert 1 <= CASCADE_THINKING_METADATA.complexity <= 10
        assert CASCADE_THINKING_METADATA.complexity == 6

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert CASCADE_THINKING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert CASCADE_THINKING_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates no context requirement."""
        assert CASCADE_THINKING_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test metadata has minimum thoughts requirement."""
        assert CASCADE_THINKING_METADATA.min_thoughts >= 4  # All 4 levels

    def test_metadata_max_thoughts(self):
        """Test metadata has unlimited max thoughts (0)."""
        assert CASCADE_THINKING_METADATA.max_thoughts == 0

    def test_metadata_best_for(self):
        """Test metadata best_for contains relevant use cases."""
        best_for_str = " ".join(CASCADE_THINKING_METADATA.best_for)
        assert "planning" in best_for_str.lower()
        assert "strategy" in best_for_str.lower()


# Test Initialization


class TestInitialization:
    """Tests for CascadeThinking initialization and setup."""

    def test_create_method(self, cascade_method: CascadeThinkingMethod):
        """Test that CascadeThinkingMethod can be instantiated."""
        assert cascade_method is not None
        assert isinstance(cascade_method, CascadeThinkingMethod)

    def test_initial_state(self, cascade_method: CascadeThinkingMethod):
        """Test that a new method starts in the correct initial state."""
        assert cascade_method._initialized is False
        assert cascade_method._step_counter == 0
        assert cascade_method._current_level == "STRATEGIC"
        assert len(cascade_method._level_outputs) == 0
        assert len(cascade_method._feedback_items) == 0

    def test_cascade_levels_defined(self, cascade_method: CascadeThinkingMethod):
        """Test that cascade levels are properly defined."""
        expected_levels = ["STRATEGIC", "TACTICAL", "OPERATIONAL", "DETAILED"]
        assert cascade_method._cascade_levels == expected_levels

    @pytest.mark.asyncio
    async def test_initialize(self, cascade_method: CascadeThinkingMethod):
        """Test that initialize() sets up the method correctly."""
        await cascade_method.initialize()
        assert cascade_method._initialized is True
        assert cascade_method._step_counter == 0
        assert cascade_method._current_level == "STRATEGIC"
        assert len(cascade_method._level_outputs) == 0
        assert len(cascade_method._feedback_items) == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize() resets state even if called multiple times."""
        method = CascadeThinkingMethod()
        await method.initialize()

        # Simulate some usage
        method._step_counter = 5
        method._current_level = "DETAILED"
        method._level_outputs["STRATEGIC"] = "test"
        method._feedback_items.append({"test": "data"})

        # Re-initialize
        await method.initialize()
        assert method._step_counter == 0
        assert method._current_level == "STRATEGIC"
        assert len(method._level_outputs) == 0
        assert len(method._feedback_items) == 0
        assert method._initialized is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, cascade_method: CascadeThinkingMethod):
        """Test that health_check returns False before initialization."""
        result = await cascade_method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, cascade_method: CascadeThinkingMethod):
        """Test that health_check returns True after initialization."""
        await cascade_method.initialize()
        result = await cascade_method.health_check()
        assert result is True


# Test Properties


class TestProperties:
    """Tests for CascadeThinking property accessors."""

    def test_identifier_property(self, cascade_method: CascadeThinkingMethod):
        """Test that identifier returns the correct method identifier."""
        assert cascade_method.identifier == MethodIdentifier.CASCADE_THINKING

    def test_name_property(self, cascade_method: CascadeThinkingMethod):
        """Test that name returns the correct human-readable name."""
        assert cascade_method.name == "Cascade Thinking"

    def test_description_property(self, cascade_method: CascadeThinkingMethod):
        """Test that description returns the correct method description."""
        assert "hierarchical" in cascade_method.description.lower()
        assert "cascad" in cascade_method.description.lower()  # cascade, cascading

    def test_category_property(self, cascade_method: CascadeThinkingMethod):
        """Test that category returns the correct method category."""
        assert cascade_method.category == MethodCategory.HOLISTIC


# Test Basic Execution


class TestBasicExecution:
    """Tests for basic execute() functionality."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test that execute raises error without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await cascade_method.execute(active_session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_strategic_thought(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() creates strategic level thought."""
        await cascade_method.initialize()
        thought = await cascade_method.execute(active_session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CASCADE_THINKING

    @pytest.mark.asyncio
    async def test_execute_sets_strategic_metadata(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets strategic level metadata."""
        await cascade_method.initialize()
        thought = await cascade_method.execute(active_session, sample_problem)

        assert thought.metadata["cascade_level"] == "STRATEGIC"
        assert thought.metadata["level_index"] == 0
        assert thought.metadata["total_levels"] == 4
        assert thought.metadata["reasoning_type"] == "cascade_thinking"

    @pytest.mark.asyncio
    async def test_execute_includes_problem_in_metadata(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() includes input in metadata."""
        await cascade_method.initialize()
        thought = await cascade_method.execute(active_session, sample_problem)

        assert thought.metadata["input"] == sample_problem

    @pytest.mark.asyncio
    async def test_execute_content_structure(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() generates proper strategic content."""
        await cascade_method.initialize()
        thought = await cascade_method.execute(active_session, sample_problem)

        content = thought.content
        assert "STRATEGIC LEVEL" in content
        assert sample_problem in content
        assert "Vision & Goals" in content
        assert "Key Success Factors" in content

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets appropriate confidence."""
        await cascade_method.initialize()
        thought = await cascade_method.execute(active_session, sample_problem)

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.70  # Strategic level: 0.7 + (0.05 * 0)

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() adds thought to session."""
        await cascade_method.initialize()
        initial_count = active_session.thought_count

        await cascade_method.execute(active_session, sample_problem)

        assert active_session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_sets_current_method(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets session's current method."""
        await cascade_method.initialize()
        await cascade_method.execute(active_session, sample_problem)

        assert active_session.current_method == MethodIdentifier.CASCADE_THINKING


# Test Cascade Levels


class TestCascadeLevels:
    """Tests for cascade level progression."""

    @pytest.mark.asyncio
    async def test_cascade_strategic_to_tactical(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test cascading from strategic to tactical level."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)

        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        assert tactical.metadata["cascade_level"] == "TACTICAL"
        assert tactical.metadata["level_index"] == 1
        assert tactical.metadata["previous_level"] == "STRATEGIC"
        assert tactical.type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_cascade_tactical_to_operational(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test cascading from tactical to operational level."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        operational = await cascade_method.continue_reasoning(active_session, tactical)

        assert operational.metadata["cascade_level"] == "OPERATIONAL"
        assert operational.metadata["level_index"] == 2
        assert operational.metadata["previous_level"] == "TACTICAL"

    @pytest.mark.asyncio
    async def test_cascade_operational_to_detailed(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test cascading from operational to detailed level."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)

        detailed = await cascade_method.continue_reasoning(active_session, operational)

        assert detailed.metadata["cascade_level"] == "DETAILED"
        assert detailed.metadata["level_index"] == 3
        assert detailed.metadata["previous_level"] == "OPERATIONAL"
        assert detailed.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_all_four_levels_complete(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test completing all four cascade levels."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        # Should have 4 thoughts (one per level)
        assert active_session.thought_count == 4

        # Check level outputs are stored
        assert "STRATEGIC" in cascade_method._level_outputs
        assert "TACTICAL" in cascade_method._level_outputs
        assert "OPERATIONAL" in cascade_method._level_outputs
        assert "DETAILED" in cascade_method._level_outputs

    @pytest.mark.asyncio
    async def test_level_content_differences(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test that each level has distinct content."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        assert "STRATEGIC LEVEL" in strategic.content
        assert "TACTICAL LEVEL" in tactical.content
        assert "OPERATIONAL LEVEL" in operational.content
        assert "DETAILED LEVEL" in detailed.content


# Test Level Transitions


class TestLevelTransitions:
    """Tests for proper transitions between cascade levels."""

    @pytest.mark.asyncio
    async def test_parent_child_relationships(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test parent-child relationships between levels."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)

        assert tactical.parent_id == strategic.id
        assert operational.parent_id == tactical.id

    @pytest.mark.asyncio
    async def test_depth_increments_per_level(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test depth increments with each level."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        assert strategic.depth == 0
        assert tactical.depth == 1
        assert operational.depth == 2
        assert detailed.depth == 3

    @pytest.mark.asyncio
    async def test_step_counter_increments(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test step counter increments with each level."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)

        assert strategic.step_number == 1
        assert tactical.step_number == 2
        assert operational.step_number == 3

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test confidence increases as we cascade down."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        # Confidence should increase: 0.7 + (0.05 * level_index)
        assert tactical.confidence > strategic.confidence
        assert operational.confidence > tactical.confidence
        assert detailed.confidence > operational.confidence


# Test Continue Reasoning


class TestContinueReasoning:
    """Tests for continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization_raises_error(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test continue_reasoning raises error without initialization."""
        # Create a dummy thought
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CASCADE_THINKING,
            content="Test",
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await cascade_method.continue_reasoning(active_session, thought)

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test continue_reasoning with custom guidance."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)

        guidance = "Focus on scalability and performance"
        tactical = await cascade_method.continue_reasoning(
            active_session, strategic, guidance=guidance
        )

        assert tactical.metadata["guidance"] == guidance

    @pytest.mark.asyncio
    async def test_continue_without_guidance(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test continue_reasoning without guidance."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)

        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        assert tactical is not None
        assert tactical.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test continue_reasoning with context."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)

        context: dict[str, Any] = {"priority": "high", "deadline": "Q4"}
        tactical = await cascade_method.continue_reasoning(
            active_session, strategic, context=context
        )

        assert tactical.metadata["context"] == context


# Test Feedback Loops


class TestFeedbackLoops:
    """Tests for upward feedback when lower levels reveal issues."""

    @pytest.mark.asyncio
    async def test_feedback_with_keyword(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test feedback loop triggered by 'feedback' keyword."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        feedback = await cascade_method.continue_reasoning(
            active_session,
            tactical,
            guidance="Feedback: Strategic goals need adjustment based on constraints",
        )

        assert feedback.type == ThoughtType.REVISION
        assert feedback.metadata["feedback_type"] == "upward_refinement"
        assert feedback.metadata["is_revision"] is True

    @pytest.mark.asyncio
    async def test_feedback_with_revise_keyword(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test feedback loop triggered by 'revise' keyword."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)

        revision = await cascade_method.continue_reasoning(
            active_session,
            strategic,
            guidance="Revise: Need to reconsider assumptions",
        )

        assert revision.type == ThoughtType.REVISION
        assert revision.metadata["is_revision"] is True

    @pytest.mark.asyncio
    async def test_feedback_content_structure(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test feedback revision content structure."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)

        feedback_guidance = "Feedback: Operational constraints require tactical changes"
        feedback = await cascade_method.continue_reasoning(
            active_session, operational, guidance=feedback_guidance
        )

        assert "FEEDBACK REVISION" in feedback.content
        assert "Revision Analysis" in feedback.content
        assert feedback_guidance in feedback.content

    @pytest.mark.asyncio
    async def test_feedback_tracking(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test feedback items are tracked."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        assert len(cascade_method._feedback_items) == 0

        await cascade_method.continue_reasoning(
            active_session, tactical, guidance="Feedback: Test feedback"
        )

        assert len(cascade_method._feedback_items) == 1
        assert cascade_method._feedback_items[0]["from_level"] == "TACTICAL"

    @pytest.mark.asyncio
    async def test_multiple_feedback_iterations(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test multiple feedback iterations are tracked."""
        await cascade_method.initialize()
        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)

        await cascade_method.continue_reasoning(
            active_session, tactical, guidance="Feedback: First revision"
        )
        await cascade_method.continue_reasoning(
            active_session, operational, guidance="Feedback: Second revision"
        )

        assert len(cascade_method._feedback_items) == 2


# Test Final Synthesis


class TestFinalSynthesis:
    """Tests for combining all levels into final solution."""

    @pytest.mark.asyncio
    async def test_synthesis_after_all_levels(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis is created after completing all levels."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        # Try to continue past detailed - should create synthesis
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        assert synthesis.type == ThoughtType.CONCLUSION
        assert synthesis.metadata["cascade_level"] == "SYNTHESIS"

    @pytest.mark.asyncio
    async def test_synthesis_content_structure(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis content includes all levels."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        content = synthesis.content
        assert "FINAL SYNTHESIS" in content
        assert "Complete Cascade Achieved" in content
        assert "Vertical Coherence" in content
        assert "Progressive Refinement" in content

    @pytest.mark.asyncio
    async def test_synthesis_includes_level_summary(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis includes summary of all levels."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        # Should reference all completed levels
        content = synthesis.content
        assert "STRATEGIC" in content
        assert "TACTICAL" in content
        assert "OPERATIONAL" in content
        assert "DETAILED" in content

    @pytest.mark.asyncio
    async def test_synthesis_metadata(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis metadata contains summary statistics."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        assert synthesis.metadata["levels_completed"] == 4
        assert synthesis.metadata["feedback_iterations"] == 0

    @pytest.mark.asyncio
    async def test_synthesis_with_feedback(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis includes feedback iteration count."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        # Add feedback
        await cascade_method.continue_reasoning(
            active_session, tactical, guidance="Feedback: Test"
        )

        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        assert synthesis.metadata["feedback_iterations"] == 1
        assert "Feedback Iterations: 1" in synthesis.content

    @pytest.mark.asyncio
    async def test_synthesis_confidence(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test synthesis has high confidence."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)
        synthesis = await cascade_method.continue_reasoning(active_session, detailed)

        assert synthesis.confidence == 0.85


# Test Configuration


class TestConfiguration:
    """Tests for cascade_depth and refinement_threshold config options."""

    @pytest.mark.asyncio
    async def test_context_passed_to_methods(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test context is passed through cascade levels."""
        await cascade_method.initialize()

        context: dict[str, Any] = {
            "cascade_depth": 3,
            "refinement_threshold": 0.8,
        }

        strategic = await cascade_method.execute(
            active_session, sample_problem, context=context
        )

        assert strategic.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_empty_context(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execution with empty context."""
        await cascade_method.initialize()

        thought = await cascade_method.execute(
            active_session, sample_problem, context={}
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_none_context(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execution with None context."""
        await cascade_method.initialize()

        thought = await cascade_method.execute(
            active_session, sample_problem, context=None
        )

        assert thought.metadata["context"] == {}


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_problem_string(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test execution with empty problem string."""
        await cascade_method.initialize()

        thought = await cascade_method.execute(active_session, "")

        assert thought is not None
        assert thought.content != ""
        assert thought.metadata["input"] == ""

    @pytest.mark.asyncio
    async def test_very_short_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test execution with very short problem."""
        await cascade_method.initialize()

        thought = await cascade_method.execute(active_session, "Build app")

        assert thought is not None
        assert "Build app" in thought.content

    @pytest.mark.asyncio
    async def test_very_long_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        complex_problem: str,
    ):
        """Test execution with long, complex problem."""
        await cascade_method.initialize()

        thought = await cascade_method.execute(active_session, complex_problem)

        assert thought is not None
        assert complex_problem in thought.content

    @pytest.mark.asyncio
    async def test_special_characters_in_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test execution with special characters."""
        await cascade_method.initialize()

        problem = "Design: @#$%^&*() → system with émojis 🎯🚀"
        thought = await cascade_method.execute(active_session, problem)

        assert thought is not None
        assert thought.content != ""

    @pytest.mark.asyncio
    async def test_deep_cascade_chain(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test deeply cascading through all levels multiple times."""
        await cascade_method.initialize()

        # Complete full cascade
        current = await cascade_method.execute(active_session, sample_problem)
        for _ in range(3):  # Cascade through tactical, operational, detailed
            current = await cascade_method.continue_reasoning(active_session, current)

        # Create synthesis
        synthesis = await cascade_method.continue_reasoning(active_session, current)

        assert synthesis.type == ThoughtType.CONCLUSION
        assert active_session.thought_count == 5  # 4 levels + synthesis

    @pytest.mark.asyncio
    async def test_feedback_heavy_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test problem with multiple feedback iterations."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        # Multiple feedback loops
        for i in range(3):
            await cascade_method.continue_reasoning(
                active_session,
                tactical,
                guidance=f"Feedback: Iteration {i+1}",
            )

        assert len(cascade_method._feedback_items) == 3

    @pytest.mark.asyncio
    async def test_unicode_in_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test execution with Unicode characters."""
        await cascade_method.initialize()

        problem = "设计系统: Diseñar sistema with Ελληνικά characters"
        thought = await cascade_method.execute(active_session, problem)

        assert thought is not None
        assert thought.content != ""

    @pytest.mark.asyncio
    async def test_newlines_in_problem(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
    ):
        """Test execution with newlines in problem."""
        await cascade_method.initialize()

        problem = "Design:\n- Component A\n- Component B\n- Component C"
        thought = await cascade_method.execute(active_session, problem)

        assert thought is not None
        assert thought.content != ""

    @pytest.mark.asyncio
    async def test_thought_ids_are_unique(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test that all generated thoughts have unique IDs."""
        await cascade_method.initialize()

        thoughts = []
        current = await cascade_method.execute(active_session, sample_problem)
        thoughts.append(current)

        for _ in range(3):
            current = await cascade_method.continue_reasoning(active_session, current)
            thoughts.append(current)

        ids = [t.id for t in thoughts]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_reset_state_between_executions(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test state is reset for new execution."""
        await cascade_method.initialize()

        # First execution
        thought1 = await cascade_method.execute(active_session, sample_problem)
        assert cascade_method._step_counter == 1
        assert "STRATEGIC" in cascade_method._level_outputs

        # Second execution should reset
        thought2 = await cascade_method.execute(active_session, "Different problem")

        # Step counter reset to 1, then incremented
        assert thought2.step_number == 1
        assert cascade_method._current_level == "STRATEGIC"


# Test Session Integration


class TestSessionIntegration:
    """Tests for integration with Session model."""

    @pytest.mark.asyncio
    async def test_session_thought_count_updates(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session thought count updates correctly."""
        await cascade_method.initialize()

        initial_count = active_session.thought_count
        await cascade_method.execute(active_session, sample_problem)

        assert active_session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_session_metrics_update(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session metrics update after execution."""
        await cascade_method.initialize()

        await cascade_method.execute(active_session, sample_problem)

        assert active_session.metrics.total_thoughts > 0
        assert active_session.metrics.average_confidence > 0.0

    @pytest.mark.asyncio
    async def test_session_method_tracking(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session tracks method usage."""
        await cascade_method.initialize()

        await cascade_method.execute(active_session, sample_problem)

        method_key = str(MethodIdentifier.CASCADE_THINKING)
        assert method_key in active_session.metrics.methods_used
        assert active_session.metrics.methods_used[method_key] > 0

    @pytest.mark.asyncio
    async def test_session_graph_structure(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test thoughts are properly linked in session graph."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)

        # Check graph structure
        assert active_session.graph.node_count >= 2
        assert tactical.id in active_session.graph.nodes
        assert strategic.id in active_session.graph.nodes

        # Check parent-child relationship
        parent_node = active_session.graph.get_node(strategic.id)
        assert parent_node is not None
        assert tactical.id in parent_node.children_ids

    @pytest.mark.asyncio
    async def test_full_cascade_graph_structure(
        self,
        cascade_method: CascadeThinkingMethod,
        active_session: Session,
        sample_problem: str,
    ):
        """Test complete cascade creates proper graph structure."""
        await cascade_method.initialize()

        strategic = await cascade_method.execute(active_session, sample_problem)
        tactical = await cascade_method.continue_reasoning(active_session, strategic)
        operational = await cascade_method.continue_reasoning(active_session, tactical)
        detailed = await cascade_method.continue_reasoning(active_session, operational)

        # Should have 4 nodes in linear chain
        assert active_session.graph.node_count == 4

        # Check linear structure
        assert tactical.parent_id == strategic.id
        assert operational.parent_id == tactical.id
        assert detailed.parent_id == operational.id

        # Check depth progression
        assert active_session.current_depth == 3
