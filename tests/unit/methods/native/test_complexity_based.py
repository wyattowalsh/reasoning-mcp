"""Unit tests for ComplexityBased reasoning method.

This module provides comprehensive tests for the ComplexityBased method
implementation, covering initialization, execution, complexity measurement,
example selection, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.complexity_based import (
    COMPLEXITY_BASED_METADATA,
    ComplexityBased,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def cb_method() -> ComplexityBased:
    """Create a ComplexityBased method instance for testing.

    Returns:
        A fresh ComplexityBased instance
    """
    return ComplexityBased()


@pytest.fixture
def cb_no_elicitation() -> ComplexityBased:
    """Create a ComplexityBased method with elicitation disabled.

    Returns:
        A ComplexityBased instance with elicitation disabled
    """
    return ComplexityBased(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> ComplexityBased:
    """Create an initialized ComplexityBased method instance.

    Returns:
        An initialized ComplexityBased instance
    """
    method = ComplexityBased()
    await method.initialize()
    return method


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
        A sample problem string
    """
    return "Solve this multi-step math problem: If a train travels at 60 mph for 2 hours, then 80 mph for 1 hour, what is the total distance?"


class TestComplexityBasedMetadata:
    """Tests for COMPLEXITY_BASED_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert COMPLEXITY_BASED_METADATA.identifier == MethodIdentifier.COMPLEXITY_BASED

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert COMPLEXITY_BASED_METADATA.name == "Complexity-Based Prompting"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = COMPLEXITY_BASED_METADATA.description.lower()
        assert "complex" in desc
        assert "reasoning" in desc or "examples" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in SPECIALIZED category."""
        assert COMPLEXITY_BASED_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has moderate complexity."""
        assert COMPLEXITY_BASED_METADATA.complexity == 5
        assert 1 <= COMPLEXITY_BASED_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that ComplexityBased doesn't support branching."""
        assert COMPLEXITY_BASED_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that ComplexityBased doesn't support revision."""
        assert COMPLEXITY_BASED_METADATA.supports_revision is False

    def test_metadata_requires_context(self) -> None:
        """Test that ComplexityBased requires context."""
        assert COMPLEXITY_BASED_METADATA.requires_context is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert COMPLEXITY_BASED_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert COMPLEXITY_BASED_METADATA.max_thoughts == 7

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "complexity" in COMPLEXITY_BASED_METADATA.tags
        assert "few-shot" in COMPLEXITY_BASED_METADATA.tags
        assert "examples" in COMPLEXITY_BASED_METADATA.tags
        assert "multi-step" in COMPLEXITY_BASED_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(COMPLEXITY_BASED_METADATA.best_for).lower()
        assert "multi-step" in best_for_text or "complex" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(COMPLEXITY_BASED_METADATA.not_recommended_for).lower()
        assert "simple" in not_recommended


class TestComplexityBasedInitialization:
    """Tests for ComplexityBased method initialization."""

    def test_create_instance(self, cb_method: ComplexityBased) -> None:
        """Test that we can create a ComplexityBased instance."""
        assert isinstance(cb_method, ComplexityBased)

    def test_default_constants(self) -> None:
        """Test default constants."""
        assert ComplexityBased.DEFAULT_EXAMPLES == 5
        assert ComplexityBased.DEFAULT_SELECTED == 3

    def test_initial_state(self, cb_method: ComplexityBased) -> None:
        """Test that initial state is correct before initialization."""
        assert cb_method._initialized is False
        assert cb_method._step_counter == 0
        assert cb_method._current_phase == "measure"
        assert cb_method._available_examples == []
        assert cb_method._selected_examples == []

    def test_default_elicitation_enabled(self, cb_method: ComplexityBased) -> None:
        """Test that elicitation is enabled by default."""
        assert cb_method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, cb_no_elicitation: ComplexityBased) -> None:
        """Test that elicitation can be disabled."""
        assert cb_no_elicitation.enable_elicitation is False

    async def test_initialize(self, cb_method: ComplexityBased) -> None:
        """Test that initialize sets up the method correctly."""
        await cb_method.initialize()
        assert cb_method._initialized is True
        assert cb_method._step_counter == 0
        assert cb_method._current_phase == "measure"
        assert cb_method._available_examples == []
        assert cb_method._selected_examples == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = ComplexityBased()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._available_examples = [{"id": 1}]
        method._selected_examples = [{"id": 1}]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "measure"
        assert method._available_examples == []
        assert method._selected_examples == []

    async def test_health_check_before_init(self, cb_method: ComplexityBased) -> None:
        """Test health_check returns False before initialization."""
        health = await cb_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: ComplexityBased) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestComplexityBasedProperties:
    """Tests for ComplexityBased method properties."""

    def test_identifier_property(self, cb_method: ComplexityBased) -> None:
        """Test that identifier property returns correct value."""
        assert cb_method.identifier == MethodIdentifier.COMPLEXITY_BASED

    def test_name_property(self, cb_method: ComplexityBased) -> None:
        """Test that name property returns correct value."""
        assert cb_method.name == "Complexity-Based Prompting"

    def test_description_property(self, cb_method: ComplexityBased) -> None:
        """Test that description property returns correct value."""
        assert cb_method.description == COMPLEXITY_BASED_METADATA.description

    def test_category_property(self, cb_method: ComplexityBased) -> None:
        """Test that category property returns correct value."""
        assert cb_method.category == MethodCategory.SPECIALIZED


class TestComplexityBasedExecution:
    """Tests for basic execution of ComplexityBased reasoning."""

    async def test_execute_without_initialization_fails(
        self, cb_method: ComplexityBased, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await cb_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates measure phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.COMPLEXITY_BASED
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_measure(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to measure."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "measure"
        assert thought.metadata["phase"] == "measure"

    async def test_execute_creates_examples(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute creates examples with complexity metrics."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._available_examples) == 5
        for example in initialized_method._available_examples:
            assert "id" in example
            assert "steps" in example
            assert "complexity" in example

    async def test_execute_adds_to_session(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.COMPLEXITY_BASED

    async def test_execute_content_format(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert "complexity" in thought.content.lower()

    async def test_execute_confidence_level(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for measure phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.6

    async def test_execute_metadata(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "measure"
        assert thought.metadata["examples"] == 5


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, cb_method: ComplexityBased, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "measure"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await cb_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_measure_to_select(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from measure to select."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )

        assert initialized_method._current_phase == "select"
        assert select_thought.metadata["phase"] == "select"
        assert select_thought.type == ThoughtType.REASONING

    async def test_complex_examples_selected(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that high-complexity examples are selected."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )

        # Should select top 3 by complexity (steps)
        assert len(initialized_method._selected_examples) == ComplexityBased.DEFAULT_SELECTED
        # Verify they are sorted by steps (descending)
        steps = [e["steps"] for e in initialized_method._selected_examples]
        assert steps == sorted(steps, reverse=True)

    async def test_phase_transition_select_to_demonstrate(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from select to demonstrate."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )

        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        assert initialized_method._current_phase == "demonstrate"
        assert demonstrate_thought.metadata["phase"] == "demonstrate"
        assert demonstrate_thought.type == ThoughtType.REASONING

    async def test_phase_transition_demonstrate_to_reason(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from demonstrate to reason."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )
        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )

        assert initialized_method._current_phase == "reason"
        assert reason_thought.metadata["phase"] == "reason"
        assert reason_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_reason_to_conclude(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from reason to conclude."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )
        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert measure_thought.step_number == 1

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )
        assert select_thought.step_number == 2

    async def test_parent_id_set_correctly(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )

        assert select_thought.parent_id == measure_thought.id


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        measure_confidence = measure_thought.confidence

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )
        select_confidence = select_thought.confidence

        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        demonstrate_confidence = demonstrate_thought.confidence

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )
        reason_confidence = reason_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert measure_confidence < select_confidence
        assert select_confidence <= demonstrate_confidence
        assert demonstrate_confidence < reason_confidence
        assert reason_confidence <= conclude_confidence


class TestEdgeCases:
    """Tests for edge cases in ComplexityBased reasoning."""

    async def test_empty_query(self, initialized_method: ComplexityBased, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: ComplexityBased, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Solve this multi-step problem: " + "step " * 200
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: ComplexityBased, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(
        self, initialized_method: ComplexityBased, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="解决这个多步骤数学问题",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Measure
        measure_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert measure_thought.type == ThoughtType.INITIAL
        assert measure_thought.metadata["phase"] == "measure"

        # Phase 2: Select
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=measure_thought,
        )
        assert select_thought.type == ThoughtType.REASONING
        assert select_thought.metadata["phase"] == "select"

        # Phase 3: Demonstrate
        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert demonstrate_thought.type == ThoughtType.REASONING
        assert demonstrate_thought.metadata["phase"] == "demonstrate"

        # Phase 4: Reason
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )
        assert reason_thought.type == ThoughtType.SYNTHESIS
        assert reason_thought.metadata["phase"] == "reason"

        # Phase 5: Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 5
        assert session.current_method == MethodIdentifier.COMPLEXITY_BASED

    async def test_multiple_execution_cycles(
        self, initialized_method: ComplexityBased, session: Session
    ) -> None:
        """Test that method can handle multiple execution cycles."""
        # First execution
        thought1 = await initialized_method.execute(
            session=session,
            input_text="First problem",
        )
        assert thought1.step_number == 1

        # Reinitialize
        await initialized_method.initialize()

        # Second execution
        thought2 = await initialized_method.execute(
            session=session,
            input_text="Second problem",
        )
        assert thought2.step_number == 1


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.COMPLEXITY_BASED)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        cb_thoughts = session.get_thoughts_by_method(MethodIdentifier.COMPLEXITY_BASED)
        assert len(cb_thoughts) > 0


class TestElicitationBehavior:
    """Tests for elicitation-related behavior."""

    async def test_elicitation_disabled_skips_interactions(
        self, cb_no_elicitation: ComplexityBased, session: Session, sample_problem: str
    ) -> None:
        """Test that disabled elicitation skips user interactions."""
        await cb_no_elicitation.initialize()

        # Execute should work without any elicitation
        thought = await cb_no_elicitation.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
