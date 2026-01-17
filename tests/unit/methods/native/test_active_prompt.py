"""Unit tests for ActivePrompt reasoning method.

This module provides comprehensive tests for the ActivePrompt method implementation,
covering initialization, execution, phase transitions, uncertainty-based selection,
elicitation, and edge cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.active_prompt import (
    ACTIVE_PROMPT_METADATA,
    ActivePrompt,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def active_prompt_method() -> ActivePrompt:
    """Create an ActivePrompt method instance for testing.

    Returns:
        A fresh ActivePrompt instance
    """
    return ActivePrompt()


@pytest.fixture
def method_no_elicitation() -> ActivePrompt:
    """Create an ActivePrompt method with elicitation disabled.

    Returns:
        An ActivePrompt instance with elicitation disabled
    """
    return ActivePrompt(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> ActivePrompt:
    """Create an initialized ActivePrompt method instance.

    Returns:
        An initialized ActivePrompt instance
    """
    method = ActivePrompt()
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
    return "Solve the equation: 3x + 7 = 22"


class TestActivePromptMetadata:
    """Tests for ACTIVE_PROMPT_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert ACTIVE_PROMPT_METADATA.identifier == MethodIdentifier.ACTIVE_PROMPT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert ACTIVE_PROMPT_METADATA.name == "Active Prompt"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = ACTIVE_PROMPT_METADATA.description.lower()
        assert "uncertainty" in desc
        assert "examples" in desc or "demonstrations" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in SPECIALIZED category."""
        assert ACTIVE_PROMPT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity."""
        assert ACTIVE_PROMPT_METADATA.complexity == 5
        assert 1 <= ACTIVE_PROMPT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that Active Prompt doesn't support branching."""
        assert ACTIVE_PROMPT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that Active Prompt doesn't support revision."""
        assert ACTIVE_PROMPT_METADATA.supports_revision is False

    def test_metadata_requires_context(self) -> None:
        """Test that Active Prompt requires context."""
        assert ACTIVE_PROMPT_METADATA.requires_context is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert ACTIVE_PROMPT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert ACTIVE_PROMPT_METADATA.max_thoughts == 7

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "active-learning" in ACTIVE_PROMPT_METADATA.tags
        assert "uncertainty" in ACTIVE_PROMPT_METADATA.tags
        assert "few-shot" in ACTIVE_PROMPT_METADATA.tags
        assert "selection" in ACTIVE_PROMPT_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(ACTIVE_PROMPT_METADATA.best_for).lower()
        assert "few-shot" in best_for_text or "example" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(ACTIVE_PROMPT_METADATA.not_recommended_for).lower()
        assert "zero-shot" in not_recommended or "simple" in not_recommended


class TestActivePromptInitialization:
    """Tests for ActivePrompt method initialization."""

    def test_create_instance(self, active_prompt_method: ActivePrompt) -> None:
        """Test that we can create an ActivePrompt instance."""
        assert isinstance(active_prompt_method, ActivePrompt)

    def test_initial_state(self, active_prompt_method: ActivePrompt) -> None:
        """Test that initial state is correct before initialization."""
        assert active_prompt_method._initialized is False
        assert active_prompt_method._step_counter == 0
        assert active_prompt_method._current_phase == "query"
        assert active_prompt_method._candidate_examples == []
        assert active_prompt_method._selected_examples == []
        assert active_prompt_method._uncertainty_scores == []

    def test_default_elicitation_enabled(self, active_prompt_method: ActivePrompt) -> None:
        """Test that elicitation is enabled by default."""
        assert active_prompt_method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, method_no_elicitation: ActivePrompt) -> None:
        """Test that elicitation can be disabled."""
        assert method_no_elicitation.enable_elicitation is False

    async def test_initialize(self, active_prompt_method: ActivePrompt) -> None:
        """Test that initialize sets up the method correctly."""
        await active_prompt_method.initialize()
        assert active_prompt_method._initialized is True
        assert active_prompt_method._step_counter == 0
        assert active_prompt_method._current_phase == "query"
        assert active_prompt_method._candidate_examples == []
        assert active_prompt_method._selected_examples == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = ActivePrompt()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._candidate_examples = [{"id": 1}]
        method._selected_examples = [{"id": 1}]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "query"
        assert method._candidate_examples == []
        assert method._selected_examples == []

    async def test_health_check_before_init(self, active_prompt_method: ActivePrompt) -> None:
        """Test health_check returns False before initialization."""
        health = await active_prompt_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: ActivePrompt) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestActivePromptProperties:
    """Tests for ActivePrompt method properties."""

    def test_identifier_property(self, active_prompt_method: ActivePrompt) -> None:
        """Test that identifier property returns correct value."""
        assert active_prompt_method.identifier == MethodIdentifier.ACTIVE_PROMPT

    def test_name_property(self, active_prompt_method: ActivePrompt) -> None:
        """Test that name property returns correct value."""
        assert active_prompt_method.name == "Active Prompt"

    def test_description_property(self, active_prompt_method: ActivePrompt) -> None:
        """Test that description property returns correct value."""
        assert active_prompt_method.description == ACTIVE_PROMPT_METADATA.description

    def test_category_property(self, active_prompt_method: ActivePrompt) -> None:
        """Test that category property returns correct value."""
        assert active_prompt_method.category == MethodCategory.SPECIALIZED


class TestActivePromptExecution:
    """Tests for basic execution of ActivePrompt reasoning."""

    async def test_execute_without_initialization_fails(
        self, active_prompt_method: ActivePrompt, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await active_prompt_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates query phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.ACTIVE_PROMPT
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_query(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to query."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "query"
        assert thought.metadata["phase"] == "query"

    async def test_execute_generates_candidates(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates candidate examples."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._candidate_examples) == ActivePrompt.DEFAULT_CANDIDATES
        for candidate in initialized_method._candidate_examples:
            assert "id" in candidate
            assert "question" in candidate
            assert "uncertainty" in candidate

    async def test_execute_generates_uncertainty_scores(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates uncertainty scores."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._uncertainty_scores) == ActivePrompt.DEFAULT_CANDIDATES
        for score in initialized_method._uncertainty_scores:
            assert 0.0 <= score <= 1.0

    async def test_execute_adds_to_session(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.ACTIVE_PROMPT

    async def test_execute_content_format(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "uncertainty" in thought.content.lower()

    async def test_execute_confidence_level(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for query phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.5

    async def test_execute_metadata(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "query"
        assert thought.metadata["candidates"] == ActivePrompt.DEFAULT_CANDIDATES
        assert thought.metadata["input"] == sample_problem


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, active_prompt_method: ActivePrompt, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "query"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await active_prompt_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_query_to_select(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from query to select."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )

        assert initialized_method._current_phase == "select"
        assert select_thought.metadata["phase"] == "select"
        assert select_thought.type == ThoughtType.REASONING

    async def test_phase_transition_select_to_annotate(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from select to annotate."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )

        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        assert initialized_method._current_phase == "annotate"
        assert annotate_thought.metadata["phase"] == "annotate"
        assert annotate_thought.type == ThoughtType.REASONING

    async def test_phase_transition_annotate_to_reason(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from annotate to reason."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=annotate_thought,
        )

        assert initialized_method._current_phase == "reason"
        assert reason_thought.metadata["phase"] == "reason"
        assert reason_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_reason_to_conclude(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from reason to conclude."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=annotate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert query_thought.step_number == 1

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        assert select_thought.step_number == 2

        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert annotate_thought.step_number == 3

    async def test_parent_id_set_correctly(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )

        assert select_thought.parent_id == query_thought.id

    async def test_depth_increases(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert query_thought.depth == 0

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        assert select_thought.depth == 1


class TestUncertaintyBasedSelection:
    """Tests for uncertainty-based example selection."""

    async def test_examples_sorted_by_uncertainty(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that examples are sorted by uncertainty during selection."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )

        # Selected examples should have highest uncertainty
        selected_uncertainties = [e["uncertainty"] for e in initialized_method._selected_examples]
        # Verify selected are among highest uncertainty
        all_uncertainties = [e["uncertainty"] for e in initialized_method._candidate_examples]
        sorted_uncertainties = sorted(all_uncertainties, reverse=True)
        top_n = sorted_uncertainties[: ActivePrompt.DEFAULT_SELECTED]

        for selected in selected_uncertainties:
            assert selected in top_n

    async def test_correct_number_selected(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that correct number of examples are selected."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )

        assert len(initialized_method._selected_examples) == ActivePrompt.DEFAULT_SELECTED


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        query_confidence = query_thought.confidence

        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        select_confidence = select_thought.confidence

        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        annotate_confidence = annotate_thought.confidence

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=annotate_thought,
        )
        reason_confidence = reason_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert query_confidence < select_confidence
        assert select_confidence < annotate_confidence
        assert annotate_confidence <= reason_confidence
        assert conclude_confidence >= query_confidence


class TestEdgeCases:
    """Tests for edge cases in Active Prompt reasoning."""

    async def test_empty_query(self, initialized_method: ActivePrompt, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: ActivePrompt, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Analyze this problem: " + "test " * 500
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_special_characters_in_query(
        self, initialized_method: ActivePrompt, session: Session
    ) -> None:
        """Test handling of special characters in query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_in_query(
        self, initialized_method: ActivePrompt, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="分析这个问题：什么是人工智能？",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Query
        query_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert query_thought.type == ThoughtType.INITIAL
        assert query_thought.metadata["phase"] == "query"

        # Phase 2: Select
        select_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=query_thought,
        )
        assert select_thought.type == ThoughtType.REASONING
        assert select_thought.metadata["phase"] == "select"

        # Phase 3: Annotate
        annotate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert annotate_thought.type == ThoughtType.REASONING
        assert annotate_thought.metadata["phase"] == "annotate"

        # Phase 4: Reason
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=annotate_thought,
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
        assert session.current_method == MethodIdentifier.ACTIVE_PROMPT

    async def test_context_passed_through(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that context parameters are passed through."""
        context: dict[str, Any] = {"custom_param": "value"}

        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )

        assert thought is not None

    async def test_multiple_execution_cycles(
        self, initialized_method: ActivePrompt, session: Session
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
        assert initialized_method._candidate_examples != []


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.ACTIVE_PROMPT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        active_prompt_thoughts = session.get_thoughts_by_method(MethodIdentifier.ACTIVE_PROMPT)
        assert len(active_prompt_thoughts) > 0


class TestElicitationBehavior:
    """Tests for elicitation-related behavior."""

    async def test_elicitation_disabled_skips_interactions(
        self, method_no_elicitation: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that disabled elicitation skips user interactions."""
        await method_no_elicitation.initialize()

        # Execute should work without any elicitation
        thought = await method_no_elicitation.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None

    async def test_elicitation_context_not_set_by_default(
        self, initialized_method: ActivePrompt, session: Session, sample_problem: str
    ) -> None:
        """Test that elicitation context is not set without execution context."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._ctx is None
