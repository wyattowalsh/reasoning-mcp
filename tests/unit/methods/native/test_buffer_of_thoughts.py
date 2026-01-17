"""Unit tests for BufferOfThoughts reasoning method.

This module provides comprehensive tests for the BufferOfThoughts method
implementation, covering initialization, execution, template retrieval,
instantiation, reasoning, and edge cases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.buffer_of_thoughts import (
    BUFFER_OF_THOUGHTS_METADATA,
    BufferOfThoughts,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def bot_method() -> BufferOfThoughts:
    """Create a BufferOfThoughts method instance for testing.

    Returns:
        A fresh BufferOfThoughts instance
    """
    return BufferOfThoughts()


@pytest.fixture
async def initialized_method() -> BufferOfThoughts:
    """Create an initialized BufferOfThoughts method instance.

    Returns:
        An initialized BufferOfThoughts instance
    """
    method = BufferOfThoughts()
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
    return "A store sells apples at $3 each. If you buy 5 apples, how much do you pay?"


class TestBufferOfThoughtsMetadata:
    """Tests for BUFFER_OF_THOUGHTS_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert BUFFER_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.BUFFER_OF_THOUGHTS

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert BUFFER_OF_THOUGHTS_METADATA.name == "Buffer of Thoughts"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = BUFFER_OF_THOUGHTS_METADATA.description.lower()
        assert "buffer" in desc or "template" in desc
        assert "reusable" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in ADVANCED category."""
        assert BUFFER_OF_THOUGHTS_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity."""
        assert BUFFER_OF_THOUGHTS_METADATA.complexity == 7
        assert 1 <= BUFFER_OF_THOUGHTS_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that BoT supports branching."""
        assert BUFFER_OF_THOUGHTS_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that BoT supports revision."""
        assert BUFFER_OF_THOUGHTS_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that BoT doesn't require context."""
        assert BUFFER_OF_THOUGHTS_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert BUFFER_OF_THOUGHTS_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert BUFFER_OF_THOUGHTS_METADATA.max_thoughts == 10

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "buffer" in BUFFER_OF_THOUGHTS_METADATA.tags
        assert "templates" in BUFFER_OF_THOUGHTS_METADATA.tags
        assert "reusable" in BUFFER_OF_THOUGHTS_METADATA.tags
        assert "efficient" in BUFFER_OF_THOUGHTS_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(BUFFER_OF_THOUGHTS_METADATA.best_for).lower()
        assert "template" in best_for_text or "recurring" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(BUFFER_OF_THOUGHTS_METADATA.not_recommended_for).lower()
        assert "novel" in not_recommended or "unique" in not_recommended


class TestBufferOfThoughtsInitialization:
    """Tests for BufferOfThoughts method initialization."""

    def test_create_instance(self, bot_method: BufferOfThoughts) -> None:
        """Test that we can create a BufferOfThoughts instance."""
        assert isinstance(bot_method, BufferOfThoughts)

    def test_initial_state(self, bot_method: BufferOfThoughts) -> None:
        """Test that initial state is correct before initialization."""
        assert bot_method._initialized is False
        assert bot_method._step_counter == 0
        assert bot_method._current_phase == "distill"
        assert bot_method._selected_template is None
        assert bot_method._instantiated_steps == []

    def test_template_buffer_exists(self, bot_method: BufferOfThoughts) -> None:
        """Test that template buffer exists and has templates."""
        assert hasattr(BufferOfThoughts, "TEMPLATE_BUFFER")
        assert len(BufferOfThoughts.TEMPLATE_BUFFER) > 0
        for template in BufferOfThoughts.TEMPLATE_BUFFER:
            assert "id" in template
            assert "name" in template
            assert "structure" in template

    async def test_initialize(self, bot_method: BufferOfThoughts) -> None:
        """Test that initialize sets up the method correctly."""
        await bot_method.initialize()
        assert bot_method._initialized is True
        assert bot_method._step_counter == 0
        assert bot_method._current_phase == "distill"
        assert bot_method._selected_template is None
        assert bot_method._instantiated_steps == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = BufferOfThoughts()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._selected_template = {"id": "test"}
        method._instantiated_steps = ["step1", "step2"]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "distill"
        assert method._selected_template is None
        assert method._instantiated_steps == []

    async def test_health_check_before_init(self, bot_method: BufferOfThoughts) -> None:
        """Test health_check returns False before initialization."""
        health = await bot_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: BufferOfThoughts) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestBufferOfThoughtsProperties:
    """Tests for BufferOfThoughts method properties."""

    def test_identifier_property(self, bot_method: BufferOfThoughts) -> None:
        """Test that identifier property returns correct value."""
        assert bot_method.identifier == MethodIdentifier.BUFFER_OF_THOUGHTS

    def test_name_property(self, bot_method: BufferOfThoughts) -> None:
        """Test that name property returns correct value."""
        assert bot_method.name == "Buffer of Thoughts"

    def test_description_property(self, bot_method: BufferOfThoughts) -> None:
        """Test that description property returns correct value."""
        assert bot_method.description == BUFFER_OF_THOUGHTS_METADATA.description

    def test_category_property(self, bot_method: BufferOfThoughts) -> None:
        """Test that category property returns correct value."""
        assert bot_method.category == MethodCategory.ADVANCED


class TestBufferOfThoughtsExecution:
    """Tests for basic execution of BufferOfThoughts reasoning."""

    async def test_execute_without_initialization_fails(
        self, bot_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await bot_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates distill phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.BUFFER_OF_THOUGHTS
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_distill(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to distill."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "distill"
        assert thought.metadata["phase"] == "distill"

    async def test_execute_shows_templates(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute shows available templates."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["templates_available"] == len(BufferOfThoughts.TEMPLATE_BUFFER)
        assert "Templates" in thought.content or "template" in thought.content.lower()

    async def test_execute_adds_to_session(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.BUFFER_OF_THOUGHTS

    async def test_execute_content_format(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "Distill" in thought.content

    async def test_execute_confidence_level(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for distill phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.6

    async def test_execute_metadata(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "distill"
        assert thought.metadata["templates_available"] == len(BufferOfThoughts.TEMPLATE_BUFFER)
        assert thought.metadata["input_text"] == sample_problem


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, bot_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        from unittest.mock import MagicMock

        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "distill"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await bot_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_distill_to_retrieve(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from distill to retrieve."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )

        assert initialized_method._current_phase == "retrieve"
        assert retrieve_thought.metadata["phase"] == "retrieve"
        assert retrieve_thought.type == ThoughtType.REASONING

    async def test_template_selected_on_retrieve(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that a template is selected during retrieve phase."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )

        assert initialized_method._selected_template is not None
        assert "id" in initialized_method._selected_template
        assert "name" in initialized_method._selected_template
        assert "structure" in initialized_method._selected_template

    async def test_phase_transition_retrieve_to_instantiate(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from retrieve to instantiate."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )

        instantiate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )

        assert initialized_method._current_phase == "instantiate"
        assert instantiate_thought.metadata["phase"] == "instantiate"
        assert instantiate_thought.type == ThoughtType.SYNTHESIS

    async def test_steps_instantiated_on_instantiate(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that steps are instantiated during instantiate phase."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )

        assert len(initialized_method._instantiated_steps) > 0

    async def test_phase_transition_instantiate_to_reason(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from instantiate to reason."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        instantiate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=instantiate_thought,
        )

        assert initialized_method._current_phase == "reason"
        assert reason_thought.metadata["phase"] == "reason"
        assert reason_thought.type == ThoughtType.REASONING

    async def test_phase_transition_reason_to_conclude(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from reason to conclude."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        instantiate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=instantiate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert distill_thought.step_number == 1

        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        assert retrieve_thought.step_number == 2

    async def test_parent_id_set_correctly(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )

        assert retrieve_thought.parent_id == distill_thought.id

    async def test_depth_increases(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert distill_thought.depth == 0

        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        assert retrieve_thought.depth == 1


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        distill_confidence = distill_thought.confidence

        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        retrieve_confidence = retrieve_thought.confidence

        instantiate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )
        instantiate_confidence = instantiate_thought.confidence

        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=instantiate_thought,
        )
        reason_confidence = reason_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=reason_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert distill_confidence < retrieve_confidence
        assert retrieve_confidence <= instantiate_confidence
        assert instantiate_confidence < reason_confidence
        assert reason_confidence < conclude_confidence


class TestEdgeCases:
    """Tests for edge cases in BufferOfThoughts reasoning."""

    async def test_empty_query(
        self, initialized_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Analyze this problem: " + "test " * 500
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(
        self, initialized_method: BufferOfThoughts, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="分析这个问题：什么是人工智能？",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Distill
        distill_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert distill_thought.type == ThoughtType.INITIAL
        assert distill_thought.metadata["phase"] == "distill"

        # Phase 2: Retrieve
        retrieve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=distill_thought,
        )
        assert retrieve_thought.type == ThoughtType.REASONING
        assert retrieve_thought.metadata["phase"] == "retrieve"

        # Phase 3: Instantiate
        instantiate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=retrieve_thought,
        )
        assert instantiate_thought.type == ThoughtType.SYNTHESIS
        assert instantiate_thought.metadata["phase"] == "instantiate"

        # Phase 4: Reason
        reason_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=instantiate_thought,
        )
        assert reason_thought.type == ThoughtType.REASONING
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
        assert session.current_method == MethodIdentifier.BUFFER_OF_THOUGHTS

    async def test_multiple_execution_cycles(
        self, initialized_method: BufferOfThoughts, session: Session
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
        assert initialized_method._selected_template is None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.BUFFER_OF_THOUGHTS)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: BufferOfThoughts, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        bot_thoughts = session.get_thoughts_by_method(MethodIdentifier.BUFFER_OF_THOUGHTS)
        assert len(bot_thoughts) > 0
