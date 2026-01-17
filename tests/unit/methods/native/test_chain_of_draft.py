"""Unit tests for ChainOfDraft reasoning method.

This module provides comprehensive tests for the ChainOfDraft method
implementation, covering initialization, execution, draft generation,
refinement, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.chain_of_draft import (
    CHAIN_OF_DRAFT_METADATA,
    ChainOfDraft,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def cod_method() -> ChainOfDraft:
    """Create a ChainOfDraft method instance for testing.

    Returns:
        A fresh ChainOfDraft instance
    """
    return ChainOfDraft()


@pytest.fixture
async def initialized_method() -> ChainOfDraft:
    """Create an initialized ChainOfDraft method instance.

    Returns:
        An initialized ChainOfDraft instance
    """
    method = ChainOfDraft()
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
    return "What is 5 * 3 + 2?"


class TestChainOfDraftMetadata:
    """Tests for CHAIN_OF_DRAFT_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert CHAIN_OF_DRAFT_METADATA.identifier == MethodIdentifier.CHAIN_OF_DRAFT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert CHAIN_OF_DRAFT_METADATA.name == "Chain of Draft"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = CHAIN_OF_DRAFT_METADATA.description.lower()
        assert "concise" in desc or "draft" in desc
        assert "latency" in desc or "reduction" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in SPECIALIZED category."""
        assert CHAIN_OF_DRAFT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has low complexity (efficient method)."""
        assert CHAIN_OF_DRAFT_METADATA.complexity == 4
        assert 1 <= CHAIN_OF_DRAFT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that CoD doesn't support branching."""
        assert CHAIN_OF_DRAFT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that CoD supports revision."""
        assert CHAIN_OF_DRAFT_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that CoD doesn't require context."""
        assert CHAIN_OF_DRAFT_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert CHAIN_OF_DRAFT_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert CHAIN_OF_DRAFT_METADATA.max_thoughts == 6

    def test_metadata_avg_tokens_per_thought(self) -> None:
        """Test that metadata specifies low average tokens (efficient)."""
        assert CHAIN_OF_DRAFT_METADATA.avg_tokens_per_thought == 80

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "efficient" in CHAIN_OF_DRAFT_METADATA.tags
        assert "concise" in CHAIN_OF_DRAFT_METADATA.tags
        assert "draft" in CHAIN_OF_DRAFT_METADATA.tags
        assert "fast" in CHAIN_OF_DRAFT_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(CHAIN_OF_DRAFT_METADATA.best_for).lower()
        assert "latency" in best_for_text or "quick" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(CHAIN_OF_DRAFT_METADATA.not_recommended_for).lower()
        assert "complex" in not_recommended or "detailed" in not_recommended


class TestChainOfDraftInitialization:
    """Tests for ChainOfDraft method initialization."""

    def test_create_instance(self, cod_method: ChainOfDraft) -> None:
        """Test that we can create a ChainOfDraft instance."""
        assert isinstance(cod_method, ChainOfDraft)

    def test_initial_state(self, cod_method: ChainOfDraft) -> None:
        """Test that initial state is correct before initialization."""
        assert cod_method._initialized is False
        assert cod_method._step_counter == 0
        assert cod_method._current_phase == "draft"
        assert cod_method._drafts == []
        assert cod_method._refined_steps == []

    async def test_initialize(self, cod_method: ChainOfDraft) -> None:
        """Test that initialize sets up the method correctly."""
        await cod_method.initialize()
        assert cod_method._initialized is True
        assert cod_method._step_counter == 0
        assert cod_method._current_phase == "draft"
        assert cod_method._drafts == []
        assert cod_method._refined_steps == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = ChainOfDraft()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._drafts = ["draft1", "draft2"]
        method._refined_steps = ["refined1"]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "draft"
        assert method._drafts == []
        assert method._refined_steps == []

    async def test_health_check_before_init(self, cod_method: ChainOfDraft) -> None:
        """Test health_check returns False before initialization."""
        health = await cod_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: ChainOfDraft) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestChainOfDraftProperties:
    """Tests for ChainOfDraft method properties."""

    def test_identifier_property(self, cod_method: ChainOfDraft) -> None:
        """Test that identifier property returns correct value."""
        assert cod_method.identifier == MethodIdentifier.CHAIN_OF_DRAFT

    def test_name_property(self, cod_method: ChainOfDraft) -> None:
        """Test that name property returns correct value."""
        assert cod_method.name == "Chain of Draft"

    def test_description_property(self, cod_method: ChainOfDraft) -> None:
        """Test that description property returns correct value."""
        assert cod_method.description == CHAIN_OF_DRAFT_METADATA.description

    def test_category_property(self, cod_method: ChainOfDraft) -> None:
        """Test that category property returns correct value."""
        assert cod_method.category == MethodCategory.SPECIALIZED


class TestChainOfDraftExecution:
    """Tests for basic execution of ChainOfDraft reasoning."""

    async def test_execute_without_initialization_fails(
        self, cod_method: ChainOfDraft, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await cod_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates draft phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CHAIN_OF_DRAFT
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_draft(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to draft."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "draft"
        assert thought.metadata["phase"] == "draft"

    async def test_execute_generates_drafts(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates telegraphic drafts."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._drafts) > 0
        # Drafts should be concise (approximately 5 words)
        for draft in initialized_method._drafts:
            assert len(draft.split()) <= 10  # Allow some flexibility

    async def test_execute_adds_to_session(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.CHAIN_OF_DRAFT

    async def test_execute_content_format(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "Draft" in thought.content

    async def test_execute_confidence_level(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for draft phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.7

    async def test_execute_metadata(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "draft"
        assert thought.metadata["draft_count"] > 0


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, cod_method: ChainOfDraft, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "draft"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await cod_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_draft_to_refine(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from draft to refine."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )

        assert initialized_method._current_phase == "refine"
        assert refine_thought.metadata["phase"] == "refine"
        assert refine_thought.type == ThoughtType.REVISION

    async def test_refinements_generated(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that refinements are generated during refine phase."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )

        assert len(initialized_method._refined_steps) > 0

    async def test_phase_transition_refine_to_answer(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from refine to answer."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )

        answer_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )

        assert initialized_method._current_phase == "answer"
        assert answer_thought.metadata["phase"] == "answer"
        assert answer_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert draft_thought.step_number == 1

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )
        assert refine_thought.step_number == 2

        answer_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )
        assert answer_thought.step_number == 3

    async def test_parent_id_set_correctly(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )

        assert refine_thought.parent_id == draft_thought.id

    async def test_depth_increases(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert draft_thought.depth == 0

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )
        assert refine_thought.depth == 1


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        draft_confidence = draft_thought.confidence

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )
        refine_confidence = refine_thought.confidence

        answer_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )
        answer_confidence = answer_thought.confidence

        # Confidence should generally increase
        assert draft_confidence < refine_confidence
        assert refine_confidence < answer_confidence


class TestEdgeCases:
    """Tests for edge cases in ChainOfDraft reasoning."""

    async def test_empty_query(self, initialized_method: ChainOfDraft, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: ChainOfDraft, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Compute: " + "x + " * 100 + "y"
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: ChainOfDraft, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(self, initialized_method: ChainOfDraft, session: Session) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="计算 5 * 3 + 2",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Draft
        draft_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert draft_thought.type == ThoughtType.INITIAL
        assert draft_thought.metadata["phase"] == "draft"

        # Phase 2: Refine
        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=draft_thought,
        )
        assert refine_thought.type == ThoughtType.REVISION
        assert refine_thought.metadata["phase"] == "refine"

        # Phase 3: Answer
        answer_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )
        assert answer_thought.type == ThoughtType.CONCLUSION
        assert answer_thought.metadata["phase"] == "answer"

        # Verify session state
        assert session.thought_count == 3
        assert session.current_method == MethodIdentifier.CHAIN_OF_DRAFT

    async def test_multiple_execution_cycles(
        self, initialized_method: ChainOfDraft, session: Session
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
        assert initialized_method._drafts != []


class TestDraftGeneration:
    """Tests for draft generation behavior."""

    async def test_drafts_are_concise(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that generated drafts are concise (telegraphic)."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        total_words = sum(len(draft.split()) for draft in initialized_method._drafts)
        avg_words = total_words / len(initialized_method._drafts)
        # Average should be around 5 words per draft (allowing flexibility)
        assert avg_words <= 10

    async def test_content_mentions_latency_reduction(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that content mentions latency reduction benefit."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "latency" in thought.content.lower() or "76%" in thought.content


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.CHAIN_OF_DRAFT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: ChainOfDraft, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        cod_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_DRAFT)
        assert len(cod_thoughts) > 0
