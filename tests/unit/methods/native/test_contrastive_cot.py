"""Unit tests for ContrastiveCoT reasoning method.

This module provides comprehensive tests for the ContrastiveCoT method
implementation, covering initialization, execution, contrasting paths,
analysis, refinement, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.contrastive_cot import (
    CONTRASTIVE_COT_METADATA,
    ContrastiveCoT,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def ccot_method() -> ContrastiveCoT:
    """Create a ContrastiveCoT method instance for testing.

    Returns:
        A fresh ContrastiveCoT instance
    """
    return ContrastiveCoT()


@pytest.fixture
async def initialized_method() -> ContrastiveCoT:
    """Create an initialized ContrastiveCoT method instance.

    Returns:
        An initialized ContrastiveCoT instance
    """
    method = ContrastiveCoT()
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
    return "What is 15% of 80?"


class TestContrastiveCoTMetadata:
    """Tests for CONTRASTIVE_COT_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert CONTRASTIVE_COT_METADATA.identifier == MethodIdentifier.CONTRASTIVE_COT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert CONTRASTIVE_COT_METADATA.name == "Contrastive Chain-of-Thought"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = CONTRASTIVE_COT_METADATA.description.lower()
        assert "contrast" in desc
        assert "correct" in desc or "incorrect" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in ADVANCED category."""
        assert CONTRASTIVE_COT_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has moderate-high complexity."""
        assert CONTRASTIVE_COT_METADATA.complexity == 6
        assert 1 <= CONTRASTIVE_COT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that ContrastiveCoT supports branching."""
        assert CONTRASTIVE_COT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that ContrastiveCoT supports revision."""
        assert CONTRASTIVE_COT_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that ContrastiveCoT doesn't require context."""
        assert CONTRASTIVE_COT_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert CONTRASTIVE_COT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert CONTRASTIVE_COT_METADATA.max_thoughts == 8

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "contrastive" in CONTRASTIVE_COT_METADATA.tags
        assert "error-analysis" in CONTRASTIVE_COT_METADATA.tags
        assert "negative-examples" in CONTRASTIVE_COT_METADATA.tags
        assert "accuracy" in CONTRASTIVE_COT_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(CONTRASTIVE_COT_METADATA.best_for).lower()
        assert "error" in best_for_text or "logical" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(CONTRASTIVE_COT_METADATA.not_recommended_for).lower()
        assert "creative" in not_recommended or "subjective" in not_recommended


class TestContrastiveCoTInitialization:
    """Tests for ContrastiveCoT method initialization."""

    def test_create_instance(self, ccot_method: ContrastiveCoT) -> None:
        """Test that we can create a ContrastiveCoT instance."""
        assert isinstance(ccot_method, ContrastiveCoT)

    def test_max_contrasts(self) -> None:
        """Test max contrasts constant."""
        assert ContrastiveCoT.MAX_CONTRASTS == 2

    def test_initial_state(self, ccot_method: ContrastiveCoT) -> None:
        """Test that initial state is correct before initialization."""
        assert ccot_method._initialized is False
        assert ccot_method._step_counter == 0
        assert ccot_method._current_phase == "generate"
        assert ccot_method._correct_path == ""
        assert ccot_method._incorrect_paths == []
        assert ccot_method._contrast_count == 0

    async def test_initialize(self, ccot_method: ContrastiveCoT) -> None:
        """Test that initialize sets up the method correctly."""
        await ccot_method.initialize()
        assert ccot_method._initialized is True
        assert ccot_method._step_counter == 0
        assert ccot_method._current_phase == "generate"
        assert ccot_method._correct_path == ""
        assert ccot_method._incorrect_paths == []
        assert ccot_method._contrast_count == 0

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = ContrastiveCoT()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._correct_path = "correct reasoning"
        method._incorrect_paths = ["wrong 1", "wrong 2"]
        method._contrast_count = 2

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._correct_path == ""
        assert method._incorrect_paths == []
        assert method._contrast_count == 0

    async def test_health_check_before_init(self, ccot_method: ContrastiveCoT) -> None:
        """Test health_check returns False before initialization."""
        health = await ccot_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: ContrastiveCoT) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestContrastiveCoTProperties:
    """Tests for ContrastiveCoT method properties."""

    def test_identifier_property(self, ccot_method: ContrastiveCoT) -> None:
        """Test that identifier property returns correct value."""
        assert ccot_method.identifier == MethodIdentifier.CONTRASTIVE_COT

    def test_name_property(self, ccot_method: ContrastiveCoT) -> None:
        """Test that name property returns correct value."""
        assert ccot_method.name == "Contrastive Chain-of-Thought"

    def test_description_property(self, ccot_method: ContrastiveCoT) -> None:
        """Test that description property returns correct value."""
        assert ccot_method.description == CONTRASTIVE_COT_METADATA.description

    def test_category_property(self, ccot_method: ContrastiveCoT) -> None:
        """Test that category property returns correct value."""
        assert ccot_method.category == MethodCategory.ADVANCED


class TestContrastiveCoTExecution:
    """Tests for basic execution of ContrastiveCoT reasoning."""

    async def test_execute_without_initialization_fails(
        self, ccot_method: ContrastiveCoT, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ccot_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates generate phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CONTRASTIVE_COT
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_generate(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to generate."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "generate"
        assert thought.metadata["phase"] == "generate"

    async def test_execute_creates_correct_path(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute creates correct reasoning path."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._correct_path != ""
        assert "CORRECT" in initialized_method._correct_path

    async def test_execute_adds_to_session(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.CONTRASTIVE_COT

    async def test_execute_content_format(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert "Correct" in thought.content or "CORRECT" in thought.content

    async def test_execute_confidence_level(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for generate phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.75

    async def test_execute_metadata(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "generate"
        assert thought.metadata["path_type"] == "correct"
        assert thought.metadata["contrast_count"] == 0


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, ccot_method: ContrastiveCoT, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "generate"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await ccot_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_generate_to_contrast(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from generate to contrast."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        contrast_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert initialized_method._current_phase == "contrast"
        assert contrast_thought.metadata["phase"] == "contrast"
        assert contrast_thought.type == ThoughtType.HYPOTHESIS

    async def test_contrast_creates_incorrect_path(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that contrast phase creates incorrect path."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert len(initialized_method._incorrect_paths) == 1
        assert initialized_method._contrast_count == 1

    async def test_multiple_contrasts(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test generating multiple contrasting paths."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        assert initialized_method._contrast_count == 1

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )
        assert initialized_method._contrast_count == 2

    async def test_phase_transition_contrast_to_analyze(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from contrast to analyze after max contrasts."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        contrast2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )

        analyze_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast2,
        )

        assert initialized_method._current_phase == "analyze"
        assert analyze_thought.metadata["phase"] == "analyze"
        assert analyze_thought.type == ThoughtType.REASONING

    async def test_phase_transition_analyze_to_refine(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from analyze to refine."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        contrast2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )
        analyze_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast2,
        )

        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=analyze_thought,
        )

        assert initialized_method._current_phase == "refine"
        assert refine_thought.metadata["phase"] == "refine"
        assert refine_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_refine_to_conclude(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from refine to conclude."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        contrast2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )
        analyze_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast2,
        )
        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=analyze_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert generate_thought.step_number == 1

        contrast_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        assert contrast_thought.step_number == 2

    async def test_parent_id_set_correctly(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        contrast_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert contrast_thought.parent_id == generate_thought.id


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_varies_by_path_type(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence is lower for incorrect paths."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        correct_confidence = generate_thought.confidence

        contrast_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        incorrect_confidence = contrast_thought.confidence

        # Incorrect paths should have lower confidence
        assert incorrect_confidence < correct_confidence

    async def test_confidence_increases_after_analysis(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence increases after contrastive analysis."""
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        contrast2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )

        analyze_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast2,
        )

        assert analyze_thought.confidence > contrast2.confidence


class TestEdgeCases:
    """Tests for edge cases in ContrastiveCoT reasoning."""

    async def test_empty_query(self, initialized_method: ContrastiveCoT, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: ContrastiveCoT, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Calculate: " + "x + " * 100 + "y"
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: ContrastiveCoT, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(
        self, initialized_method: ContrastiveCoT, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="计算 80 的 15%",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Generate correct path
        generate_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert generate_thought.type == ThoughtType.INITIAL
        assert generate_thought.metadata["phase"] == "generate"

        # Phase 2: First contrast (incorrect path)
        contrast1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        assert contrast1.type == ThoughtType.HYPOTHESIS
        assert contrast1.metadata["phase"] == "contrast"

        # Phase 3: Second contrast (another incorrect path)
        contrast2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast1,
        )
        assert contrast2.type == ThoughtType.HYPOTHESIS
        assert contrast2.metadata["phase"] == "contrast"

        # Phase 4: Analyze
        analyze_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=contrast2,
        )
        assert analyze_thought.type == ThoughtType.REASONING
        assert analyze_thought.metadata["phase"] == "analyze"

        # Phase 5: Refine
        refine_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=analyze_thought,
        )
        assert refine_thought.type == ThoughtType.SYNTHESIS
        assert refine_thought.metadata["phase"] == "refine"

        # Phase 6: Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=refine_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 6
        assert session.current_method == MethodIdentifier.CONTRASTIVE_COT

    async def test_multiple_execution_cycles(
        self, initialized_method: ContrastiveCoT, session: Session
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
        assert initialized_method._correct_path != ""
        assert initialized_method._incorrect_paths == []


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.CONTRASTIVE_COT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: ContrastiveCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        ccot_thoughts = session.get_thoughts_by_method(MethodIdentifier.CONTRASTIVE_COT)
        assert len(ccot_thoughts) > 0
