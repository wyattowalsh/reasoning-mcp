"""Unit tests for AutoCoT reasoning method.

This module provides comprehensive tests for the AutoCoT method implementation,
covering initialization, execution, clustering, sampling, generation, and
demonstration phases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.auto_cot import (
    AUTO_COT_METADATA,
    AutoCoT,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def auto_cot_method() -> AutoCoT:
    """Create an AutoCoT method instance for testing.

    Returns:
        A fresh AutoCoT instance
    """
    return AutoCoT()


@pytest.fixture
def auto_cot_custom_clusters() -> AutoCoT:
    """Create an AutoCoT method with custom number of clusters.

    Returns:
        An AutoCoT instance with 6 clusters
    """
    return AutoCoT(num_clusters=6)


@pytest.fixture
async def initialized_method() -> AutoCoT:
    """Create an initialized AutoCoT method instance.

    Returns:
        An initialized AutoCoT instance
    """
    method = AutoCoT()
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
    return "Calculate the area of a triangle with base 10 and height 8"


class TestAutoCoTMetadata:
    """Tests for AUTO_COT_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert AUTO_COT_METADATA.identifier == MethodIdentifier.AUTO_COT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert AUTO_COT_METADATA.name == "Auto-CoT"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = AUTO_COT_METADATA.description.lower()
        assert "automatic" in desc or "auto" in desc
        assert "chain-of-thought" in desc or "cot" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in SPECIALIZED category."""
        assert AUTO_COT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity."""
        assert AUTO_COT_METADATA.complexity == 5
        assert 1 <= AUTO_COT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that Auto-CoT doesn't support branching."""
        assert AUTO_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that Auto-CoT doesn't support revision."""
        assert AUTO_COT_METADATA.supports_revision is False

    def test_metadata_requires_context(self) -> None:
        """Test that Auto-CoT requires context."""
        assert AUTO_COT_METADATA.requires_context is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert AUTO_COT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert AUTO_COT_METADATA.max_thoughts == 7

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "automatic" in AUTO_COT_METADATA.tags
        assert "clustering" in AUTO_COT_METADATA.tags
        assert "diverse" in AUTO_COT_METADATA.tags
        assert "few-shot" in AUTO_COT_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(AUTO_COT_METADATA.best_for).lower()
        assert "automated" in best_for_text or "scalable" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(AUTO_COT_METADATA.not_recommended_for).lower()
        assert "specialized" in not_recommended or "small" in not_recommended


class TestAutoCoTInitialization:
    """Tests for AutoCoT method initialization."""

    def test_create_instance(self, auto_cot_method: AutoCoT) -> None:
        """Test that we can create an AutoCoT instance."""
        assert isinstance(auto_cot_method, AutoCoT)

    def test_default_num_clusters(self, auto_cot_method: AutoCoT) -> None:
        """Test default number of clusters."""
        assert auto_cot_method._num_clusters == AutoCoT.DEFAULT_CLUSTERS

    def test_custom_num_clusters(self, auto_cot_custom_clusters: AutoCoT) -> None:
        """Test custom number of clusters."""
        assert auto_cot_custom_clusters._num_clusters == 6

    def test_initial_state(self, auto_cot_method: AutoCoT) -> None:
        """Test that initial state is correct before initialization."""
        assert auto_cot_method._initialized is False
        assert auto_cot_method._step_counter == 0
        assert auto_cot_method._current_phase == "cluster"
        assert auto_cot_method._clusters == []
        assert auto_cot_method._generated_examples == []

    async def test_initialize(self, auto_cot_method: AutoCoT) -> None:
        """Test that initialize sets up the method correctly."""
        await auto_cot_method.initialize()
        assert auto_cot_method._initialized is True
        assert auto_cot_method._step_counter == 0
        assert auto_cot_method._current_phase == "cluster"
        assert auto_cot_method._clusters == []
        assert auto_cot_method._generated_examples == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = AutoCoT()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._clusters = [{"id": 1}]
        method._generated_examples = [{"cluster": 1}]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "cluster"
        assert method._clusters == []
        assert method._generated_examples == []

    async def test_health_check_before_init(self, auto_cot_method: AutoCoT) -> None:
        """Test health_check returns False before initialization."""
        health = await auto_cot_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: AutoCoT) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestAutoCoTProperties:
    """Tests for AutoCoT method properties."""

    def test_identifier_property(self, auto_cot_method: AutoCoT) -> None:
        """Test that identifier property returns correct value."""
        assert auto_cot_method.identifier == MethodIdentifier.AUTO_COT

    def test_name_property(self, auto_cot_method: AutoCoT) -> None:
        """Test that name property returns correct value."""
        assert auto_cot_method.name == "Auto-CoT"

    def test_description_property(self, auto_cot_method: AutoCoT) -> None:
        """Test that description property returns correct value."""
        assert auto_cot_method.description == AUTO_COT_METADATA.description

    def test_category_property(self, auto_cot_method: AutoCoT) -> None:
        """Test that category property returns correct value."""
        assert auto_cot_method.category == MethodCategory.SPECIALIZED


class TestAutoCoTExecution:
    """Tests for basic execution of AutoCoT reasoning."""

    async def test_execute_without_initialization_fails(
        self, auto_cot_method: AutoCoT, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await auto_cot_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates cluster phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.AUTO_COT
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_cluster(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to cluster."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "cluster"
        assert thought.metadata["phase"] == "cluster"

    async def test_execute_generates_clusters(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates clusters."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._clusters) == AutoCoT.DEFAULT_CLUSTERS
        for cluster in initialized_method._clusters:
            assert "id" in cluster
            assert "topic" in cluster
            assert "size" in cluster

    async def test_execute_adds_to_session(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.AUTO_COT

    async def test_execute_content_format(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "Cluster" in thought.content

    async def test_execute_confidence_level(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for cluster phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.6

    async def test_execute_metadata(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "cluster"
        assert thought.metadata["clusters"] == AutoCoT.DEFAULT_CLUSTERS
        assert thought.metadata["input_text"] == sample_problem


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, auto_cot_method: AutoCoT, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        from unittest.mock import MagicMock

        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "cluster"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await auto_cot_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_cluster_to_sample(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from cluster to sample."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )

        assert initialized_method._current_phase == "sample"
        assert sample_thought.metadata["phase"] == "sample"
        assert sample_thought.type == ThoughtType.REASONING

    async def test_phase_transition_sample_to_generate(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from sample to generate."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )

        assert initialized_method._current_phase == "generate"
        assert generate_thought.metadata["phase"] == "generate"
        assert generate_thought.type == ThoughtType.REASONING

    async def test_phase_transition_generate_to_demonstrate(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from generate to demonstrate."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )

        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert initialized_method._current_phase == "demonstrate"
        assert demonstrate_thought.metadata["phase"] == "demonstrate"
        assert demonstrate_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_demonstrate_to_conclude(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from demonstrate to conclude."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )
        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert cluster_thought.step_number == 1

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        assert sample_thought.step_number == 2

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )
        assert generate_thought.step_number == 3

    async def test_parent_id_set_correctly(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )

        assert sample_thought.parent_id == cluster_thought.id

    async def test_depth_increases(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert cluster_thought.depth == 0

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        assert sample_thought.depth == 1


class TestCoTGeneration:
    """Tests for Chain-of-Thought generation."""

    async def test_generate_phase_creates_examples(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that generate phase creates examples."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )

        assert len(initialized_method._generated_examples) == len(initialized_method._clusters)

    async def test_generated_examples_have_required_fields(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that generated examples have required fields."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )

        for example in initialized_method._generated_examples:
            assert "cluster" in example
            assert "question" in example
            assert "cot" in example


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        cluster_confidence = cluster_thought.confidence

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        sample_confidence = sample_thought.confidence

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )
        generate_confidence = generate_thought.confidence

        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        demonstrate_confidence = demonstrate_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert cluster_confidence <= sample_confidence
        assert sample_confidence <= generate_confidence
        assert generate_confidence <= demonstrate_confidence
        assert demonstrate_confidence >= conclude_confidence - 0.01  # Allow small variance


class TestEdgeCases:
    """Tests for edge cases in Auto-CoT reasoning."""

    async def test_empty_query(self, initialized_method: AutoCoT, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(self, initialized_method: AutoCoT, session: Session) -> None:
        """Test handling of very long query."""
        long_query = "Analyze this problem: " + "test " * 500
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_special_characters_in_query(
        self, initialized_method: AutoCoT, session: Session
    ) -> None:
        """Test handling of special characters in query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_in_query(self, initialized_method: AutoCoT, session: Session) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="分析这个问题：什么是人工智能？",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Cluster
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert cluster_thought.type == ThoughtType.INITIAL
        assert cluster_thought.metadata["phase"] == "cluster"

        # Phase 2: Sample
        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        assert sample_thought.type == ThoughtType.REASONING
        assert sample_thought.metadata["phase"] == "sample"

        # Phase 3: Generate
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=sample_thought,
        )
        assert generate_thought.type == ThoughtType.REASONING
        assert generate_thought.metadata["phase"] == "generate"

        # Phase 4: Demonstrate
        demonstrate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        assert demonstrate_thought.type == ThoughtType.SYNTHESIS
        assert demonstrate_thought.metadata["phase"] == "demonstrate"

        # Phase 5: Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=demonstrate_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 5
        assert session.current_method == MethodIdentifier.AUTO_COT

    async def test_custom_num_clusters(self, session: Session, sample_problem: str) -> None:
        """Test with custom number of clusters."""
        method = AutoCoT(num_clusters=6)
        await method.initialize()

        await method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(method._clusters) == 6

    async def test_multiple_execution_cycles(
        self, initialized_method: AutoCoT, session: Session
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
        assert initialized_method._clusters != []


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.AUTO_COT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        auto_cot_thoughts = session.get_thoughts_by_method(MethodIdentifier.AUTO_COT)
        assert len(auto_cot_thoughts) > 0

    async def test_input_text_preserved_in_metadata(
        self, initialized_method: AutoCoT, session: Session, sample_problem: str
    ) -> None:
        """Test that input_text is preserved in metadata through phases."""
        cluster_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert cluster_thought.metadata["input_text"] == sample_problem

        sample_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=cluster_thought,
        )
        assert sample_thought.metadata["input_text"] == sample_problem
