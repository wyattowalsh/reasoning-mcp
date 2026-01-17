"""Unit tests for AGoT (Adaptive Graph of Thoughts) reasoning method.

This module provides comprehensive tests for the AGoT method implementation,
covering initialization, execution, graph construction, adaptation cycles,
propagation, synthesis, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.agot import (
    AGOT_METADATA,
    AGoT,
    GraphNode,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def agot_method() -> AGoT:
    """Create an AGoT method instance for testing.

    Returns:
        A fresh AGoT instance
    """
    return AGoT()


@pytest.fixture
async def initialized_method() -> AGoT:
    """Create an initialized AGoT method instance.

    Returns:
        An initialized AGoT instance
    """
    method = AGoT()
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
    return "Analyze the relationship between AI safety and alignment"


@pytest.fixture
def complex_problem() -> str:
    """Provide a complex problem for testing.

    Returns:
        A complex problem string
    """
    return (
        "Design a comprehensive system for autonomous vehicle decision-making that "
        "balances safety, efficiency, regulatory compliance, and ethical considerations"
    )


class TestAGoTMetadata:
    """Tests for AGOT_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert AGOT_METADATA.identifier == MethodIdentifier.AGOT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert AGOT_METADATA.name == "Adaptive Graph of Thoughts (AGoT)"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = AGOT_METADATA.description.lower()
        assert "adaptive" in desc or "adapts" in desc
        assert "graph" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in ADVANCED category."""
        assert AGOT_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has high complexity (9)."""
        assert AGOT_METADATA.complexity == 9
        assert 1 <= AGOT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that AGoT supports branching."""
        assert AGOT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that AGoT supports revision."""
        assert AGOT_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that AGoT doesn't require context."""
        assert AGOT_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert AGOT_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert AGOT_METADATA.max_thoughts == 40

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "adaptive" in AGOT_METADATA.tags
        assert "graph-based" in AGOT_METADATA.tags
        assert "dynamic-structure" in AGOT_METADATA.tags
        assert "confidence-driven" in AGOT_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(AGOT_METADATA.best_for).lower()
        assert "complex" in best_for_text or "multi" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(AGOT_METADATA.not_recommended_for).lower()
        assert "simple" in not_recommended or "linear" in not_recommended


class TestGraphNode:
    """Tests for GraphNode class."""

    def test_create_graph_node(self) -> None:
        """Test creating a basic graph node."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test thought",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought, confidence=0.7)

        assert node.id == "node_1"
        assert node.thought == thought
        assert node.confidence == 0.7
        assert node.edges_to == {}
        assert node.edges_from == {}
        assert node.visited is False
        assert node.activation == 0.0

    def test_default_confidence(self) -> None:
        """Test default confidence value."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        assert node.confidence == 0.5

    def test_add_edge_to(self) -> None:
        """Test adding outgoing edge."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)

        node.add_edge_to("node_2", weight=0.8)

        assert "node_2" in node.edges_to
        assert node.edges_to["node_2"] == 0.8

    def test_add_edge_from(self) -> None:
        """Test adding incoming edge."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)

        node.add_edge_from("node_0", weight=0.6)

        assert "node_0" in node.edges_from
        assert node.edges_from["node_0"] == 0.6

    def test_remove_edge_to(self) -> None:
        """Test removing outgoing edge."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        node.add_edge_to("node_2", weight=0.8)

        node.remove_edge_to("node_2")

        assert "node_2" not in node.edges_to

    def test_remove_edge_from(self) -> None:
        """Test removing incoming edge."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        node.add_edge_from("node_0", weight=0.6)

        node.remove_edge_from("node_0")

        assert "node_0" not in node.edges_from

    def test_remove_nonexistent_edge(self) -> None:
        """Test removing edge that doesn't exist (should not raise)."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)

        # Should not raise
        node.remove_edge_to("nonexistent")
        node.remove_edge_from("nonexistent")

    def test_degree_property(self) -> None:
        """Test degree property calculation."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        node.add_edge_to("node_2")
        node.add_edge_to("node_3")
        node.add_edge_from("node_0")

        assert node.degree == 3

    def test_out_degree_property(self) -> None:
        """Test out_degree property."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        node.add_edge_to("node_2")
        node.add_edge_to("node_3")

        assert node.out_degree == 2

    def test_in_degree_property(self) -> None:
        """Test in_degree property."""
        thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
        )
        node = GraphNode("node_1", thought)
        node.add_edge_from("node_0")
        node.add_edge_from("node_3")

        assert node.in_degree == 2


class TestAGoTInitialization:
    """Tests for AGoT method initialization."""

    def test_create_instance(self, agot_method: AGoT) -> None:
        """Test that we can create an AGoT instance."""
        assert isinstance(agot_method, AGoT)

    def test_initial_state(self, agot_method: AGoT) -> None:
        """Test that initial state is correct before initialization."""
        assert agot_method._initialized is False
        assert agot_method._step_counter == 0
        assert agot_method._graph_nodes == {}
        assert agot_method._current_phase == "initialize"
        assert agot_method._adaptation_cycle == 0
        assert agot_method._nodes_added == 0
        assert agot_method._nodes_removed == 0
        assert agot_method._edges_added == 0
        assert agot_method._edges_removed == 0
        assert agot_method._confidence_history == []

    async def test_initialize(self, agot_method: AGoT) -> None:
        """Test that initialize sets up the method correctly."""
        await agot_method.initialize()
        assert agot_method._initialized is True
        assert agot_method._step_counter == 0
        assert agot_method._current_phase == "initialize"
        assert agot_method._graph_nodes == {}

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = AGoT()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._graph_nodes = {"test": None}  # type: ignore
        method._adaptation_cycle = 3
        method._nodes_added = 10
        method._confidence_history = [0.5, 0.6, 0.7]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "initialize"
        assert method._graph_nodes == {}
        assert method._adaptation_cycle == 0
        assert method._nodes_added == 0
        assert method._confidence_history == []

    async def test_health_check_before_init(self, agot_method: AGoT) -> None:
        """Test health_check returns False before initialization."""
        health = await agot_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: AGoT) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestAGoTProperties:
    """Tests for AGoT method properties."""

    def test_identifier_property(self, agot_method: AGoT) -> None:
        """Test that identifier property returns correct value."""
        assert agot_method.identifier == MethodIdentifier.AGOT

    def test_name_property(self, agot_method: AGoT) -> None:
        """Test that name property returns correct value."""
        assert agot_method.name == "Adaptive Graph of Thoughts (AGoT)"

    def test_description_property(self, agot_method: AGoT) -> None:
        """Test that description property returns correct value."""
        assert agot_method.description == AGOT_METADATA.description

    def test_category_property(self, agot_method: AGoT) -> None:
        """Test that category property returns correct value."""
        assert agot_method.category == MethodCategory.ADVANCED


class TestAGoTExecution:
    """Tests for basic execution of AGoT reasoning."""

    async def test_execute_without_initialization_fails(
        self, agot_method: AGoT, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await agot_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates initial thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.AGOT
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to initialize."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "initialize"
        assert thought.metadata["phase"] == "initialize"

    async def test_execute_creates_graph_nodes(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute creates initial graph nodes."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._graph_nodes) > 0
        assert initialized_method._nodes_added > 0

    async def test_execute_adds_to_session(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.AGOT

    async def test_execute_content_format(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert "Graph" in thought.content or "graph" in thought.content

    async def test_execute_metadata(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "initialize"
        assert thought.metadata["reasoning_type"] == "adaptive_graph_of_thoughts"
        assert "graph_stats" in thought.metadata
        assert "nodes" in thought.metadata["graph_stats"]

    async def test_execute_with_custom_context(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test execution with custom context parameters."""
        context: dict[str, Any] = {
            "max_nodes": 10,
            "confidence_threshold": 0.6,
            "adaptation_cycles": 2,
        }

        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )

        assert thought.metadata["max_nodes"] == 10
        assert thought.metadata["confidence_threshold"] == 0.6
        assert thought.metadata["adaptation_cycles"] == 2


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, agot_method: AGoT, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.AGOT,
            content="Test",
            step_number=1,
            depth=0,
            metadata={"phase": "initialize"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await agot_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_initialize_to_adapt(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from initialize to adapt."""
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )

        assert initialized_method._current_phase == "adapt"
        assert adapt_thought.metadata["phase"] == "adapt"
        assert adapt_thought.type == ThoughtType.BRANCH

    async def test_phase_transition_adapt_to_propagate(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from adapt to propagate."""
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )

        propagate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt_thought,
        )

        assert initialized_method._current_phase == "propagate"
        assert propagate_thought.metadata["phase"] == "propagate"
        assert propagate_thought.type == ThoughtType.CONTINUATION

    async def test_adaptation_cycles(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test multiple adaptation cycles."""
        # Start with default 3 adaptation cycles
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        # First adaptation cycle
        adapt1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )
        assert adapt1.metadata["phase"] == "adapt"

        prop1 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt1,
        )
        assert prop1.metadata["phase"] == "propagate"

        # Second adaptation cycle (should go back to adapt)
        adapt2 = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=prop1,
        )
        assert adapt2.metadata["phase"] == "adapt"
        assert initialized_method._adaptation_cycle == 1

    async def test_phase_transition_to_synthesize(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition to synthesize after all adaptation cycles."""
        context: dict[str, Any] = {"adaptation_cycles": 1}

        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )

        propagate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt_thought,
        )

        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=propagate_thought,
        )

        assert initialized_method._current_phase == "synthesize"
        assert synthesize_thought.metadata["phase"] == "synthesize"
        assert synthesize_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_synthesize_to_conclude(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from synthesize to conclude."""
        context: dict[str, Any] = {"adaptation_cycles": 1}

        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )

        propagate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt_thought,
        )

        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=propagate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=synthesize_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert init_thought.step_number == 1

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )
        assert adapt_thought.step_number == 2

    async def test_depth_increases(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert init_thought.depth == 0

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )
        assert adapt_thought.depth == 1


class TestGraphAdaptation:
    """Tests for graph adaptation behavior."""

    async def test_nodes_added_tracked(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that nodes added are tracked."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._nodes_added > 0

    async def test_edges_added_tracked(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that edges added are tracked."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._edges_added > 0

    async def test_confidence_history_tracked(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence history is tracked."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._confidence_history) > 0

    async def test_graph_stats_in_metadata(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that graph statistics are included in thought metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        graph_stats = thought.metadata["graph_stats"]
        assert "nodes" in graph_stats
        assert "nodes_added" in graph_stats
        assert "nodes_removed" in graph_stats
        assert "edges_added" in graph_stats
        assert "edges_removed" in graph_stats


class TestEdgeCases:
    """Tests for edge cases in AGoT reasoning."""

    async def test_empty_query(self, initialized_method: AGoT, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(self, initialized_method: AGoT, session: Session) -> None:
        """Test handling of very long query."""
        long_query = "Analyze this complex problem: " + "test " * 500
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(self, initialized_method: AGoT, session: Session) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(self, initialized_method: AGoT, session: Session) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="分析这个问题：什么是人工智能？",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        context: dict[str, Any] = {"adaptation_cycles": 1}

        # Initialize
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )
        assert init_thought.type == ThoughtType.INITIAL

        # Adapt
        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )
        assert adapt_thought.type == ThoughtType.BRANCH

        # Propagate
        propagate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt_thought,
        )
        assert propagate_thought.type == ThoughtType.CONTINUATION

        # Synthesize
        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=propagate_thought,
        )
        assert synthesize_thought.type == ThoughtType.SYNTHESIS

        # Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=synthesize_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION

        # Verify session state
        assert session.thought_count == 5
        assert session.current_method == MethodIdentifier.AGOT

    async def test_guidance_parameter(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that guidance parameter is accepted."""
        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
            guidance="Focus on causal relationships",
        )

        assert adapt_thought is not None
        assert adapt_thought.metadata.get("guidance") == "Focus on causal relationships"


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        context: dict[str, Any] = {"adaptation_cycles": 1}

        init_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )
        init_confidence = init_thought.confidence

        adapt_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=init_thought,
        )
        adapt_confidence = adapt_thought.confidence

        propagate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=adapt_thought,
        )
        propagate_confidence = propagate_thought.confidence

        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=propagate_thought,
        )
        synthesize_confidence = synthesize_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=synthesize_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert init_confidence <= adapt_confidence
        assert adapt_confidence <= propagate_confidence
        assert propagate_confidence <= synthesize_confidence
        assert synthesize_confidence <= conclude_confidence


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.AGOT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: AGoT, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        agot_thoughts = session.get_thoughts_by_method(MethodIdentifier.AGOT)
        assert len(agot_thoughts) > 0
