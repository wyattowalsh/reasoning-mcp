"""Unit tests for ThinkOnGraph reasoning method.

This module provides comprehensive tests for the ThinkOnGraph method implementation,
covering initialization, execution, beam search exploration, entity extraction,
path scoring, continuation, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from reasoning_mcp.methods.native.think_on_graph import (
    THINK_ON_GRAPH_METADATA,
    ThinkOnGraph,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

# Fixtures


@pytest.fixture
def tog_method() -> ThinkOnGraph:
    """Create a ThinkOnGraph instance for testing.

    Returns:
        A fresh ThinkOnGraph instance
    """
    return ThinkOnGraph()


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
    return "What is the relationship between Einstein and the theory of relativity?"


@pytest.fixture
def mock_execution_context() -> AsyncMock:
    """Create a mock execution context for testing sampling."""
    ctx = AsyncMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Entity1\nEntity2\nEntity3")
    return ctx


# Test Metadata


class TestMetadata:
    """Tests for ThinkOnGraph metadata."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert THINK_ON_GRAPH_METADATA.identifier == MethodIdentifier.THINK_ON_GRAPH

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert THINK_ON_GRAPH_METADATA.name == "Think-on-Graph"

    def test_metadata_description(self):
        """Test metadata has descriptive text."""
        assert len(THINK_ON_GRAPH_METADATA.description) > 0
        assert "beam search" in THINK_ON_GRAPH_METADATA.description.lower()
        assert "knowledge graph" in THINK_ON_GRAPH_METADATA.description.lower()

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert THINK_ON_GRAPH_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test metadata contains expected tags."""
        expected_tags = {"knowledge-graph", "beam-search", "iterative", "exploration"}
        assert expected_tags.issubset(THINK_ON_GRAPH_METADATA.tags)

    def test_metadata_complexity(self):
        """Test metadata has reasonable complexity rating."""
        assert 1 <= THINK_ON_GRAPH_METADATA.complexity <= 10
        assert THINK_ON_GRAPH_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert THINK_ON_GRAPH_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert THINK_ON_GRAPH_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context requirement."""
        assert THINK_ON_GRAPH_METADATA.requires_context is True

    def test_metadata_min_thoughts(self):
        """Test metadata has minimum thoughts requirement."""
        assert THINK_ON_GRAPH_METADATA.min_thoughts >= 5

    def test_metadata_best_for(self):
        """Test metadata best_for contains relevant use cases."""
        best_for_str = " ".join(THINK_ON_GRAPH_METADATA.best_for)
        assert "knowledge graph" in best_for_str.lower()


# Test Initialization


class TestInitialization:
    """Tests for ThinkOnGraph initialization and setup."""

    def test_create_method(self, tog_method: ThinkOnGraph):
        """Test that ThinkOnGraph can be instantiated."""
        assert tog_method is not None
        assert isinstance(tog_method, ThinkOnGraph)

    def test_initial_state(self, tog_method: ThinkOnGraph):
        """Test that a new method starts in the correct initial state."""
        assert tog_method._initialized is False
        assert tog_method._step_counter == 0
        assert tog_method._current_phase == "initialize"
        assert tog_method._current_hop == 0
        assert len(tog_method._start_entities) == 0
        assert len(tog_method._beam_paths) == 0
        assert tog_method._best_path is None

    def test_custom_beam_width(self):
        """Test custom beam width configuration."""
        method = ThinkOnGraph(beam_width=5)
        assert method._beam_width == 5

    def test_custom_max_hops(self):
        """Test custom max hops configuration."""
        method = ThinkOnGraph(max_hops=5)
        assert method._max_hops == 5

    async def test_initialize(self, tog_method: ThinkOnGraph):
        """Test that initialize() sets up the method correctly."""
        await tog_method.initialize()
        assert tog_method._initialized is True
        assert tog_method._step_counter == 0
        assert tog_method._current_phase == "initialize"
        assert tog_method._current_hop == 0

    async def test_initialize_resets_state(self):
        """Test that initialize() resets state even if called multiple times."""
        method = ThinkOnGraph()
        await method.initialize()

        # Simulate some usage
        method._step_counter = 5
        method._current_phase = "evaluate"
        method._current_hop = 3
        method._start_entities = ["Entity_A"]
        method._beam_paths = [{"path": ["test"], "score": 0.5}]

        # Re-initialize
        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "initialize"
        assert method._current_hop == 0
        assert len(method._start_entities) == 0
        assert len(method._beam_paths) == 0
        assert method._initialized is True

    async def test_health_check_not_initialized(self, tog_method: ThinkOnGraph):
        """Test that health_check returns False before initialization."""
        result = await tog_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, tog_method: ThinkOnGraph):
        """Test that health_check returns True after initialization."""
        await tog_method.initialize()
        result = await tog_method.health_check()
        assert result is True


# Test Properties


class TestProperties:
    """Tests for ThinkOnGraph property accessors."""

    def test_identifier_property(self, tog_method: ThinkOnGraph):
        """Test that identifier returns the correct method identifier."""
        assert tog_method.identifier == MethodIdentifier.THINK_ON_GRAPH

    def test_name_property(self, tog_method: ThinkOnGraph):
        """Test that name returns the correct human-readable name."""
        assert tog_method.name == "Think-on-Graph"

    def test_description_property(self, tog_method: ThinkOnGraph):
        """Test that description returns the correct method description."""
        assert "beam search" in tog_method.description.lower()
        assert "knowledge graph" in tog_method.description.lower()

    def test_category_property(self, tog_method: ThinkOnGraph):
        """Test that category returns the correct method category."""
        assert tog_method.category == MethodCategory.ADVANCED


# Test Basic Execution


class TestBasicExecution:
    """Tests for basic execute() functionality."""

    async def test_execute_without_initialization_raises_error(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test that execute raises error without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await tog_method.execute(active_session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() creates initial thought."""
        await tog_method.initialize()
        thought = await tog_method.execute(active_session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.THINK_ON_GRAPH

    async def test_execute_sets_initialize_metadata(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets initialize phase metadata."""
        await tog_method.initialize()
        thought = await tog_method.execute(active_session, sample_problem)

        assert thought.metadata["phase"] == "initialize"
        assert "entities" in thought.metadata

    async def test_execute_content_structure(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() generates proper initial content."""
        await tog_method.initialize()
        thought = await tog_method.execute(active_session, sample_problem)

        content = thought.content
        assert "Initialize ToG Exploration" in content
        assert sample_problem in content
        assert "Beam width" in content
        assert "Max hops" in content

    async def test_execute_sets_confidence(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets appropriate confidence."""
        await tog_method.initialize()
        thought = await tog_method.execute(active_session, sample_problem)

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.6

    async def test_execute_adds_to_session(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() adds thought to session."""
        await tog_method.initialize()
        initial_count = active_session.thought_count

        await tog_method.execute(active_session, sample_problem)

        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_current_method(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test execute() sets session's current method."""
        await tog_method.initialize()
        await tog_method.execute(active_session, sample_problem)

        assert active_session.current_method == MethodIdentifier.THINK_ON_GRAPH


# Test Entity Extraction


class TestEntityExtraction:
    """Tests for entity extraction functionality."""

    def test_fallback_entity_extraction_with_capitalized(self, tog_method: ThinkOnGraph):
        """Test fallback entity extraction finds capitalized words."""
        entities = tog_method._fallback_entity_extraction("Einstein developed Relativity in Germany")
        assert "Einstein" in entities
        assert "Relativity" in entities
        assert "Germany" in entities

    def test_fallback_entity_extraction_defaults(self, tog_method: ThinkOnGraph):
        """Test fallback entity extraction returns defaults for no caps."""
        entities = tog_method._fallback_entity_extraction("all lowercase words here")
        assert entities == ["Entity_A", "Entity_B"]

    def test_fallback_entity_extraction_limits_results(self, tog_method: ThinkOnGraph):
        """Test fallback entity extraction limits to 3 entities."""
        entities = tog_method._fallback_entity_extraction(
            "Alice Bob Charlie David Eve Frank"
        )
        assert len(entities) <= 3


# Test Beam Path Expansion


class TestBeamPathExpansion:
    """Tests for beam path expansion."""

    def test_fallback_neighbors(self, tog_method: ThinkOnGraph):
        """Test fallback neighbor generation."""
        neighbors = tog_method._fallback_neighbors("TestEntity")
        assert len(neighbors) == 2
        assert all("TestEntity_neighbor_" in n for n in neighbors)


# Test Path Scoring


class TestPathScoring:
    """Tests for path scoring functionality."""

    def test_fallback_path_score(self, tog_method: ThinkOnGraph):
        """Test fallback path scoring."""
        path = {"path": ["A", "B"], "score": 0.9}
        score = tog_method._fallback_path_score(path)
        assert 0.0 <= score <= 1.0
        # Score should decrease with path length
        assert score < 0.9

    def test_fallback_path_score_longer_path(self, tog_method: ThinkOnGraph):
        """Test fallback scoring penalizes longer paths."""
        short_path = {"path": ["A", "B"], "score": 0.9}
        long_path = {"path": ["A", "B", "C", "D"], "score": 0.9}

        short_score = tog_method._fallback_path_score(short_path)
        long_score = tog_method._fallback_path_score(long_path)

        assert short_score > long_score


# Test Continue Reasoning


class TestContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_raises_error(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
    ):
        """Test continue_reasoning raises error without initialization."""
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THINK_ON_GRAPH,
            content="Test",
            metadata={"phase": "initialize"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await tog_method.continue_reasoning(active_session, thought)

    async def test_continue_to_explore_phase(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test continuation from initialize to explore phase."""
        await tog_method.initialize()
        initial = await tog_method.execute(active_session, sample_problem)

        explore = await tog_method.continue_reasoning(active_session, initial)

        assert explore.metadata["phase"] == "explore"
        assert explore.metadata["hop"] == 1
        assert explore.type == ThoughtType.REASONING

    async def test_explore_multiple_hops(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test multiple exploration hops."""
        await tog_method.initialize()
        current = await tog_method.execute(active_session, sample_problem)

        # First hop
        current = await tog_method.continue_reasoning(active_session, current)
        assert current.metadata["hop"] == 1

        # Second hop
        current = await tog_method.continue_reasoning(active_session, current)
        assert current.metadata["hop"] == 2

        # Third hop
        current = await tog_method.continue_reasoning(active_session, current)
        assert current.metadata["hop"] == 3

    async def test_evaluate_phase_after_max_hops(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test transition to evaluate phase after max hops."""
        await tog_method.initialize()
        current = await tog_method.execute(active_session, sample_problem)

        # Complete all hops
        for _ in range(tog_method._max_hops):
            current = await tog_method.continue_reasoning(active_session, current)

        # Should be in evaluate phase
        evaluate = await tog_method.continue_reasoning(active_session, current)
        assert evaluate.metadata["phase"] == "evaluate"
        assert evaluate.type == ThoughtType.VERIFICATION

    async def test_select_phase_after_evaluate(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test transition to select phase after evaluate."""
        await tog_method.initialize()
        current = await tog_method.execute(active_session, sample_problem)

        # Complete all hops
        for _ in range(tog_method._max_hops):
            current = await tog_method.continue_reasoning(active_session, current)

        # Evaluate phase
        evaluate = await tog_method.continue_reasoning(active_session, current)

        # Select phase
        select = await tog_method.continue_reasoning(active_session, evaluate)
        assert select.metadata["phase"] == "select"
        assert select.type == ThoughtType.SYNTHESIS

    async def test_conclusion_phase(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test final conclusion phase."""
        await tog_method.initialize()
        current = await tog_method.execute(active_session, sample_problem)

        # Complete all hops
        for _ in range(tog_method._max_hops):
            current = await tog_method.continue_reasoning(active_session, current)

        # Evaluate phase
        current = await tog_method.continue_reasoning(active_session, current)

        # Select phase
        current = await tog_method.continue_reasoning(active_session, current)

        # Conclusion phase
        conclusion = await tog_method.continue_reasoning(active_session, current)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION
        assert conclusion.confidence == 0.87


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
    ):
        """Test execution with empty problem string."""
        await tog_method.initialize()

        thought = await tog_method.execute(active_session, "")

        assert thought is not None
        assert thought.content != ""

    async def test_very_short_problem(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
    ):
        """Test execution with very short problem."""
        await tog_method.initialize()

        thought = await tog_method.execute(active_session, "Test")

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
    ):
        """Test execution with special characters."""
        await tog_method.initialize()

        problem = "What is @#$%^&*() relation to Einstein?"
        thought = await tog_method.execute(active_session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_unicode_in_problem(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
    ):
        """Test execution with Unicode characters."""
        await tog_method.initialize()

        problem = "What is the relationship between Einstein and relativity?"
        thought = await tog_method.execute(active_session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_parent_child_relationships(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test parent-child relationships between phases."""
        await tog_method.initialize()

        initial = await tog_method.execute(active_session, sample_problem)
        explore = await tog_method.continue_reasoning(active_session, initial)

        assert explore.parent_id == initial.id

    async def test_depth_increments_per_phase(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test depth increments with each phase."""
        await tog_method.initialize()

        initial = await tog_method.execute(active_session, sample_problem)
        explore = await tog_method.continue_reasoning(active_session, initial)

        assert initial.depth == 0
        assert explore.depth == 1

    async def test_step_counter_increments(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test step counter increments with each continuation."""
        await tog_method.initialize()

        initial = await tog_method.execute(active_session, sample_problem)
        explore1 = await tog_method.continue_reasoning(active_session, initial)
        explore2 = await tog_method.continue_reasoning(active_session, explore1)

        assert initial.step_number == 1
        assert explore1.step_number == 2
        assert explore2.step_number == 3

    async def test_thought_ids_are_unique(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test that all generated thoughts have unique IDs."""
        await tog_method.initialize()

        thoughts = []
        current = await tog_method.execute(active_session, sample_problem)
        thoughts.append(current)

        for _ in range(3):
            current = await tog_method.continue_reasoning(active_session, current)
            thoughts.append(current)

        ids = [t.id for t in thoughts]
        assert len(ids) == len(set(ids))


# Test Session Integration


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session thought count updates correctly."""
        await tog_method.initialize()

        initial_count = active_session.thought_count
        await tog_method.execute(active_session, sample_problem)

        assert active_session.thought_count == initial_count + 1

    async def test_session_metrics_update(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session metrics update after execution."""
        await tog_method.initialize()

        await tog_method.execute(active_session, sample_problem)

        assert active_session.metrics.total_thoughts > 0
        assert active_session.metrics.average_confidence > 0.0

    async def test_session_method_tracking(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test session tracks method usage."""
        await tog_method.initialize()

        await tog_method.execute(active_session, sample_problem)

        method_key = str(MethodIdentifier.THINK_ON_GRAPH)
        assert method_key in active_session.metrics.methods_used
        assert active_session.metrics.methods_used[method_key] > 0


# Test Sampling with Fallback


class TestSamplingWithFallback:
    """Tests for _sample_with_fallback integration."""

    async def test_identify_entities_without_context(
        self,
        tog_method: ThinkOnGraph,
        active_session: Session,
        sample_problem: str,
    ):
        """Test entity identification falls back without execution context."""
        await tog_method.initialize()

        # Execute without execution context - should use fallback
        thought = await tog_method.execute(active_session, sample_problem)

        assert thought is not None
        assert len(tog_method._start_entities) > 0

    async def test_expand_path_without_context(
        self,
        tog_method: ThinkOnGraph,
        sample_problem: str,
    ):
        """Test path expansion falls back without execution context."""
        await tog_method.initialize()

        path = {"path": ["TestEntity"], "score": 1.0}
        expanded = await tog_method._expand_beam_path(path, sample_problem)

        assert len(expanded) == 2
        assert all("path" in p and "score" in p for p in expanded)

    async def test_score_path_without_context(
        self,
        tog_method: ThinkOnGraph,
        sample_problem: str,
    ):
        """Test path scoring falls back without execution context."""
        await tog_method.initialize()

        path = {"path": ["A", "B", "C"], "score": 0.9}
        score = await tog_method._score_path(path, sample_problem)

        assert 0.0 <= score <= 1.0
