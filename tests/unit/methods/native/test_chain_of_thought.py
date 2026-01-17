"""Unit tests for ChainOfThought reasoning method.

This module provides comprehensive tests for the ChainOfThought method implementation,
covering initialization, execution, chain structure, configuration, continuation,
evidence tracking, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.chain_of_thought import (
    CHAIN_OF_THOUGHT_METADATA,
    ChainOfThought,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def cot_method() -> ChainOfThought:
    """Create a ChainOfThought method instance for testing.

    Returns:
        A fresh ChainOfThought instance
    """
    return ChainOfThought()


@pytest.fixture
def initialized_method() -> ChainOfThought:
    """Create an initialized ChainOfThought method instance.

    Returns:
        An initialized ChainOfThought instance
    """
    method = ChainOfThought()
    # Since initialize is async, we can't call it here
    # Tests will need to await method.initialize() themselves
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
    return "If x + 5 = 12, what is the value of x?"


@pytest.fixture
def long_problem() -> str:
    """Provide a longer problem for testing.

    Returns:
        A longer, more complex problem string
    """
    return (
        "Consider a scenario where you have three containers: A contains 100ml of water, "
        "B contains 50ml of oil, and C is empty. You transfer 25ml from A to C, then 15ml "
        "from B to C, and finally 10ml from C back to A. What is the final volume in each "
        "container, and what is the approximate ratio of water to oil in container C?"
    )


class TestChainOfThoughtInitialization:
    """Tests for ChainOfThought initialization and setup."""

    def test_create_method(self, cot_method: ChainOfThought):
        """Test that ChainOfThought can be instantiated."""
        assert cot_method is not None
        assert isinstance(cot_method, ChainOfThought)

    def test_initial_state(self, cot_method: ChainOfThought):
        """Test that a new method starts in the correct initial state."""
        assert cot_method._step_count == 0
        assert cot_method._is_initialized is False

    async def test_initialize(self, cot_method: ChainOfThought):
        """Test that initialize() sets up the method correctly."""
        await cot_method.initialize()
        assert cot_method._is_initialized is True
        assert cot_method._step_count == 0

    async def test_initialize_resets_state(self):
        """Test that initialize() resets step count even if called multiple times."""
        method = ChainOfThought()
        await method.initialize()
        method._step_count = 5  # Simulate some usage

        # Re-initialize
        await method.initialize()
        assert method._step_count == 0
        assert method._is_initialized is True

    async def test_health_check_not_initialized(self, cot_method: ChainOfThought):
        """Test that health_check returns False before initialization."""
        result = await cot_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, cot_method: ChainOfThought):
        """Test that health_check returns True after initialization."""
        await cot_method.initialize()
        result = await cot_method.health_check()
        assert result is True


class TestChainOfThoughtProperties:
    """Tests for ChainOfThought property accessors."""

    def test_identifier_property(self, cot_method: ChainOfThought):
        """Test that identifier returns the correct method identifier."""
        assert cot_method.identifier == str(MethodIdentifier.CHAIN_OF_THOUGHT)

    def test_name_property(self, cot_method: ChainOfThought):
        """Test that name returns the correct human-readable name."""
        assert cot_method.name == "Chain of Thought"

    def test_description_property(self, cot_method: ChainOfThought):
        """Test that description returns the correct method description."""
        assert "step-by-step" in cot_method.description.lower()
        assert "reasoning" in cot_method.description.lower()

    def test_category_property(self, cot_method: ChainOfThought):
        """Test that category returns the correct method category."""
        assert cot_method.category == str(MethodCategory.CORE)


class TestChainOfThoughtMetadata:
    """Tests for ChainOfThought metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert CHAIN_OF_THOUGHT_METADATA.identifier == MethodIdentifier.CHAIN_OF_THOUGHT

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert CHAIN_OF_THOUGHT_METADATA.category == MethodCategory.CORE

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"sequential", "explicit", "logical", "step-by-step"}
        assert expected_tags.issubset(CHAIN_OF_THOUGHT_METADATA.tags)

    def test_metadata_supports_branching(self):
        """Test that metadata correctly indicates branching support."""
        assert CHAIN_OF_THOUGHT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self):
        """Test that metadata correctly indicates revision support."""
        assert CHAIN_OF_THOUGHT_METADATA.supports_revision is True

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= CHAIN_OF_THOUGHT_METADATA.complexity <= 10

    def test_metadata_thought_bounds(self):
        """Test that metadata defines min/max thought counts."""
        assert CHAIN_OF_THOUGHT_METADATA.min_thoughts >= 1
        assert CHAIN_OF_THOUGHT_METADATA.max_thoughts >= CHAIN_OF_THOUGHT_METADATA.min_thoughts


class TestChainOfThoughtExecution:
    """Tests for ChainOfThought execute() method."""

    async def test_execute_basic(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test basic execution creates a thought."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.CHAIN_OF_THOUGHT

    async def test_execute_auto_initializes(
        self,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute auto-initializes if not initialized."""
        method = ChainOfThought()
        assert method._is_initialized is False

        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert method._is_initialized is True

    async def test_execute_creates_initial_thought(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates an INITIAL thought type for first execution."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_creates_continuation_thought(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates CONTINUATION thought when session has thoughts."""
        await cot_method.initialize()

        # First thought
        first_thought = await cot_method.execute(session, sample_problem)
        assert first_thought.type == ThoughtType.INITIAL

        # Second thought should be continuation
        second_thought = await cot_method.execute(session, "Continue the analysis")
        assert second_thought.type == ThoughtType.CONTINUATION
        assert second_thought.parent_id == first_thought.id
        assert second_thought.depth == first_thought.depth + 1

    async def test_execute_adds_to_session(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute adds thought to the session."""
        await cot_method.initialize()
        initial_count = session.thought_count

        await cot_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_execute_with_context(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test execute with custom context parameters."""
        await cot_method.initialize()
        context: dict[str, Any] = {"max_steps": 10, "target_steps": 7}

        thought = await cot_method.execute(session, sample_problem, context=context)

        assert thought is not None
        assert "input_text" in thought.metadata
        assert thought.metadata["input_text"] == sample_problem

    async def test_execute_sets_confidence(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets a confidence score."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence > 0.0  # Should have some confidence

    async def test_execute_sets_step_number(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets correct step numbers."""
        await cot_method.initialize()

        first = await cot_method.execute(session, sample_problem)
        assert first.step_number == 1

        second = await cot_method.execute(session, "Continue")
        assert second.step_number == 2


class TestChainStructure:
    """Tests for the structure and content of reasoning chains."""

    async def test_chain_has_opening_statement(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that reasoning chain starts with opening statement."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        content_lower = thought.content.lower()
        assert "let me think" in content_lower or "let's" in content_lower

    async def test_chain_has_first_step(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that reasoning chain includes a first step indicator."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        content_lower = thought.content.lower()
        assert "first" in content_lower

    async def test_chain_has_intermediate_steps(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that reasoning chain includes intermediate step markers."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        content_lower = thought.content.lower()
        # Should have at least one intermediate connector
        intermediate_markers = ["next", "this means", "given"]
        assert any(marker in content_lower for marker in intermediate_markers)

    async def test_chain_has_conclusion(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that reasoning chain includes conclusion marker."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        content_lower = thought.content.lower()
        conclusion_markers = ["therefore", "in conclusion", "thus"]
        assert any(marker in content_lower for marker in conclusion_markers)

    async def test_chain_includes_problem_reference(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that reasoning chain references the original problem."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert sample_problem in thought.content

    async def test_chain_step_count_tracked(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that method tracks step count in metadata."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert "reasoning_steps" in thought.metadata
        assert thought.metadata["reasoning_steps"] > 0
        assert thought.metadata["reasoning_steps"] == cot_method._step_count

    async def test_chain_metadata_fields(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that thought metadata contains required fields."""
        await cot_method.initialize()
        thought = await cot_method.execute(session, sample_problem)

        assert "method" in thought.metadata
        assert thought.metadata["method"] == "chain_of_thought"
        assert "explicit_steps" in thought.metadata
        assert thought.metadata["explicit_steps"] is True


class TestChainOfThoughtContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_reasoning_basic(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test basic continuation of reasoning."""
        await cot_method.initialize()

        # Create initial thought
        initial = await cot_method.execute(session, sample_problem)

        # Continue reasoning
        continuation = await cot_method.continue_reasoning(
            session,
            initial,
            guidance="Explore edge cases",
        )

        assert continuation is not None
        assert isinstance(continuation, ThoughtNode)

    async def test_continue_auto_initializes(
        self,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning auto-initializes if needed."""
        method = ChainOfThought()
        await method.initialize()

        initial = await method.execute(session, sample_problem)

        # Create new uninitialized method
        method2 = ChainOfThought()
        assert method2._is_initialized is False

        continuation = await method2.continue_reasoning(session, initial)
        assert continuation is not None
        assert method2._is_initialized is True

    async def test_continue_creates_continuation_type(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning creates CONTINUATION thought type."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        continuation = await cot_method.continue_reasoning(session, initial)

        assert continuation.type == ThoughtType.CONTINUATION

    async def test_continue_sets_parent(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation has correct parent_id."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        continuation = await cot_method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation increments depth."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        continuation = await cot_method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1

    async def test_continue_with_guidance(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation with custom guidance."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        guidance = "Now consider the mathematical properties"
        continuation = await cot_method.continue_reasoning(
            session,
            initial,
            guidance=guidance,
        )

        assert "guidance" in continuation.metadata
        assert continuation.metadata["guidance"] == guidance
        assert guidance in continuation.content

    async def test_continue_without_guidance(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation without explicit guidance."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        continuation = await cot_method.continue_reasoning(session, initial)

        assert continuation.content != ""
        # Should use default guidance
        assert "continued_from" in continuation.metadata

    async def test_continue_adds_to_session(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation is added to session."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)
        count_before = session.thought_count

        await cot_method.continue_reasoning(session, initial)

        assert session.thought_count == count_before + 1

    async def test_continue_has_building_phrase(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation includes building/extending language."""
        await cot_method.initialize()
        initial = await cot_method.execute(session, sample_problem)

        continuation = await cot_method.continue_reasoning(session, initial)

        content_lower = continuation.content.lower()
        building_markers = ["building", "given", "extending", "continue"]
        assert any(marker in content_lower for marker in building_markers)


class TestConfiguration:
    """Tests for configuration options."""

    async def test_max_steps_configuration(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that max_steps configuration is respected."""
        await cot_method.initialize()

        context: dict[str, Any] = {"max_steps": 3}
        thought = await cot_method.execute(session, sample_problem, context=context)

        # Step count should not exceed max_steps
        assert thought.metadata.get("reasoning_steps", 0) <= 3

    async def test_target_steps_configuration(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that target_steps configuration affects generation."""
        await cot_method.initialize()

        context: dict[str, Any] = {"target_steps": 7, "max_steps": 10}
        thought = await cot_method.execute(session, sample_problem, context=context)

        # Should generate steps (exact number depends on implementation)
        assert thought.metadata.get("reasoning_steps", 0) >= 3

    async def test_empty_context(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test execution with empty context dictionary."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, sample_problem, context={})

        assert thought is not None
        # Should use default values

    async def test_none_context(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test execution with None context (default)."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, sample_problem, context=None)

        assert thought is not None
        # Should use default values


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test execution with empty problem string."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, "")

        assert thought is not None
        # Should still generate some reasoning structure

    async def test_very_short_problem(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test execution with very short problem."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, "2+2?")

        assert thought is not None
        assert thought.content != ""

    async def test_very_long_problem(
        self,
        cot_method: ChainOfThought,
        session: Session,
        long_problem: str,
    ):
        """Test execution with long, complex problem."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, long_problem)

        assert thought is not None
        assert long_problem in thought.content
        # Should still generate complete chain

    async def test_special_characters_in_problem(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test execution with special characters."""
        await cot_method.initialize()

        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"
        thought = await cot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_multiple_executions_same_session(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test multiple executions in the same session."""
        await cot_method.initialize()

        thoughts = []
        for i in range(5):
            thought = await cot_method.execute(session, f"Problem {i}")
            thoughts.append(thought)

        assert len(thoughts) == 5
        assert session.thought_count == 5
        # First should be INITIAL, rest should be CONTINUATION
        assert thoughts[0].type == ThoughtType.INITIAL
        for thought in thoughts[1:]:
            assert thought.type == ThoughtType.CONTINUATION

    async def test_nested_reasoning_depth(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test deeply nested reasoning chain."""
        await cot_method.initialize()

        current = await cot_method.execute(session, sample_problem)

        # Create a chain of continuations
        for i in range(5):
            current = await cot_method.continue_reasoning(
                session,
                current,
                guidance=f"Explore aspect {i + 1}",
            )

        # Last thought should have depth of 5 (0-indexed parent + 5 continuations)
        assert current.depth >= 5

    async def test_thought_ids_are_unique(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test that generated thoughts have unique IDs."""
        await cot_method.initialize()

        thoughts = []
        for i in range(10):
            thought = await cot_method.execute(session, f"Problem {i}")
            thoughts.append(thought)

        # All IDs should be unique
        ids = [t.id for t in thoughts]
        assert len(ids) == len(set(ids))

    async def test_unicode_in_problem(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test execution with Unicode characters."""
        await cot_method.initialize()

        problem = "解决这个问题: 如何优化算法? (Solve: How to optimize algorithm?)"
        thought = await cot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_newlines_in_problem(
        self,
        cot_method: ChainOfThought,
        session: Session,
    ):
        """Test execution with newlines in problem."""
        await cot_method.initialize()

        problem = "Problem:\nGiven: x = 5\nFind: x + 3\nShow work."
        thought = await cot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""


class TestEvidenceTracking:
    """Tests for evidence tracking in thought metadata."""

    async def test_metadata_contains_input(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata tracks the input text."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, sample_problem)

        assert "input_text" in thought.metadata
        assert thought.metadata["input_text"] == sample_problem

    async def test_metadata_contains_step_count(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata tracks reasoning steps."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, sample_problem)

        assert "reasoning_steps" in thought.metadata
        assert isinstance(thought.metadata["reasoning_steps"], int)
        assert thought.metadata["reasoning_steps"] > 0

    async def test_continuation_metadata(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation metadata references parent."""
        await cot_method.initialize()

        initial = await cot_method.execute(session, sample_problem)
        continuation = await cot_method.continue_reasoning(session, initial)

        assert "continued_from" in continuation.metadata
        assert continuation.metadata["continued_from"] == initial.id

    async def test_thought_evidence_field_available(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that thoughts can have evidence field populated."""
        await cot_method.initialize()

        thought = await cot_method.execute(session, sample_problem)

        # ThoughtNode has evidence field (empty list by default)
        assert hasattr(thought, "evidence")
        assert isinstance(thought.evidence, list)


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that session thought count updates correctly."""
        await cot_method.initialize()

        initial_count = session.thought_count
        await cot_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_metrics_update(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that session metrics update after execution."""
        await cot_method.initialize()

        await cot_method.execute(session, sample_problem)

        assert session.metrics.total_thoughts > 0
        assert session.metrics.average_confidence > 0.0

    async def test_session_method_tracking(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that session tracks method usage."""
        await cot_method.initialize()

        await cot_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that session can filter thoughts by method."""
        await cot_method.initialize()

        await cot_method.execute(session, sample_problem)

        cot_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert len(cot_thoughts) > 0

    async def test_session_graph_structure(
        self,
        cot_method: ChainOfThought,
        session: Session,
        sample_problem: str,
    ):
        """Test that thoughts are properly linked in session graph."""
        await cot_method.initialize()

        initial = await cot_method.execute(session, sample_problem)
        continuation = await cot_method.continue_reasoning(session, initial)

        # Check graph structure
        assert session.graph.node_count >= 2
        assert continuation.id in session.graph.nodes
        assert initial.id in session.graph.nodes

        # Check parent-child relationship
        parent_node = session.graph.get_node(initial.id)
        assert parent_node is not None
        assert continuation.id in parent_node.children_ids
