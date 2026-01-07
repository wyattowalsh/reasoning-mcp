"""
Comprehensive tests for LeastToMost reasoning method.

This module provides complete test coverage for:
- LeastToMostMethod: Progressive problem decomposition method

Tests cover:
1. Initialization: Test initialize() and health_check() methods
2. Basic execution: Test execute() decomposes problem progressively
3. Decomposition: Verify problem broken into ordered subproblems
4. Configuration: Test decomposition_strategy config options
5. Continue reasoning: Test continue_reasoning() solves next subproblem
6. Ordering: Test that simpler subproblems solved first
7. Dependency tracking: Test subproblem dependencies respected
8. Solution building: Test progressive solution building from subproblems
9. Final integration: Test combination of subproblem solutions
10. Edge cases: Already simple problem, deeply nested decomposition
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.least_to_most import (
    LEAST_TO_MOST_METADATA,
    LeastToMost,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> LeastToMost:
    """Provide a LeastToMost method instance for testing.

    Returns:
        LeastToMost instance (not initialized).
    """
    return LeastToMost()


@pytest.fixture
async def initialized_method() -> LeastToMost:
    """Provide an initialized LeastToMost method for testing.

    Returns:
        Initialized LeastToMost instance.
    """
    method = LeastToMost()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Provide an active Session for testing.

    Returns:
        Session that has been started (status=ACTIVE).
    """
    session = Session()
    session.start()
    return session


@pytest.fixture
def sample_input() -> str:
    """Provide a sample input problem for testing.

    Returns:
        A sample problem string.
    """
    return "Prove the Pythagorean theorem step by step"


# ============================================================================
# Test Metadata
# ============================================================================


class TestLeastToMostMetadata:
    """Test suite for LeastToMost metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert LEAST_TO_MOST_METADATA.identifier == MethodIdentifier.LEAST_TO_MOST

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert LEAST_TO_MOST_METADATA.name == "Least to Most"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert LEAST_TO_MOST_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has correct complexity level."""
        assert LEAST_TO_MOST_METADATA.complexity == 4

    def test_metadata_tags(self):
        """Test that metadata has correct tags."""
        expected_tags = {
            "decomposition",
            "progressive",
            "sequential",
            "building",
            "subproblems",
            "ordered",
        }
        assert LEAST_TO_MOST_METADATA.tags == frozenset(expected_tags)

    def test_metadata_branching_support(self):
        """Test that metadata indicates no branching support."""
        assert LEAST_TO_MOST_METADATA.supports_branching is False

    def test_metadata_revision_support(self):
        """Test that metadata indicates revision support."""
        assert LEAST_TO_MOST_METADATA.supports_revision is True

    def test_metadata_context_requirement(self):
        """Test that metadata indicates no special context required."""
        assert LEAST_TO_MOST_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test that metadata has minimum thought count."""
        assert LEAST_TO_MOST_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata has no max thought limit."""
        assert LEAST_TO_MOST_METADATA.max_thoughts == 0  # Unlimited

    def test_metadata_avg_tokens(self):
        """Test that metadata has average token estimate."""
        assert LEAST_TO_MOST_METADATA.avg_tokens_per_thought == 400


# ============================================================================
# Test Initialization
# ============================================================================


class TestLeastToMostInitialization:
    """Test suite for LeastToMost initialization."""

    def test_create_uninitialized(self, method: LeastToMost):
        """Test creating an uninitialized method instance."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._subproblems == []
        assert method._subproblem_solutions == []
        assert method._decomposition_complete is False

    @pytest.mark.asyncio
    async def test_initialize(self, method: LeastToMost):
        """Test initializing the method."""
        await method.initialize()

        assert method._initialized is True
        assert method._step_counter == 0
        assert method._subproblems == []
        assert method._subproblem_solutions == []
        assert method._decomposition_complete is False

    @pytest.mark.asyncio
    async def test_health_check_uninitialized(self, method: LeastToMost):
        """Test health_check returns False for uninitialized method."""
        is_healthy = await method.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: LeastToMost):
        """Test health_check returns True for initialized method."""
        is_healthy = await initialized_method.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_reinitialize_resets_state(self, initialized_method: LeastToMost):
        """Test that re-initialization resets internal state."""
        # Modify state
        initialized_method._step_counter = 5
        initialized_method._subproblems = ["test1", "test2"]
        initialized_method._subproblem_solutions = ["solution1"]
        initialized_method._decomposition_complete = True

        # Re-initialize
        await initialized_method.initialize()

        # Verify state reset
        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._subproblems == []
        assert initialized_method._subproblem_solutions == []
        assert initialized_method._decomposition_complete is False


# ============================================================================
# Test Properties
# ============================================================================


class TestLeastToMostProperties:
    """Test suite for LeastToMost properties."""

    def test_identifier_property(self, method: LeastToMost):
        """Test that identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.LEAST_TO_MOST

    def test_name_property(self, method: LeastToMost):
        """Test that name property returns correct value."""
        assert method.name == "Least to Most"

    def test_description_property(self, method: LeastToMost):
        """Test that description property returns correct value."""
        assert "Progressive problem decomposition" in method.description

    def test_category_property(self, method: LeastToMost):
        """Test that category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Test Basic Execution
# ============================================================================


class TestLeastToMostExecution:
    """Test suite for LeastToMost execution."""

    @pytest.mark.asyncio
    async def test_execute_uninitialized_raises_error(
        self, method: LeastToMost, active_session: Session, sample_input: str
    ):
        """Test that executing uninitialized method raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be initialized before execution"):
            await method.execute(session=active_session, input_text=sample_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute creates an initial thought."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.LEAST_TO_MOST
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_content_contains_decomposition(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute creates thought with decomposition content."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert "Decomposition" in thought.content
        assert "Subproblem" in thought.content
        assert sample_input in thought.content

    @pytest.mark.asyncio
    async def test_execute_updates_session(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute updates the session."""
        initial_count = active_session.thought_count

        await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert active_session.thought_count == initial_count + 1
        assert active_session.current_method == MethodIdentifier.LEAST_TO_MOST

    @pytest.mark.asyncio
    async def test_execute_with_context(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute accepts and stores context."""
        context = {"domain": "mathematics", "level": "advanced"}

        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input, context=context
        )

        assert thought.metadata["context"] == context
        assert thought.metadata["input"] == sample_input

    @pytest.mark.asyncio
    async def test_execute_resets_state(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that execute resets internal state for new execution."""
        # Execute once to set state
        await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Verify state was set
        assert initialized_method._step_counter == 1
        assert len(initialized_method._subproblems) > 0

        # Execute again with different input
        session2 = Session().start()
        await initialized_method.execute(
            session=session2, input_text="Different problem"
        )

        # Verify state was reset
        assert initialized_method._step_counter == 1
        assert len(initialized_method._subproblems) == 3  # Default number


# ============================================================================
# Test Decomposition
# ============================================================================


class TestLeastToMostDecomposition:
    """Test suite for problem decomposition."""

    @pytest.mark.asyncio
    async def test_decomposition_creates_subproblems(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that decomposition creates subproblems."""
        await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert len(initialized_method._subproblems) > 0
        assert initialized_method._decomposition_complete is False

    @pytest.mark.asyncio
    async def test_decomposition_metadata(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that decomposition thought has correct metadata."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert thought.metadata["stage"] == "decomposition"
        assert thought.metadata["reasoning_type"] == "least_to_most"
        assert thought.metadata["subproblems_identified"] > 0

    @pytest.mark.asyncio
    async def test_decomposition_confidence(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that decomposition has appropriate confidence."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert thought.confidence == 0.75  # Initial decomposition confidence


# ============================================================================
# Test Continue Reasoning
# ============================================================================


class TestLeastToMostContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_uninitialized_raises_error(
        self, method: LeastToMost, active_session: Session
    ):
        """Test that continue_reasoning on uninitialized method raises error."""
        # Create a dummy thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LEAST_TO_MOST,
            content="Test",
        )

        with pytest.raises(RuntimeError, match="must be initialized before continuation"):
            await method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

    @pytest.mark.asyncio
    async def test_continue_from_decomposition_creates_ordering(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that first continuation creates ordering step."""
        # Execute to get decomposition
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Continue to get ordering
        second_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought
        )

        assert second_thought.type == ThoughtType.CONTINUATION
        assert second_thought.step_number == 2
        assert second_thought.parent_id == first_thought.id
        assert "Ordering" in second_thought.content
        assert initialized_method._decomposition_complete is True

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning increments step counter."""
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert initialized_method._step_counter == 1

        second_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought
        )

        assert initialized_method._step_counter == 2

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning accepts guidance parameter."""
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        guidance = "Focus on geometric approach"
        second_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought, guidance=guidance
        )

        assert second_thought.metadata["guidance"] == guidance

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that continue_reasoning accepts context parameter."""
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        context = {"constraint": "use only Euclidean geometry"}
        second_thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            context=context,
        )

        assert second_thought.metadata["context"] == context


# ============================================================================
# Test Subproblem Solving
# ============================================================================


class TestLeastToMostSubproblemSolving:
    """Test suite for sequential subproblem solving."""

    @pytest.mark.asyncio
    async def test_solve_first_subproblem(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test solving the first subproblem."""
        # Decomposition
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Ordering
        second_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought
        )

        # First subproblem solution
        third_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=second_thought
        )

        assert "Subproblem 1" in third_thought.content
        assert third_thought.metadata["stage"] == "solving"
        assert len(initialized_method._subproblem_solutions) == 1

    @pytest.mark.asyncio
    async def test_solve_multiple_subproblems(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test solving multiple subproblems sequentially."""
        # Start reasoning
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Continue through ordering and first two subproblems
        for i in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Should have solved 2 subproblems (after ordering step)
        assert len(initialized_method._subproblem_solutions) == 2

    @pytest.mark.asyncio
    async def test_subproblem_progress_tracking(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that subproblem progress is tracked in metadata."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        num_subproblems = len(initialized_method._subproblems)

        # Ordering step
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        # First subproblem
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        assert thought.metadata["subproblems_total"] == num_subproblems
        assert thought.metadata["subproblems_solved"] == 1
        assert thought.metadata["progress"] == f"1/{num_subproblems}"

    @pytest.mark.asyncio
    async def test_solution_builds_on_previous(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that later solutions reference previous solutions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Ordering
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        # First subproblem
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        # Second subproblem - should reference first
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        assert "previous solutions" in thought.content.lower()


# ============================================================================
# Test Progressive Confidence
# ============================================================================


class TestLeastToMostConfidence:
    """Test suite for progressive confidence tracking."""

    @pytest.mark.asyncio
    async def test_confidence_increases_with_progress(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that confidence increases as subproblems are solved."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )
        initial_confidence = thought.confidence

        # Ordering
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        # First subproblem
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        first_confidence = thought.confidence

        # Second subproblem
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        second_confidence = thought.confidence

        # Confidence should increase
        assert first_confidence > 0.7
        assert second_confidence > first_confidence

    @pytest.mark.asyncio
    async def test_final_confidence_is_highest(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that final synthesis has highest confidence."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Continue through all steps to synthesis
        for _ in range(5):  # Ordering + 3 subproblems + synthesis
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Final synthesis should have high confidence
        assert thought.confidence >= 0.9


# ============================================================================
# Test Final Synthesis
# ============================================================================


class TestLeastToMostSynthesis:
    """Test suite for final solution synthesis."""

    @pytest.mark.asyncio
    async def test_synthesis_after_all_subproblems(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that synthesis occurs after all subproblems are solved."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Continue through ordering and all subproblems
        num_subproblems = len(initialized_method._subproblems)
        for _ in range(num_subproblems + 1):  # +1 for ordering
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Next step should be synthesis
        synthesis_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        assert synthesis_thought.metadata["stage"] == "synthesis"
        assert "Synthesis" in synthesis_thought.content

    @pytest.mark.asyncio
    async def test_synthesis_thought_type(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that synthesis uses appropriate thought type."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        # Continue to synthesis
        num_subproblems = len(initialized_method._subproblems)
        for _ in range(num_subproblems + 2):  # +1 ordering, +1 to get past last subproblem
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Synthesis should use SYNTHESIS or CONCLUSION type
        assert thought.type in (ThoughtType.SYNTHESIS, ThoughtType.CONCLUSION)

    @pytest.mark.asyncio
    async def test_synthesis_includes_all_solutions(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that synthesis references all subproblem solutions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        num_subproblems = len(initialized_method._subproblems)

        # Continue to synthesis
        for _ in range(num_subproblems + 2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Synthesis should reference all solutions
        for i in range(num_subproblems):
            assert f"Solution {i+1}" in thought.content or "Subproblem Solutions:" in thought.content


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestLeastToMostEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_simple_problem(
        self, initialized_method: LeastToMost, active_session: Session
    ):
        """Test handling of already simple problem."""
        simple_input = "What is 2 + 2?"

        thought = await initialized_method.execute(
            session=active_session, input_text=simple_input
        )

        # Should still decompose, even if simple
        assert len(initialized_method._subproblems) > 0
        assert thought.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_empty_input(
        self, initialized_method: LeastToMost, active_session: Session
    ):
        """Test handling of empty input."""
        thought = await initialized_method.execute(
            session=active_session, input_text=""
        )

        # Should still create thought with empty input
        assert thought is not None
        assert thought.metadata["input"] == ""

    @pytest.mark.asyncio
    async def test_very_long_input(
        self, initialized_method: LeastToMost, active_session: Session
    ):
        """Test handling of very long input."""
        long_input = "Solve this problem: " + "x " * 1000

        thought = await initialized_method.execute(
            session=active_session, input_text=long_input
        )

        assert thought is not None
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_multiple_executions_different_sessions(
        self, initialized_method: LeastToMost, sample_input: str
    ):
        """Test executing method on different sessions."""
        session1 = Session().start()
        session2 = Session().start()

        # Execute on first session
        thought1 = await initialized_method.execute(
            session=session1, input_text=sample_input
        )

        # Execute on second session
        thought2 = await initialized_method.execute(
            session=session2, input_text="Different problem"
        )

        # Both should succeed
        assert thought1 is not None
        assert thought2 is not None
        assert session1.thought_count == 1
        assert session2.thought_count == 1

    @pytest.mark.asyncio
    async def test_ordering_stage_metadata(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that ordering stage has correct metadata."""
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        ordering_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought
        )

        assert ordering_thought.metadata["stage"] == "ordering"
        assert "Difficulty" in ordering_thought.content

    @pytest.mark.asyncio
    async def test_parent_child_relationships(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that thoughts maintain proper parent-child relationships."""
        thoughts = []

        # Create initial thought
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )
        thoughts.append(thought)

        # Create several continuation thoughts
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
            thoughts.append(thought)

        # Verify parent-child relationships
        for i in range(1, len(thoughts)):
            assert thoughts[i].parent_id == thoughts[i - 1].id

    @pytest.mark.asyncio
    async def test_depth_increases_with_continuation(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that depth increases with each continuation."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )
        assert thought.depth == 0

        # First continuation
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        assert thought.depth == 1

        # Second continuation
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        assert thought.depth == 2

    @pytest.mark.asyncio
    async def test_session_metrics_updated(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that session metrics are properly updated."""
        initial_thoughts = active_session.metrics.total_thoughts

        await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        assert active_session.metrics.total_thoughts == initial_thoughts + 1
        assert (
            active_session.metrics.methods_used[MethodIdentifier.LEAST_TO_MOST] == 1
        )

    @pytest.mark.asyncio
    async def test_none_context_handling(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that None context is handled correctly."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input, context=None
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_none_guidance_handling(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test that None guidance is handled correctly."""
        first_thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        second_thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first_thought, guidance=None
        )

        assert second_thought.metadata["guidance"] == ""


# ============================================================================
# Test Full Reasoning Chain
# ============================================================================


class TestLeastToMostFullChain:
    """Test suite for complete reasoning chains."""

    @pytest.mark.asyncio
    async def test_complete_reasoning_chain(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test a complete reasoning chain from start to synthesis."""
        # Execute initial decomposition
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )
        assert thought.metadata["stage"] == "decomposition"

        # Ordering step
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        assert thought.metadata["stage"] == "ordering"

        # Solve all subproblems
        num_subproblems = len(initialized_method._subproblems)
        for i in range(num_subproblems):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
            assert thought.metadata["stage"] == "solving"
            assert thought.metadata["subproblems_solved"] == i + 1

        # Final synthesis
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        assert thought.metadata["stage"] == "synthesis"

        # Verify complete chain
        assert active_session.thought_count == num_subproblems + 3  # decomp + ordering + subproblems + synthesis

    @pytest.mark.asyncio
    async def test_session_state_after_completion(
        self,
        initialized_method: LeastToMost,
        active_session: Session,
        sample_input: str,
    ):
        """Test session state after completing full reasoning chain."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_input
        )

        num_subproblems = len(initialized_method._subproblems)

        # Continue through all steps
        for _ in range(num_subproblems + 2):  # ordering + subproblems + synthesis
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Check session state
        assert active_session.current_method == MethodIdentifier.LEAST_TO_MOST
        assert active_session.thought_count == num_subproblems + 3
        assert active_session.status == SessionStatus.ACTIVE
