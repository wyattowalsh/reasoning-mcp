"""Unit tests for Everything of Thoughts (EoT) reasoning method.

This module provides comprehensive tests for the EverythingOfThoughts method
implementation, covering initialization, execution, topology selection,
dynamic structure switching, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.everything_of_thoughts import (
    EVERYTHING_OF_THOUGHTS_METADATA,
    EverythingOfThoughts,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> EverythingOfThoughts:
    """Create an EverythingOfThoughts method instance for testing.

    Returns:
        A fresh EverythingOfThoughts instance
    """
    return EverythingOfThoughts()


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
        A sample complex problem string
    """
    return "Design a sustainable city infrastructure that balances economic growth, environmental protection, and social equity."


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(
        return_value="Analysis: High complexity problem requiring tree topology for exploration."
    )
    return ctx


class TestEverythingOfThoughtsInitialization:
    """Tests for EverythingOfThoughts initialization and setup."""

    def test_create_method(self, method: EverythingOfThoughts) -> None:
        """Test that EverythingOfThoughts can be instantiated."""
        assert method is not None
        assert isinstance(method, EverythingOfThoughts)

    def test_initial_state(self, method: EverythingOfThoughts) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._selected_topology == ""
        assert method._topology_history == []
        assert method._execution_count == 0

    def test_topologies_defined(self, method: EverythingOfThoughts) -> None:
        """Test that available topologies are defined."""
        assert "chain" in method.TOPOLOGIES
        assert "tree" in method.TOPOLOGIES
        assert "graph" in method.TOPOLOGIES

    async def test_initialize(self, method: EverythingOfThoughts) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._selected_topology == ""
        assert method._topology_history == []
        assert method._execution_count == 0

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = EverythingOfThoughts()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._selected_topology = "graph"
        method._topology_history = ["tree", "graph"]
        method._execution_count = 4

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._selected_topology == ""
        assert method._topology_history == []
        assert method._execution_count == 0

    async def test_health_check_not_initialized(self, method: EverythingOfThoughts) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: EverythingOfThoughts) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestEverythingOfThoughtsProperties:
    """Tests for EverythingOfThoughts property accessors."""

    def test_identifier_property(self, method: EverythingOfThoughts) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.EVERYTHING_OF_THOUGHTS

    def test_name_property(self, method: EverythingOfThoughts) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Everything of Thoughts"

    def test_description_property(self, method: EverythingOfThoughts) -> None:
        """Test that description returns the correct method description."""
        assert "meta-framework" in method.description.lower()
        assert "chain" in method.description.lower()
        assert "tree" in method.description.lower()
        assert "graph" in method.description.lower()

    def test_category_property(self, method: EverythingOfThoughts) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.HOLISTIC


class TestEverythingOfThoughtsMetadata:
    """Tests for EVERYTHING_OF_THOUGHTS metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert EVERYTHING_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.EVERYTHING_OF_THOUGHTS

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert EVERYTHING_OF_THOUGHTS_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"meta-framework", "adaptive", "topology", "chain", "tree", "graph"}
        assert expected_tags.issubset(EVERYTHING_OF_THOUGHTS_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert EVERYTHING_OF_THOUGHTS_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert EVERYTHING_OF_THOUGHTS_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has very high complexity rating."""
        assert EVERYTHING_OF_THOUGHTS_METADATA.complexity == 9


class TestEverythingOfThoughtsExecution:
    """Tests for EverythingOfThoughts execute() method."""

    async def test_execute_basic(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.EVERYTHING_OF_THOUGHTS

    async def test_execute_without_initialization_raises(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_analyze_phase(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets analyze phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "analyze"
        assert thought.metadata.get("reasoning_type") == "everything_of_thoughts"

    async def test_execute_resets_state(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute resets state for new execution."""
        await method.initialize()
        method._selected_topology = "graph"
        method._topology_history = ["tree"]
        method._execution_count = 3

        await method.execute(session, sample_problem)

        assert method._selected_topology == ""
        assert method._topology_history == []
        assert method._execution_count == 0

    async def test_execute_with_execution_context(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test execute with execution context for sampling."""
        await method.initialize()
        thought = await method.execute(
            session,
            sample_problem,
            execution_context=mock_execution_context,
        )

        assert thought is not None
        assert thought.content != ""


class TestEverythingOfThoughtsContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_analyze_to_select(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from analyze to select phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "select"
        assert continuation.type == ThoughtType.HYPOTHESIS

    async def test_continue_select_to_execute(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from select to execute phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)

        execute = await method.continue_reasoning(session, select)

        assert execute is not None
        assert execute.metadata.get("phase") == "execute"
        assert execute.type == ThoughtType.REASONING

    async def test_continue_execute_multiple_times(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test multiple execute continuations."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)

        exec1 = await method.continue_reasoning(session, select)
        exec2 = await method.continue_reasoning(session, exec1)

        assert exec1.metadata.get("phase") == "execute"
        assert exec2.metadata.get("phase") == "execute"
        assert method._execution_count >= 2

    async def test_continue_to_integrate(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to integrate phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)

        # Execute multiple times to trigger integration
        exec1 = await method.continue_reasoning(session, select)
        exec2 = await method.continue_reasoning(session, exec1)
        exec3 = await method.continue_reasoning(session, exec2)
        exec4 = await method.continue_reasoning(session, exec3)

        integrate = await method.continue_reasoning(session, exec4)

        assert integrate.metadata.get("phase") == "integrate"
        assert integrate.type == ThoughtType.SYNTHESIS

    async def test_continue_to_conclusion(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)
        exec1 = await method.continue_reasoning(session, select)
        exec2 = await method.continue_reasoning(session, exec1)
        exec3 = await method.continue_reasoning(session, exec2)
        exec4 = await method.continue_reasoning(session, exec3)
        integrate = await method.continue_reasoning(session, exec4)

        conclusion = await method.continue_reasoning(session, integrate)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: EverythingOfThoughts,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.EVERYTHING_OF_THOUGHTS,
            content="Test",
            metadata={"phase": "analyze"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestTopologySelection:
    """Tests for topology selection and switching."""

    async def test_topology_selected_during_select_phase(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that a topology is selected during select phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert method._selected_topology in method.TOPOLOGIES
        assert method._selected_topology in method._topology_history

    async def test_topology_history_tracked(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that topology history is tracked."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        assert len(method._topology_history) >= 1

    async def test_topology_switch_possible(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that topology switching is possible."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)

        # Execute to trigger switch
        exec1 = await method.continue_reasoning(session, select)
        exec2 = await method.continue_reasoning(session, exec1)
        await method.continue_reasoning(session, exec2)

        # After switch, should have new topology in history
        assert len(method._topology_history) >= 2


class TestTopologyExecution:
    """Tests for topology-specific execution."""

    async def test_execution_content_reflects_topology(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execution content reflects selected topology."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)
        execute = await method.continue_reasoning(session, select)

        topology = method._selected_topology
        assert topology in execute.content.lower() or topology.upper() in execute.content


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: EverythingOfThoughts,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: EverythingOfThoughts,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Design a solution for: " + "requirement " * 500

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: EverythingOfThoughts,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Analyze: √∑∏ → ∞ with α, β, γ constraints"

        thought = await method.execute(session, problem)

        assert thought is not None

    async def test_unicode_in_problem(
        self,
        method: EverythingOfThoughts,
        session: Session,
    ) -> None:
        """Test execution with Unicode characters."""
        await method.initialize()
        problem = "解决复杂的战略规划问题：考虑多个变量和约束"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "analyze"

        # Select topology
        select = await method.continue_reasoning(session, initial)
        assert select.metadata.get("phase") == "select"

        # Execute multiple times
        exec1 = await method.continue_reasoning(session, select)
        assert exec1.metadata.get("phase") == "execute"

        exec2 = await method.continue_reasoning(session, exec1)
        exec3 = await method.continue_reasoning(session, exec2)
        exec4 = await method.continue_reasoning(session, exec3)

        # Integrate
        integrate = await method.continue_reasoning(session, exec4)
        assert integrate.metadata.get("phase") == "integrate"

        # Conclude
        conclude = await method.continue_reasoning(session, integrate)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 7
        assert conclude.type == ThoughtType.CONCLUSION

    async def test_session_thought_count_updates(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_thought_parent_chain(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought parent chain is correct."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)
        execute = await method.continue_reasoning(session, select)

        assert initial.parent_id is None
        assert select.parent_id == initial.id
        assert execute.parent_id == select.id

    async def test_thought_depth_increments(
        self,
        method: EverythingOfThoughts,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        select = await method.continue_reasoning(session, initial)
        execute = await method.continue_reasoning(session, select)

        assert initial.depth == 0
        assert select.depth == 1
        assert execute.depth == 2
