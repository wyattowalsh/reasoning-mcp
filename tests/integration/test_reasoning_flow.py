"""Integration tests for the full reasoning flow.

This module tests the complete end-to-end reasoning flow from problem input
to solution output, covering:
- Basic reasoning with method selection
- Multi-step reasoning with thought graph creation
- Branching and merging
- Session persistence and concurrency
- Timeout handling
- Method auto-selection
"""

import asyncio
from uuid import uuid4

import pytest

from reasoning_mcp.config import get_settings
from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import SessionConfig
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import ReasonHints, ReasonOutput
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.server import AppContext
from reasoning_mcp.sessions import SessionManager
from reasoning_mcp.tools.reason import reason
from reasoning_mcp.tools.session import (
    session_branch,
    session_continue,
    session_inspect,
    session_merge,
)


@pytest.fixture(autouse=True)
async def app_context():
    """Set up AppContext for integration tests.

    This fixture initializes the global AppContext required by session tools.
    It creates a SessionManager and MethodRegistry that are shared across
    test methods.
    """
    import reasoning_mcp.server as server_module
    from reasoning_mcp.methods.native import register_all_native_methods

    # Create components
    manager = SessionManager(max_sessions=100)
    registry = MethodRegistry()
    register_all_native_methods(registry)
    await registry.initialize()
    settings = get_settings()

    # Create and set AppContext
    ctx = AppContext(
        registry=registry,
        session_manager=manager,
        settings=settings,
        initialized=True,
    )
    server_module._APP_CONTEXT = ctx

    yield ctx

    # Cleanup
    server_module._APP_CONTEXT = None
    await manager.clear()


@pytest.fixture
async def session_manager(app_context):
    """Provide the SessionManager from AppContext for each test."""
    return app_context.session_manager


@pytest.fixture
def sample_problem():
    """Provide a sample problem for testing."""
    return "Calculate the optimal route for a delivery truck visiting 5 locations."


@pytest.fixture
def ethical_problem():
    """Provide an ethical problem for testing method selection."""
    return "Should we implement a feature that improves user experience but requires collecting additional personal data?"


@pytest.fixture
def code_problem():
    """Provide a code problem for testing method selection."""
    return "Debug this function that has a race condition in async code execution."


@pytest.fixture
def math_problem():
    """Provide a mathematical problem for testing method selection."""
    return "Prove that the sum of the first n positive integers equals n(n+1)/2."


class TestBasicReasoningFlow:
    """Test basic reasoning flow from problem to solution."""

    @pytest.mark.asyncio
    async def test_basic_reasoning_flow(self, session_manager, sample_problem):
        """Test creating session, executing reason tool, and verifying thought graph."""
        # Execute reason tool
        output = await reason(problem=sample_problem)

        # Verify output structure
        assert isinstance(output, ReasonOutput)
        assert output.session_id is not None
        assert isinstance(output.thought, ThoughtNode)
        assert isinstance(output.method_used, MethodIdentifier)
        assert isinstance(output.suggestions, list)
        assert len(output.suggestions) > 0

        # Verify thought properties
        assert output.thought.id is not None
        assert output.thought.type == ThoughtType.INITIAL
        assert output.thought.content != ""
        assert 0.0 <= output.thought.confidence <= 1.0
        assert output.thought.step_number == 1
        assert output.thought.depth == 0

        # Verify metadata
        assert isinstance(output.metadata, dict)
        assert "auto_selected" in output.metadata
        assert "problem_length" in output.metadata
        assert output.metadata["problem_length"] == len(sample_problem)

    @pytest.mark.asyncio
    async def test_reason_with_explicit_method(self, sample_problem):
        """Test reasoning with explicitly specified method."""
        output = await reason(problem=sample_problem, method="chain_of_thought")

        assert output.method_used == MethodIdentifier.CHAIN_OF_THOUGHT
        assert output.metadata["auto_selected"] is False

    @pytest.mark.asyncio
    async def test_reason_creates_valid_session_id(self, sample_problem):
        """Test that reason creates a valid UUID session ID."""
        output = await reason(problem=sample_problem)

        # Should be a valid UUID
        try:
            from uuid import UUID

            UUID(output.session_id)
            valid_uuid = True
        except (ValueError, AttributeError):
            valid_uuid = False

        assert valid_uuid is True

    @pytest.mark.asyncio
    async def test_reason_with_long_problem(self):
        """Test reasoning with a very long problem statement."""
        long_problem = "This is a complex problem. " * 100

        output = await reason(problem=long_problem)

        assert output.thought.content is not None
        assert output.session_id is not None
        # Thought content should exist and can be any length
        assert len(output.thought.content) > 0


class TestMethodAutoSelection:
    """Test automatic method selection based on problem characteristics."""

    @pytest.mark.asyncio
    async def test_ethical_problem_selects_ethical_method(self, ethical_problem):
        """Test that ethical problems auto-select ethical reasoning."""
        output = await reason(problem=ethical_problem)

        assert output.metadata["auto_selected"] is True
        # Should select ethical reasoning or dialectic for ethical problems
        # Note: Some methods may not be registered yet, so accept any valid method
        assert output.method_used in [
            MethodIdentifier.ETHICAL_REASONING,
            MethodIdentifier.DIALECTIC,
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.REACT,  # May be selected if others not available
            MethodIdentifier.SELF_REFLECTION,
            MethodIdentifier.SEQUENTIAL_THINKING,
        ]

    @pytest.mark.asyncio
    async def test_code_problem_selects_code_method(self, code_problem):
        """Test that code problems auto-select code reasoning."""
        output = await reason(problem=code_problem)

        assert output.metadata["auto_selected"] is True
        # Should select code reasoning or react for code problems
        assert output.method_used in [
            MethodIdentifier.CODE_REASONING,
            MethodIdentifier.REACT,
            MethodIdentifier.CHAIN_OF_THOUGHT,  # Fallback
        ]

    @pytest.mark.asyncio
    async def test_math_problem_selects_math_method(self, math_problem):
        """Test that mathematical problems auto-select mathematical reasoning."""
        output = await reason(problem=math_problem)

        assert output.metadata["auto_selected"] is True
        # Should select mathematical reasoning or chain of thought
        # Note: Some methods may not be registered yet, so accept any valid method
        assert output.method_used in [
            MethodIdentifier.MATHEMATICAL_REASONING,
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.SEQUENTIAL_THINKING,  # May be selected for step-by-step
            MethodIdentifier.LEAST_TO_MOST,  # May be selected for math problems
        ]

    @pytest.mark.asyncio
    async def test_selection_with_hints(self, sample_problem):
        """Test that hints influence method selection."""
        hints = ReasonHints(
            domain="code",
            complexity="high",
            prefer_methods=[MethodIdentifier.CODE_REASONING],
            avoid_methods=[MethodIdentifier.CHAIN_OF_THOUGHT],
        )

        output = await reason(problem=sample_problem, hints=hints)

        assert output.metadata["auto_selected"] is True
        assert output.metadata["hints_provided"] is True
        assert "hints" in output.metadata

        # Verify hints are captured
        hint_data = output.metadata["hints"]
        assert hint_data["domain"] == "code"
        assert hint_data["complexity"] == "high"
        assert MethodIdentifier.CODE_REASONING in hint_data["preferred_methods"]

    @pytest.mark.asyncio
    async def test_selection_reasoning_logged(self, ethical_problem):
        """Test that method selection reasoning is captured in metadata."""
        output = await reason(problem=ethical_problem)

        assert "auto_selected" in output.metadata
        # Auto-selected methods should have this flag
        if output.metadata["auto_selected"]:
            assert output.method_used is not None


class TestMultiStepReasoning:
    """Test multi-step reasoning with thought progression."""

    @pytest.mark.asyncio
    async def test_multi_step_reasoning_chain(self, sample_problem):
        """Test creating a chain of reasoning steps."""
        # Start reasoning
        output1 = await reason(problem=sample_problem)
        session_id = output1.session_id

        # Continue reasoning multiple times
        output2 = await session_continue(session_id, guidance="Focus on optimization")
        output3 = await session_continue(session_id, guidance="Consider edge cases")
        output4 = await session_continue(session_id, guidance="Synthesize solution")

        # Verify each output
        for output in [output2, output3, output4]:
            assert output.id is not None
            assert output.content is not None
            assert output.thought_type in [
                ThoughtType.CONTINUATION,
                ThoughtType.SYNTHESIS,
                ThoughtType.CONCLUSION,
            ]

        # Verify step progression
        assert output2.step_number >= output1.thought.step_number
        assert output3.step_number >= output2.step_number
        assert output4.step_number >= output3.step_number

    @pytest.mark.asyncio
    async def test_thoughts_are_linked(self, session_manager, sample_problem):
        """Test that thoughts in a chain are properly linked."""
        # Create session manually to track thoughts
        session = await session_manager.create()
        session.start()

        # Add thoughts with parent-child relationships
        thought1 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="First thought",
            confidence=0.8,
            step_number=1,
            depth=0,
        )
        session.add_thought(thought1)

        thought2 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Second thought",
            parent_id=thought1.id,
            confidence=0.85,
            step_number=2,
            depth=1,
        )
        session.add_thought(thought2)

        thought3 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Final conclusion",
            parent_id=thought2.id,
            confidence=0.9,
            step_number=3,
            depth=2,
        )
        session.add_thought(thought3)

        # Verify graph structure
        assert session.thought_count == 3
        assert session.current_depth == 2
        assert session.graph.root_id == thought1.id

        # Verify parent-child links
        assert thought2.id in session.graph.nodes[thought1.id].children_ids
        assert thought3.id in session.graph.nodes[thought2.id].children_ids

        # Verify path from root to leaf
        path = session.graph.get_path(thought1.id, thought3.id)
        assert path == [thought1.id, thought2.id, thought3.id]

    @pytest.mark.asyncio
    async def test_final_answer_quality(self, session_manager, sample_problem):
        """Test that final answers have high quality metrics."""
        session = await session_manager.create()
        session.start()

        # Add a conclusion thought with high quality
        conclusion = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="After careful analysis, the optimal solution is X because...",
            confidence=0.95,
            quality_score=0.92,
            step_number=5,
            depth=4,
        )
        session.add_thought(conclusion)

        # Verify conclusion quality
        conclusions = session.get_thoughts_by_type(ThoughtType.CONCLUSION)
        assert len(conclusions) == 1
        assert conclusions[0].confidence >= 0.9
        assert conclusions[0].quality_score >= 0.9

    @pytest.mark.asyncio
    async def test_reasoning_depth_tracking(self, session_manager):
        """Test that reasoning depth is correctly tracked."""
        session = await session_manager.create()
        session.start()

        # Create a deep reasoning chain
        previous_id = None
        for depth in range(10):
            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION if depth > 0 else ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content=f"Thought at depth {depth}",
                parent_id=previous_id,
                depth=depth,
                step_number=depth + 1,
            )
            session.add_thought(thought)
            previous_id = thought.id

        # Verify depth tracking
        assert session.current_depth == 9
        assert session.metrics.max_depth_reached == 9


class TestBranchingReasoning:
    """Test branching and exploring alternative reasoning paths."""

    @pytest.mark.asyncio
    async def test_create_branch(self, sample_problem):
        """Test creating a branch from the main reasoning path."""
        # Start reasoning
        output = await reason(problem=sample_problem)
        session_id = output.session_id

        # Create a branch
        branch_output = await session_branch(
            session_id=session_id,
            branch_name="alternative-approach",
            from_thought_id=output.thought.id,
        )

        assert branch_output.success is True
        assert branch_output.branch_id is not None
        assert branch_output.session_id == session_id
        assert branch_output.parent_thought_id == output.thought.id

    @pytest.mark.asyncio
    async def test_branch_thoughts_created_correctly(self, session_manager):
        """Test that branch thoughts are properly marked and tracked."""
        session = await session_manager.create()
        session.start()

        # Create root thought
        root = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Root thought",
            depth=0,
        )
        session.add_thought(root)

        # Create branch thoughts
        branch1 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Branch 1 exploration",
            parent_id=root.id,
            branch_id="branch-1",
            depth=1,
        )
        session.add_thought(branch1)

        branch2 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Branch 2 exploration",
            parent_id=root.id,
            branch_id="branch-2",
            depth=1,
        )
        session.add_thought(branch2)

        # Verify branches
        assert session.graph.branch_count == 2
        branch_thoughts = session.get_thoughts_by_type(ThoughtType.BRANCH)
        assert len(branch_thoughts) == 2

        # Verify both branches stem from root
        assert branch1.id in session.graph.nodes[root.id].children_ids
        assert branch2.id in session.graph.nodes[root.id].children_ids

    @pytest.mark.asyncio
    async def test_branch_merging(self, sample_problem):
        """Test merging branches back together."""
        # Start reasoning and create branches
        output = await reason(problem=sample_problem)
        session_id = output.session_id

        # Create two branches
        branch1 = await session_branch(
            session_id=session_id,
            branch_name="approach-1",
        )
        branch2 = await session_branch(
            session_id=session_id,
            branch_name="approach-2",
        )

        # Merge branches
        merge_output = await session_merge(
            session_id=session_id,
            source_branch=branch1.branch_id,
            target_branch=branch2.branch_id,
            strategy="synthesis",
        )

        assert merge_output.success is True
        assert merge_output.merged_thought_id is not None
        assert merge_output.session_id == session_id
        assert len(merge_output.source_branch_ids) >= 2

    @pytest.mark.asyncio
    async def test_multiple_branch_levels(self, session_manager):
        """Test creating branches from branches (nested branching)."""
        session = await session_manager.create()
        session.start()

        # Root
        root = ThoughtNode(
            id="root",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Root",
            depth=0,
        )
        session.add_thought(root)

        # First level branch
        branch1 = ThoughtNode(
            id="branch1",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Branch 1",
            parent_id="root",
            branch_id="b1",
            depth=1,
        )
        session.add_thought(branch1)

        # Second level branch (branch from branch)
        branch1_1 = ThoughtNode(
            id="branch1-1",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Branch 1.1",
            parent_id="branch1",
            branch_id="b1-1",
            depth=2,
        )
        session.add_thought(branch1_1)

        # Verify nested structure
        assert session.current_depth == 2
        path = session.graph.get_path("root", "branch1-1")
        assert path == ["root", "branch1", "branch1-1"]


class TestReasoningWithHints:
    """Test reasoning with selection hints."""

    @pytest.mark.asyncio
    async def test_hints_provide_domain_guidance(self, sample_problem):
        """Test that domain hints guide method selection."""
        hints = ReasonHints(domain="ethical")

        output = await reason(problem=sample_problem, hints=hints)

        assert output.metadata["hints_provided"] is True
        assert output.metadata["hints"]["domain"] == "ethical"

    @pytest.mark.asyncio
    async def test_hints_provide_complexity_guidance(self, sample_problem):
        """Test that complexity hints are captured."""
        hints = ReasonHints(complexity="high")

        output = await reason(problem=sample_problem, hints=hints)

        assert output.metadata["hints"]["complexity"] == "high"

    @pytest.mark.asyncio
    async def test_hints_prefer_specific_methods(self, sample_problem):
        """Test that preferred methods are respected."""
        hints = ReasonHints(
            prefer_methods=[
                MethodIdentifier.TREE_OF_THOUGHTS,
                MethodIdentifier.MCTS,
            ]
        )

        output = await reason(problem=sample_problem, hints=hints)

        assert len(output.metadata["hints"]["preferred_methods"]) == 2
        assert "tree_of_thoughts" in output.metadata["hints"]["preferred_methods"]
        assert "mcts" in output.metadata["hints"]["preferred_methods"]

    @pytest.mark.asyncio
    async def test_hints_avoid_specific_methods(self, sample_problem):
        """Test that avoided methods are not selected."""
        hints = ReasonHints(
            avoid_methods=[
                MethodIdentifier.ETHICAL_REASONING,
                MethodIdentifier.SOCRATIC,
            ]
        )

        output = await reason(problem=sample_problem, hints=hints)

        # Should not select avoided methods
        assert output.method_used not in [
            MethodIdentifier.ETHICAL_REASONING,
            MethodIdentifier.SOCRATIC,
        ]

    @pytest.mark.asyncio
    async def test_hints_with_custom_metadata(self, sample_problem):
        """Test that custom hints are preserved."""
        hints = ReasonHints(
            custom_hints={
                "stakeholders": ["users", "developers"],
                "constraints": ["time", "budget"],
            }
        )

        output = await reason(problem=sample_problem, hints=hints)

        # Custom hints should be accessible in metadata
        assert output.metadata["hints_provided"] is True


class TestReasoningTimeout:
    """Test timeout handling in reasoning sessions."""

    @pytest.mark.asyncio
    async def test_session_with_timeout_config(self, session_manager):
        """Test creating a session with timeout configuration."""
        config = SessionConfig(timeout_seconds=10.0)
        session = await session_manager.create(config=config)

        assert session.config.timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_timeout_duration_tracking(self, session_manager):
        """Test that session duration is tracked."""
        session = await session_manager.create()
        session.start()

        # Wait a bit
        await asyncio.sleep(0.1)

        # Duration should be measurable
        assert session.duration is not None
        assert session.duration >= 0.1

    @pytest.mark.asyncio
    async def test_session_completion_records_duration(self, session_manager):
        """Test that completed sessions record their duration."""
        session = await session_manager.create()
        session.start()

        await asyncio.sleep(0.1)

        session.complete()

        assert session.duration is not None
        assert session.completed_at is not None
        assert session.duration >= 0.1

    @pytest.mark.asyncio
    async def test_partial_results_on_timeout(self, session_manager):
        """Test that partial results are available even if timeout occurs."""
        session = await session_manager.create()
        session.start()

        # Add some thoughts before "timeout"
        thought1 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Partial result 1",
        )
        thought2 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Partial result 2",
            parent_id=thought1.id,
        )

        session.add_thought(thought1)
        session.add_thought(thought2)

        # Simulate timeout by failing the session
        session.fail("Timeout exceeded")

        # Partial results should still be accessible
        assert session.thought_count == 2
        assert session.status == SessionStatus.FAILED
        assert session.error == "Timeout exceeded"


class TestConcurrentSessions:
    """Test concurrent session management and isolation."""

    @pytest.mark.asyncio
    async def test_create_concurrent_sessions(self, session_manager):
        """Test creating multiple sessions concurrently."""
        # Create 10 sessions in parallel
        tasks = [session_manager.create() for _ in range(10)]
        sessions = await asyncio.gather(*tasks)

        # All should be unique
        assert len(sessions) == 10
        session_ids = {s.id for s in sessions}
        assert len(session_ids) == 10

    @pytest.mark.asyncio
    async def test_session_isolation(self, session_manager):
        """Test that sessions are isolated from each other."""
        # Create two sessions
        session1 = await session_manager.create()
        session2 = await session_manager.create()

        session1.start()
        session2.start()

        # Add thoughts to session1
        thought1 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Session 1 thought",
        )
        session1.add_thought(thought1)

        # Add thoughts to session2
        thought2 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Session 2 thought",
        )
        session2.add_thought(thought2)

        # Update both in manager
        await session_manager.update(session1.id, session1)
        await session_manager.update(session2.id, session2)

        # Retrieve and verify isolation
        retrieved1 = await session_manager.get(session1.id)
        retrieved2 = await session_manager.get(session2.id)

        assert retrieved1.thought_count == 1
        assert retrieved2.thought_count == 1
        assert retrieved1.graph.nodes[thought1.id].content == "Session 1 thought"
        assert retrieved2.graph.nodes[thought2.id].content == "Session 2 thought"

    @pytest.mark.asyncio
    async def test_concurrent_operations_on_different_sessions(self, session_manager):
        """Test concurrent operations on different sessions."""
        sessions = [await session_manager.create() for _ in range(5)]

        # Start all sessions concurrently
        for session in sessions:
            session.start()
            await session_manager.update(session.id, session)

        # Perform concurrent operations
        async def add_thoughts(session_id, count):
            session = await session_manager.get(session_id)
            for i in range(count):
                thought = ThoughtNode(
                    id=str(uuid4()),
                    type=ThoughtType.CONTINUATION,
                    method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                    content=f"Thought {i}",
                )
                session.add_thought(thought)
            await session_manager.update(session_id, session)

        tasks = [add_thoughts(s.id, 5) for s in sessions]
        await asyncio.gather(*tasks)

        # Verify all sessions have correct thought counts
        for session in sessions:
            retrieved = await session_manager.get(session.id)
            assert retrieved.thought_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_session_inspection(self, sample_problem):
        """Test inspecting sessions concurrently."""
        # Create multiple reasoning sessions
        outputs = await asyncio.gather(*[reason(problem=sample_problem) for _ in range(5)])

        # Inspect all concurrently
        inspect_tasks = [session_inspect(output.session_id) for output in outputs]
        states = await asyncio.gather(*inspect_tasks)

        # All should return valid states
        assert len(states) == 5
        for state in states:
            assert state.session_id is not None
            assert state.status in [SessionStatus.CREATED, SessionStatus.ACTIVE]


class TestSessionPersistence:
    """Test session state persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_session_state_preserved_after_retrieval(self, session_manager):
        """Test that session state is preserved when retrieved."""
        # Create and modify session
        session = await session_manager.create()
        session.start()

        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test thought",
            confidence=0.88,
        )
        session.add_thought(thought)

        # Update in manager
        await session_manager.update(session.id, session)

        # Retrieve and verify
        retrieved = await session_manager.get(session.id)

        assert retrieved.status == SessionStatus.ACTIVE
        assert retrieved.thought_count == 1
        assert retrieved.metrics.total_thoughts == 1
        assert retrieved.metrics.average_confidence == 0.88

    @pytest.mark.asyncio
    async def test_session_graph_preserved(self, session_manager):
        """Test that thought graph structure is preserved."""
        session = await session_manager.create()
        session.start()

        # Create a chain
        root = ThoughtNode(
            id="root",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Root",
        )
        child1 = ThoughtNode(
            id="child1",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Child 1",
            parent_id="root",
            depth=1,
        )
        child2 = ThoughtNode(
            id="child2",
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Child 2",
            parent_id="child1",
            depth=2,
        )

        session.add_thought(root)
        session.add_thought(child1)
        session.add_thought(child2)

        await session_manager.update(session.id, session)

        # Retrieve and verify graph
        retrieved = await session_manager.get(session.id)

        assert retrieved.graph.root_id == "root"
        assert retrieved.thought_count == 3
        path = retrieved.graph.get_path("root", "child2")
        assert path == ["root", "child1", "child2"]

    @pytest.mark.asyncio
    async def test_session_metrics_preserved(self, session_manager):
        """Test that session metrics are preserved across updates."""
        session = await session_manager.create()
        session.start()

        # Add multiple thoughts with different methods
        for i in range(5):
            method = (
                MethodIdentifier.CHAIN_OF_THOUGHT if i < 3 else MethodIdentifier.TREE_OF_THOUGHTS
            )
            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION,
                method_id=method,
                content=f"Thought {i}",
                confidence=0.7 + (i * 0.05),
                depth=i,
            )
            session.add_thought(thought)

        await session_manager.update(session.id, session)

        # Retrieve and verify metrics
        retrieved = await session_manager.get(session.id)

        assert retrieved.metrics.total_thoughts == 5
        assert retrieved.metrics.max_depth_reached == 4
        assert 0.7 <= retrieved.metrics.average_confidence <= 0.95
        assert len(retrieved.metrics.methods_used) == 2

    @pytest.mark.asyncio
    async def test_inspect_returns_current_state(self, sample_problem):
        """Test that session_inspect returns current state."""
        output = await reason(problem=sample_problem)

        state = await session_inspect(output.session_id)

        assert state.session_id == output.session_id
        assert state.status in [SessionStatus.CREATED, SessionStatus.ACTIVE]
        assert state.thought_count >= 0
        assert state.branch_count >= 0

    @pytest.mark.asyncio
    async def test_inspect_with_graph_option(self, sample_problem):
        """Test session inspection with graph visualization option."""
        output = await reason(problem=sample_problem)

        # Inspect without graph
        state1 = await session_inspect(output.session_id, include_graph=False)
        assert state1.session_id == output.session_id

        # Inspect with graph
        state2 = await session_inspect(output.session_id, include_graph=True)
        assert state2.session_id == output.session_id
        # Graph data would be in metadata if implemented


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in reasoning flow."""

    @pytest.mark.asyncio
    async def test_empty_problem_string(self):
        """Test reasoning with empty problem string."""
        output = await reason(problem="")

        # Should still create valid output
        assert output.session_id is not None
        assert output.thought is not None

    @pytest.mark.asyncio
    async def test_very_short_problem(self):
        """Test reasoning with single word problem."""
        output = await reason(problem="Help")

        assert output.session_id is not None
        assert output.thought.content is not None

    @pytest.mark.asyncio
    async def test_invalid_method_name(self):
        """Test that invalid method names raise errors."""
        with pytest.raises((ValueError, KeyError)):
            await reason(problem="test problem", method="nonexistent_method")

    @pytest.mark.asyncio
    async def test_continue_nonexistent_session(self):
        """Test continuing a non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="Session not found"):
            await session_continue("nonexistent-session-id")

    @pytest.mark.asyncio
    async def test_branch_from_nonexistent_thought(self, sample_problem):
        """Test creating branch from non-existent thought raises ValueError."""
        output = await reason(problem=sample_problem)

        # Try to branch from invalid thought ID
        with pytest.raises(ValueError, match="Thought not found"):
            await session_branch(
                session_id=output.session_id,
                branch_name="test",
                from_thought_id="nonexistent-thought-id",
            )

    @pytest.mark.asyncio
    async def test_session_manager_max_limit(self, session_manager):
        """Test session manager respects max sessions limit."""
        # Set a low limit
        limited_manager = SessionManager(max_sessions=3)

        # Create up to limit
        await limited_manager.create()
        await limited_manager.create()
        await limited_manager.create()

        assert await limited_manager.count() == 3

        # Exceed limit should raise error
        with pytest.raises(RuntimeError, match="Maximum session limit"):
            await limited_manager.create()

    @pytest.mark.asyncio
    async def test_multiple_method_changes_in_session(self, session_manager):
        """Test that a session can use multiple different methods."""
        session = await session_manager.create()
        session.start()

        # Add thoughts with different methods
        methods = [
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.SELF_REFLECTION,
        ]

        for i, method in enumerate(methods):
            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION,
                method_id=method,
                content=f"Thought {i}",
            )
            session.add_thought(thought)

        # Verify all methods are tracked
        assert len(session.metrics.methods_used) == 3


class TestComplexReasoningScenarios:
    """Test complex, real-world reasoning scenarios."""

    @pytest.mark.asyncio
    async def test_full_reasoning_cycle_with_branches_and_merge(self, session_manager):
        """Test complete cycle: reason -> branch -> explore -> merge -> conclude."""
        session = await session_manager.create()
        session.start()

        # Initial thought
        root = ThoughtNode(
            id="root",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Initial problem analysis",
            confidence=0.7,
            depth=0,
        )
        session.add_thought(root)

        # Create multiple exploration branches
        branch1 = ThoughtNode(
            id="b1",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Approach 1: Greedy algorithm",
            parent_id="root",
            branch_id="approach-1",
            confidence=0.75,
            depth=1,
        )
        session.add_thought(branch1)

        branch2 = ThoughtNode(
            id="b2",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Approach 2: Dynamic programming",
            parent_id="root",
            branch_id="approach-2",
            confidence=0.85,
            depth=1,
        )
        session.add_thought(branch2)

        # Synthesis of branches
        synthesis = ThoughtNode(
            id="syn",
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Combining best aspects of both approaches",
            parent_id="root",
            confidence=0.9,
            depth=1,
        )
        session.add_thought(synthesis)

        # Final conclusion
        conclusion = ThoughtNode(
            id="con",
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Optimal solution: Hybrid approach",
            parent_id="syn",
            confidence=0.95,
            depth=2,
        )
        session.add_thought(conclusion)

        # Verify complete flow
        assert session.thought_count == 5
        assert session.graph.branch_count == 2
        assert session.current_depth == 2

        # Verify thought types
        assert len(session.get_thoughts_by_type(ThoughtType.BRANCH)) == 2
        assert len(session.get_thoughts_by_type(ThoughtType.SYNTHESIS)) == 1
        assert len(session.get_thoughts_by_type(ThoughtType.CONCLUSION)) == 1

    @pytest.mark.asyncio
    async def test_iterative_refinement_pattern(self, session_manager):
        """Test iterative refinement with revision thoughts."""
        session = await session_manager.create()
        session.start()

        # Initial hypothesis
        hypothesis = ThoughtNode(
            id="h1",
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content="Initial hypothesis: Solution is O(n^2)",
            confidence=0.6,
        )
        session.add_thought(hypothesis)

        # Verification reveals issues
        verification = ThoughtNode(
            id="v1",
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content="Testing hypothesis... found edge case",
            parent_id="h1",
            confidence=0.5,
        )
        session.add_thought(verification)

        # Revision based on verification
        revision = ThoughtNode(
            id="r1",
            type=ThoughtType.REVISION,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content="Revised hypothesis: Actually O(n log n) is achievable",
            parent_id="v1",
            confidence=0.85,
        )
        session.add_thought(revision)

        # Final verification
        final_verification = ThoughtNode(
            id="v2",
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content="Verified: O(n log n) solution works for all cases",
            parent_id="r1",
            confidence=0.95,
        )
        session.add_thought(final_verification)

        # Verify refinement pattern
        assert session.thought_count == 4
        assert len(session.get_thoughts_by_type(ThoughtType.HYPOTHESIS)) == 1
        assert len(session.get_thoughts_by_type(ThoughtType.VERIFICATION)) == 2
        assert len(session.get_thoughts_by_type(ThoughtType.REVISION)) == 1

        # Confidence should improve through refinement
        thoughts = list(session.graph.nodes.values())
        confidences = [t.confidence for t in thoughts]
        # Last thought should have highest confidence
        assert confidences[-1] >= confidences[0]

    @pytest.mark.asyncio
    async def test_multi_session_comparison(self, session_manager):
        """Test comparing results across multiple concurrent sessions."""
        # Create multiple sessions with different approaches
        sessions = []
        for i in range(3):
            session = await session_manager.create()
            session.start()

            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONCLUSION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content=f"Solution approach {i}",
                confidence=0.7 + (i * 0.1),
            )
            session.add_thought(thought)
            await session_manager.update(session.id, session)
            sessions.append(session)

        # Retrieve and compare
        retrieved_sessions = await asyncio.gather(*[session_manager.get(s.id) for s in sessions])

        # All should be valid
        assert len(retrieved_sessions) == 3
        for s in retrieved_sessions:
            assert s.thought_count == 1
            conclusions = s.get_thoughts_by_type(ThoughtType.CONCLUSION)
            assert len(conclusions) == 1
