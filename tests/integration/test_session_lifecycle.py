"""Integration tests for session lifecycle management.

This module tests the complete lifecycle of reasoning sessions, including:
- Session creation with various configurations
- State transitions (CREATED -> ACTIVE -> PAUSED -> ACTIVE -> COMPLETED/FAILED)
- Thought management and graph updates
- Metrics tracking and computation
- Session cleanup and archiving
- Session timeout handling
- Export functionality (JSON, Mermaid, GraphViz)
- Concurrent operations and thread safety
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import Session, SessionConfig, SessionMetrics
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.sessions import SessionManager


class TestSessionCreation:
    """Test session creation with various configurations."""

    @pytest.mark.asyncio
    async def test_session_creation_with_defaults(self):
        """Test creating a session with default configuration."""
        session = Session()

        # Verify initial state
        assert session.id is not None
        assert session.status == SessionStatus.CREATED
        assert session.config is not None
        assert session.config.max_depth == 10
        assert session.config.max_thoughts == 100
        assert session.config.timeout_seconds == 300.0
        assert session.graph is not None
        assert session.metrics is not None
        assert session.created_at is not None
        assert session.started_at is None
        assert session.completed_at is None
        assert session.error is None
        assert session.is_active is False
        assert session.is_complete is False

    @pytest.mark.asyncio
    async def test_session_creation_with_custom_config(self):
        """Test creating a session with custom configuration."""
        config = SessionConfig(
            max_depth=20,
            max_thoughts=500,
            timeout_seconds=600.0,
            enable_branching=True,
            max_branches=10,
            auto_prune=True,
            min_confidence_threshold=0.5,
            metadata={"project": "test", "priority": "high"},
        )
        session = Session(config=config)

        # Verify custom config applied
        assert session.config.max_depth == 20
        assert session.config.max_thoughts == 500
        assert session.config.timeout_seconds == 600.0
        assert session.config.enable_branching is True
        assert session.config.max_branches == 10
        assert session.config.auto_prune is True
        assert session.config.min_confidence_threshold == 0.5
        assert session.config.metadata["project"] == "test"
        assert session.config.metadata["priority"] == "high"

    @pytest.mark.asyncio
    async def test_session_creation_with_manager(self):
        """Test creating sessions through SessionManager."""
        manager = SessionManager(max_sessions=10)

        # Create session with default config
        session1 = await manager.create()
        assert session1.status == SessionStatus.CREATED
        assert await manager.count() == 1

        # Create session with custom config
        config = SessionConfig(max_depth=15, timeout_seconds=450.0)
        session2 = await manager.create(config=config)
        assert session2.config.max_depth == 15
        assert session2.config.timeout_seconds == 450.0
        assert await manager.count() == 2

        # Verify sessions are different
        assert session1.id != session2.id

    @pytest.mark.asyncio
    async def test_session_creation_fields_initialized(self):
        """Test all session fields are properly initialized."""
        session = Session()

        # Identity and status
        assert len(session.id) > 0
        assert session.status == SessionStatus.CREATED

        # Graph and metrics
        assert session.graph.node_count == 0
        assert session.graph.edge_count == 0
        assert session.metrics.total_thoughts == 0
        assert session.metrics.total_edges == 0
        assert session.metrics.average_confidence == 0.0

        # Optional fields
        assert session.current_method is None
        assert session.active_branch_id is None
        assert session.started_at is None
        assert session.completed_at is None
        assert session.error is None

        # Metadata
        assert isinstance(session.metadata, dict)
        assert len(session.metadata) == 0


class TestSessionStateTransitions:
    """Test session state transitions and lifecycle."""

    @pytest.mark.asyncio
    async def test_happy_path_created_to_completed(self):
        """Test CREATED -> ACTIVE -> COMPLETED transition."""
        session = Session()
        assert session.status == SessionStatus.CREATED

        # Start session
        session.start()
        assert session.status == SessionStatus.ACTIVE
        assert session.is_active is True
        assert session.started_at is not None

        # Complete session
        session.complete()
        assert session.status == SessionStatus.COMPLETED
        assert session.is_complete is True
        assert session.completed_at is not None
        assert session.duration is not None
        assert session.duration >= 0.0

    @pytest.mark.asyncio
    async def test_pause_and_resume_transition(self):
        """Test CREATED -> ACTIVE -> PAUSED -> ACTIVE -> COMPLETED."""
        session = Session()

        # Start
        session.start()
        assert session.status == SessionStatus.ACTIVE
        start_time = session.started_at

        # Pause
        session.pause()
        assert session.status == SessionStatus.PAUSED
        assert session.is_active is False
        assert session.started_at == start_time  # Should not change

        # Resume
        session.resume()
        assert session.status == SessionStatus.ACTIVE
        assert session.is_active is True

        # Complete
        session.complete()
        assert session.status == SessionStatus.COMPLETED
        assert session.is_complete is True

    @pytest.mark.asyncio
    async def test_failure_transition(self):
        """Test CREATED -> ACTIVE -> FAILED transition."""
        session = Session()

        session.start()
        assert session.status == SessionStatus.ACTIVE

        # Fail the session
        session.fail("Something went wrong")
        assert session.status == SessionStatus.FAILED
        assert session.is_complete is True
        assert session.error == "Something went wrong"
        assert session.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancellation_transition(self):
        """Test CREATED -> ACTIVE -> CANCELLED transition."""
        session = Session()

        session.start()
        assert session.status == SessionStatus.ACTIVE

        # Cancel the session
        session.cancel()
        assert session.status == SessionStatus.CANCELLED
        assert session.is_complete is True
        assert session.completed_at is not None

    @pytest.mark.asyncio
    async def test_invalid_transition_start_when_active(self):
        """Test that starting an active session raises an error."""
        session = Session()
        session.start()

        with pytest.raises(ValueError, match="already active"):
            session.start()

    @pytest.mark.asyncio
    async def test_invalid_transition_start_when_completed(self):
        """Test that starting a completed session raises an error."""
        session = Session()
        session.start()
        session.complete()

        with pytest.raises(ValueError, match="Cannot start a completed session"):
            session.start()

    @pytest.mark.asyncio
    async def test_invalid_transition_pause_when_not_active(self):
        """Test that pausing a non-active session raises an error."""
        session = Session()

        with pytest.raises(ValueError, match="Can only pause an active session"):
            session.pause()

    @pytest.mark.asyncio
    async def test_invalid_transition_resume_when_not_paused(self):
        """Test that resuming a non-paused session raises an error."""
        session = Session()

        with pytest.raises(ValueError, match="Can only resume a paused session"):
            session.resume()


class TestSessionThoughtManagement:
    """Test session thought management and graph updates."""

    @pytest.mark.asyncio
    async def test_add_single_thought(self):
        """Test adding a single thought to a session."""
        session = Session().start()

        thought = ThoughtNode(
            id="thought1",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="First thought",
            confidence=0.8,
        )

        session.add_thought(thought)

        # Verify thought added to graph
        assert session.thought_count == 1
        assert session.graph.node_count == 1
        assert session.graph.get_node("thought1") is not None

        # Verify graph root set
        assert session.graph.root_id == "thought1"

    @pytest.mark.asyncio
    async def test_add_thoughts_with_parent_child_relationship(self):
        """Test adding thoughts with parent-child relationships."""
        session = Session().start()

        # Add root thought
        root = ThoughtNode(
            id="root",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Root thought",
            confidence=0.9,
            depth=0,
        )
        session.add_thought(root)

        # Add child thought
        child = ThoughtNode(
            id="child",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Child thought",
            parent_id="root",
            confidence=0.85,
            depth=1,
        )
        session.add_thought(child)

        # Verify graph structure
        assert session.thought_count == 2
        assert session.graph.edge_count == 1

        # Verify parent-child relationship
        root_node = session.graph.get_node("root")
        assert "child" in root_node.children_ids

        # Verify edge created
        assert session.graph.edge_count == 1

    @pytest.mark.asyncio
    async def test_add_thoughts_creates_tree_structure(self):
        """Test adding multiple thoughts creates proper tree structure."""
        session = Session().start()

        # Create a tree: root -> (child1, child2) -> grandchild
        root = ThoughtNode(
            id="root",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Root",
            depth=0,
        )
        session.add_thought(root)

        child1 = ThoughtNode(
            id="child1",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Child 1",
            parent_id="root",
            depth=1,
        )
        session.add_thought(child1)

        child2 = ThoughtNode(
            id="child2",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Child 2",
            parent_id="root",
            depth=1,
        )
        session.add_thought(child2)

        grandchild = ThoughtNode(
            id="grandchild",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Grandchild",
            parent_id="child1",
            depth=2,
        )
        session.add_thought(grandchild)

        # Verify structure
        assert session.thought_count == 4
        assert session.graph.edge_count == 3
        assert session.current_depth == 2

        # Verify relationships
        root_node = session.graph.get_node("root")
        assert len(root_node.children_ids) == 2
        assert "child1" in root_node.children_ids
        assert "child2" in root_node.children_ids

    @pytest.mark.asyncio
    async def test_thought_retrieval_by_id(self):
        """Test retrieving thoughts by ID."""
        session = Session().start()

        thought1 = ThoughtNode(
            id="t1",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Thought 1",
        )
        thought2 = ThoughtNode(
            id="t2",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Thought 2",
            parent_id="t1",
        )

        session.add_thought(thought1)
        session.add_thought(thought2)

        # Retrieve by ID
        retrieved1 = session.graph.get_node("t1")
        retrieved2 = session.graph.get_node("t2")

        assert retrieved1 is not None
        assert retrieved1.id == "t1"
        assert retrieved1.content == "Thought 1"

        assert retrieved2 is not None
        assert retrieved2.id == "t2"
        assert retrieved2.content == "Thought 2"

        # Non-existent ID
        assert session.graph.get_node("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_thoughts_by_method(self):
        """Test filtering thoughts by reasoning method."""
        session = Session().start()

        cot_thought = ThoughtNode(
            id="cot",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="CoT thought",
        )
        tot_thought = ThoughtNode(
            id="tot",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="ToT thought",
        )

        session.add_thought(cot_thought)
        session.add_thought(tot_thought)

        # Filter by method
        cot_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)
        tot_thoughts = session.get_thoughts_by_method(MethodIdentifier.TREE_OF_THOUGHTS)

        assert len(cot_thoughts) == 1
        assert cot_thoughts[0].id == "cot"

        assert len(tot_thoughts) == 1
        assert tot_thoughts[0].id == "tot"


class TestSessionMetricsTracking:
    """Test session metrics tracking and computation."""

    @pytest.mark.asyncio
    async def test_metrics_update_on_thought_addition(self):
        """Test that metrics update when thoughts are added."""
        session = Session().start()

        thought = ThoughtNode(
            id="t1",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test",
            confidence=0.8,
            quality_score=0.9,
            depth=1,
        )

        session.add_thought(thought)

        # Verify metrics updated
        assert session.metrics.total_thoughts == 1
        assert session.metrics.average_confidence == 0.8
        assert session.metrics.average_quality == 0.9
        assert session.metrics.max_depth_reached == 1

    @pytest.mark.asyncio
    async def test_average_confidence_computation(self):
        """Test average confidence calculation with multiple thoughts."""
        session = Session().start()

        thoughts = [
            ThoughtNode(
                id=f"t{i}",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content=f"Thought {i}",
                confidence=0.5 + (i * 0.1),
                depth=i,
            )
            for i in range(1, 6)  # 0.6, 0.7, 0.8, 0.9, 1.0
        ]

        for thought in thoughts:
            session.add_thought(thought)

        # Average should be (0.6 + 0.7 + 0.8 + 0.9 + 1.0) / 5 = 0.8
        assert session.metrics.total_thoughts == 5
        assert abs(session.metrics.average_confidence - 0.8) < 0.001

    @pytest.mark.asyncio
    async def test_max_depth_tracking(self):
        """Test maximum depth tracking."""
        session = Session().start()

        # Add thoughts at various depths
        thoughts = [
            ThoughtNode(
                id=f"t{i}",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content=f"Depth {i}",
                depth=i,
            )
            for i in [0, 2, 5, 3, 7, 1]
        ]

        for thought in thoughts:
            session.add_thought(thought)

        assert session.metrics.max_depth_reached == 7
        assert session.current_depth == 7

    @pytest.mark.asyncio
    async def test_method_usage_tracking(self):
        """Test tracking of which methods were used."""
        session = Session().start()

        # Add thoughts from different methods
        session.add_thought(
            ThoughtNode(
                id="cot1",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="CoT 1",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="cot2",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="CoT 2",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="tot1",
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                content="ToT 1",
            )
        )

        # Verify method usage counts
        methods = session.metrics.methods_used
        assert methods[str(MethodIdentifier.CHAIN_OF_THOUGHT)] == 2
        assert methods[str(MethodIdentifier.TREE_OF_THOUGHTS)] == 1

    @pytest.mark.asyncio
    async def test_thought_type_tracking(self):
        """Test tracking of thought types."""
        session = Session().start()

        session.add_thought(
            ThoughtNode(
                id="initial",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Initial",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="cont1",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Continuation 1",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="cont2",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Continuation 2",
            )
        )

        # Verify type counts
        types = session.metrics.thought_types
        assert types[str(ThoughtType.INITIAL)] == 1
        assert types[str(ThoughtType.CONTINUATION)] == 2

    @pytest.mark.asyncio
    async def test_timing_metrics(self):
        """Test session timing metrics."""
        session = Session()
        created_time = session.created_at

        # Not started yet
        assert session.duration is None

        # Start session
        session.start()
        assert session.started_at is not None
        assert session.duration is not None
        assert session.duration >= 0.0

        # Wait a bit
        await asyncio.sleep(0.1)

        # Complete session
        session.complete()
        final_duration = session.duration

        assert session.completed_at is not None
        assert final_duration is not None
        assert final_duration >= 0.1


class TestSessionCleanup:
    """Test session cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_session_completion_cleanup(self):
        """Test that completed sessions can be properly cleaned up."""
        manager = SessionManager()

        # Create and complete a session
        session = await manager.create()
        session.start()
        session.add_thought(
            ThoughtNode(
                id="t1",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Test",
            )
        )
        session.complete()
        await manager.update(session.id, session)

        # Verify session is completed
        retrieved = await manager.get(session.id)
        assert retrieved.status == SessionStatus.COMPLETED

        # Delete completed session
        deleted = await manager.delete(session.id)
        assert deleted is True
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions based on age."""
        manager = SessionManager()

        # Create old session
        old_session = await manager.create()
        old_session.created_at = datetime.now() - timedelta(hours=25)
        await manager.update(old_session.id, old_session)

        # Create recent session
        recent_session = await manager.create()

        assert await manager.count() == 2

        # Cleanup sessions older than 24 hours
        removed = await manager.cleanup_expired(max_age_seconds=86400)

        assert removed == 1
        assert await manager.count() == 1

        # Verify recent session remains
        remaining = await manager.get(recent_session.id)
        assert remaining is not None

    @pytest.mark.asyncio
    async def test_archive_completed_session(self):
        """Test archiving a completed session."""
        session = Session()
        session.start()

        # Add some thoughts
        session.add_thought(
            ThoughtNode(
                id="t1",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="First",
                confidence=0.8,
            )
        )
        session.add_thought(
            ThoughtNode(
                id="t2",
                type=ThoughtType.CONCLUSION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Conclusion",
                confidence=0.9,
                parent_id="t1",
                depth=1,
            )
        )

        session.complete()

        # Archive session (convert to dict for storage)
        archived_data = session.model_dump()

        # Verify archived data contains all necessary information
        assert archived_data["id"] == session.id
        assert archived_data["status"] == SessionStatus.COMPLETED
        assert archived_data["graph"]["nodes"]["t1"] is not None
        assert archived_data["graph"]["nodes"]["t2"] is not None
        assert archived_data["metrics"]["total_thoughts"] == 2

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self):
        """Test clearing all sessions from manager."""
        manager = SessionManager()

        # Create multiple sessions
        for _ in range(5):
            await manager.create()

        assert await manager.count() == 5

        # Clear all
        await manager.clear()

        assert await manager.count() == 0


class TestSessionTimeout:
    """Test session timeout handling."""

    @pytest.mark.asyncio
    async def test_session_timeout_configuration(self):
        """Test that timeout is properly configured."""
        config = SessionConfig(timeout_seconds=10.0)
        session = Session(config=config)

        assert session.config.timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_session_duration_tracking(self):
        """Test session duration is tracked correctly."""
        session = Session()
        session.start()

        # Wait a bit
        await asyncio.sleep(0.1)

        duration = session.duration
        assert duration is not None
        assert duration >= 0.1

    @pytest.mark.asyncio
    async def test_check_if_session_exceeded_timeout(self):
        """Test checking if session exceeded configured timeout."""
        config = SessionConfig(timeout_seconds=0.1)
        session = Session(config=config)
        session.start()

        # Wait for timeout to exceed
        await asyncio.sleep(0.2)

        # Check if timed out
        duration = session.duration
        assert duration > config.timeout_seconds

    @pytest.mark.asyncio
    async def test_fail_session_on_timeout(self):
        """Test failing a session when it times out."""
        config = SessionConfig(timeout_seconds=0.1)
        session = Session(config=config)
        session.start()

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Fail session due to timeout
        if session.duration > config.timeout_seconds:
            session.fail(f"Session timed out after {session.duration:.2f} seconds")

        assert session.status == SessionStatus.FAILED
        assert "timed out" in session.error


class TestSessionExport:
    """Test session export in various formats."""

    @pytest.mark.asyncio
    async def test_export_session_to_json(self):
        """Test exporting session to JSON format."""
        session = Session()
        session.start()

        # Add thoughts
        session.add_thought(
            ThoughtNode(
                id="root",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Root thought",
                confidence=0.9,
            )
        )
        session.add_thought(
            ThoughtNode(
                id="child",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Child thought",
                parent_id="root",
                confidence=0.85,
                depth=1,
            )
        )

        session.complete()

        # Export to JSON
        json_data = session.model_dump()
        json_str = json.dumps(json_data, default=str)

        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert parsed["id"] == session.id
        assert parsed["status"] == SessionStatus.COMPLETED
        assert "root" in parsed["graph"]["nodes"]
        assert "child" in parsed["graph"]["nodes"]

    @pytest.mark.asyncio
    async def test_export_session_to_mermaid(self):
        """Test exporting session thought graph to Mermaid format."""
        session = Session()
        session.start()

        # Create a simple graph
        session.add_thought(
            ThoughtNode(
                id="root",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Root",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="child1",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Child 1",
                parent_id="root",
                depth=1,
            )
        )
        session.add_thought(
            ThoughtNode(
                id="child2",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Child 2",
                parent_id="root",
                depth=1,
            )
        )

        # Generate Mermaid diagram
        mermaid = self._generate_mermaid(session)

        # Verify Mermaid format
        assert "graph TD" in mermaid or "graph LR" in mermaid
        assert "root" in mermaid
        assert "child1" in mermaid
        assert "child2" in mermaid
        assert "-->" in mermaid

    @pytest.mark.asyncio
    async def test_export_session_to_graphviz(self):
        """Test exporting session thought graph to GraphViz DOT format."""
        session = Session()
        session.start()

        # Create a graph
        session.add_thought(
            ThoughtNode(
                id="a",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Node A",
            )
        )
        session.add_thought(
            ThoughtNode(
                id="b",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Node B",
                parent_id="a",
                depth=1,
            )
        )

        # Generate GraphViz DOT
        dot = self._generate_graphviz(session)

        # Verify DOT format
        assert "digraph" in dot or "graph" in dot
        assert "a" in dot
        assert "b" in dot
        assert "->" in dot

    def _generate_mermaid(self, session: Session) -> str:
        """Helper to generate Mermaid diagram from session."""
        lines = ["graph TD"]

        for node_id, node in session.graph.nodes.items():
            # Add node
            label = node.content[:20] + "..." if len(node.content) > 20 else node.content
            lines.append(f'    {node_id}["{label}"]')

        for edge_id, edge in session.graph.edges.items():
            # Add edge
            lines.append(f"    {edge.source_id} --> {edge.target_id}")

        return "\n".join(lines)

    def _generate_graphviz(self, session: Session) -> str:
        """Helper to generate GraphViz DOT from session."""
        lines = ["digraph G {"]

        for node_id, node in session.graph.nodes.items():
            label = node.content[:20] + "..." if len(node.content) > 20 else node.content
            lines.append(f'    {node_id} [label="{label}"];')

        for edge_id, edge in session.graph.edges.items():
            lines.append(f"    {edge.source_id} -> {edge.target_id};")

        lines.append("}")
        return "\n".join(lines)


class TestSessionConcurrency:
    """Test session concurrency and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self):
        """Test creating multiple sessions concurrently."""
        manager = SessionManager(max_sessions=20)

        # Create 10 sessions concurrently
        tasks = [manager.create() for _ in range(10)]
        sessions = await asyncio.gather(*tasks)

        # Verify all created
        assert len(sessions) == 10
        assert await manager.count() == 10

        # Verify unique IDs
        session_ids = {s.id for s in sessions}
        assert len(session_ids) == 10

    @pytest.mark.asyncio
    async def test_concurrent_operations_on_same_session(self):
        """Test multiple concurrent operations on the same session."""
        manager = SessionManager()
        session = await manager.create()

        # Define concurrent operations
        async def start_session():
            s = await manager.get(session.id)
            s.start()
            await manager.update(session.id, s)

        async def add_thought(thought_id: str):
            s = await manager.get(session.id)
            s.add_thought(
                ThoughtNode(
                    id=thought_id,
                    type=ThoughtType.CONTINUATION,
                    method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                    content=f"Thought {thought_id}",
                )
            )
            await manager.update(session.id, s)

        # Start session first
        await start_session()
        await asyncio.sleep(0.01)

        # Add thoughts concurrently
        thought_tasks = [add_thought(f"t{i}") for i in range(5)]
        await asyncio.gather(*thought_tasks)

        # Verify final state
        final_session = await manager.get(session.id)
        assert final_session.status == SessionStatus.ACTIVE
        assert final_session.thought_count >= 1  # At least some thoughts added

    @pytest.mark.asyncio
    async def test_concurrent_read_operations(self):
        """Test concurrent read operations are safe."""
        manager = SessionManager()

        # Create session with thoughts
        session = await manager.create()
        session.start()
        for i in range(10):
            session.add_thought(
                ThoughtNode(
                    id=f"t{i}",
                    type=ThoughtType.CONTINUATION,
                    method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                    content=f"Thought {i}",
                    depth=i,
                )
            )
        await manager.update(session.id, session)

        # Concurrent reads
        async def read_session():
            s = await manager.get(session.id)
            return s.thought_count

        tasks = [read_session() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All reads should return same value
        assert all(count == 10 for count in results)

    @pytest.mark.asyncio
    async def test_concurrent_manager_operations(self):
        """Test concurrent manager CRUD operations."""
        manager = SessionManager(max_sessions=50)

        # Mix of operations
        async def create_op():
            return await manager.create()

        async def list_op():
            return await manager.list_sessions()

        async def count_op():
            return await manager.count()

        # Create some initial sessions
        initial = [await manager.create() for _ in range(5)]

        # Run mixed operations concurrently
        tasks = (
            [create_op() for _ in range(10)]
            + [list_op() for _ in range(5)]
            + [count_op() for _ in range(5)]
        )

        results = await asyncio.gather(*tasks)

        # Verify counts make sense
        final_count = await manager.count()
        assert final_count == 15  # 5 initial + 10 created


class TestSessionEdgeCases:
    """Test session edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_session_export(self):
        """Test exporting a session with no thoughts."""
        session = Session()
        json_data = session.model_dump()

        assert json_data["id"] == session.id
        assert len(json_data["graph"]["nodes"]) == 0
        assert len(json_data["graph"]["edges"]) == 0

    @pytest.mark.asyncio
    async def test_session_with_orphaned_thoughts(self):
        """Test session with thoughts that have non-existent parents."""
        session = Session().start()

        # Add thought with non-existent parent
        orphan = ThoughtNode(
            id="orphan",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Orphaned thought",
            parent_id="nonexistent",
            depth=1,
        )

        session.add_thought(orphan)

        # Thought should be added but no edge created
        assert session.thought_count == 1
        assert session.graph.edge_count == 0

    @pytest.mark.asyncio
    async def test_session_with_zero_confidence_thoughts(self):
        """Test session metrics with zero confidence thoughts."""
        session = Session().start()

        session.add_thought(
            ThoughtNode(
                id="t1",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="No confidence",
                confidence=0.0,
            )
        )
        session.add_thought(
            ThoughtNode(
                id="t2",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Some confidence",
                confidence=0.5,
            )
        )

        # Average should be 0.25
        assert abs(session.metrics.average_confidence - 0.25) < 0.001

    @pytest.mark.asyncio
    async def test_session_state_persistence_across_updates(self):
        """Test that session state persists across manager updates."""
        manager = SessionManager()
        session = await manager.create()

        # Modify session
        session.start()
        session.add_thought(
            ThoughtNode(
                id="t1",
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Test",
            )
        )

        # Update in manager
        await manager.update(session.id, session)

        # Retrieve and verify
        retrieved = await manager.get(session.id)
        assert retrieved.status == SessionStatus.ACTIVE
        assert retrieved.thought_count == 1

        # Modify again
        retrieved.complete()
        await manager.update(session.id, retrieved)

        # Verify final state
        final = await manager.get(session.id)
        assert final.status == SessionStatus.COMPLETED
        assert final.thought_count == 1
