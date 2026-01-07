"""
Comprehensive tests for trace resource in reasoning_mcp.resources.trace.

This module provides complete test coverage for trace resources:
- trace://{session_id} - Returns execution trace with thought sequence, branches, conclusions

Each resource is tested for:
1. Basic functionality and structure
2. Return type correctness
3. Trace structure validation
4. Thought sequence ordering
5. Branching information accuracy
6. Conclusions detection logic
7. Not-found handling
8. Metadata and metrics inclusion
"""

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import Session, SessionConfig, SessionMetrics
from reasoning_mcp.models.thought import ThoughtGraph, ThoughtNode


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_session() -> Session:
    """Create a sample session with thoughts, branches, and conclusions."""
    config = SessionConfig(
        max_depth=10,
        max_thoughts=100,
        timeout_seconds=300.0,
        enable_branching=True,
        max_branches=5,
        auto_prune=False,
        min_confidence_threshold=0.3,
    )

    metrics = SessionMetrics(
        total_thoughts=5,
        total_edges=4,
        max_depth_reached=3,
        average_confidence=0.8,
        average_quality=0.85,
        methods_used={MethodIdentifier.CHAIN_OF_THOUGHT: 5},
        thought_types={
            ThoughtType.INITIAL: 1,
            ThoughtType.CONTINUATION: 3,
            ThoughtType.CONCLUSION: 1,
        },
        branches_created=2,
        branches_merged=1,
        branches_pruned=0,
        elapsed_time=120.5,
    )

    session = Session(
        id="test-session-123",
        config=config,
        metrics=metrics,
        graph=ThoughtGraph(),
        status=SessionStatus.COMPLETED,
        current_method=MethodIdentifier.CHAIN_OF_THOUGHT,
        started_at=datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 6, 12, 2, 0, tzinfo=timezone.utc),
    )

    # Add thoughts to the graph
    thought1 = ThoughtNode(
        id="thought-1",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Let's analyze the problem step by step",
        summary="Initial problem analysis",
        confidence=0.7,
        quality_score=0.75,
        depth=0,
        step_number=1,
        created_at=datetime(2024, 1, 6, 12, 0, 1, tzinfo=timezone.utc),
    )
    session.graph.add_thought(thought1)

    thought2 = ThoughtNode(
        id="thought-2",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Breaking down into components A and B",
        summary="Component breakdown",
        confidence=0.8,
        quality_score=0.85,
        depth=1,
        step_number=2,
        parent_id="thought-1",
        created_at=datetime(2024, 1, 6, 12, 0, 30, tzinfo=timezone.utc),
    )
    session.graph.add_thought(thought2)

    thought3 = ThoughtNode(
        id="thought-3",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Analyzing component A deeply",
        summary="Deep dive into A",
        confidence=0.85,
        quality_score=0.9,
        depth=2,
        step_number=3,
        parent_id="thought-2",
        branch_id="branch-main",
        created_at=datetime(2024, 1, 6, 12, 1, 0, tzinfo=timezone.utc),
    )
    session.graph.add_thought(thought3)

    thought4 = ThoughtNode(
        id="thought-4",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Exploring alternative approach for B",
        summary="Alternative for B",
        confidence=0.75,
        quality_score=0.8,
        depth=2,
        step_number=4,
        parent_id="thought-2",
        branch_id="branch-alternative",
        created_at=datetime(2024, 1, 6, 12, 1, 15, tzinfo=timezone.utc),
    )
    session.graph.add_thought(thought4)

    thought5 = ThoughtNode(
        id="thought-5",
        type=ThoughtType.CONCLUSION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Final synthesis combining both approaches",
        summary="Final conclusion",
        confidence=0.95,
        quality_score=0.95,
        depth=3,
        step_number=5,
        parent_id="thought-3",
        created_at=datetime(2024, 1, 6, 12, 1, 45, tzinfo=timezone.utc),
    )
    session.graph.add_thought(thought5)

    return session


@pytest.fixture
def empty_session() -> Session:
    """Create an empty session with no thoughts."""
    config = SessionConfig()
    metrics = SessionMetrics()

    session = Session(
        id="empty-session",
        config=config,
        metrics=metrics,
        graph=ThoughtGraph(),
        status=SessionStatus.ACTIVE,
        current_method=MethodIdentifier.CHAIN_OF_THOUGHT,
    )

    return session


# ============================================================================
# Mock AppContext
# ============================================================================


class MockSessionManager:
    """Mock session manager for testing."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def add_session(self, session: Session) -> None:
        """Add a session to the manager."""
        self.sessions[session.id] = session


class MockAppContext:
    """Mock app context for testing."""

    def __init__(self, session_manager: MockSessionManager) -> None:
        self.session_manager = session_manager


class MockFastMCP:
    """Mock FastMCP server for testing."""

    def __init__(self, app_context: MockAppContext) -> None:
        self.app_context = app_context
        self.resources: dict[str, Any] = {}

    def resource(self, uri_pattern: str):
        """Decorator to register a resource."""

        def decorator(func):
            self.resources[uri_pattern] = func
            return func

        return decorator


# ============================================================================
# Test trace://{session_id} Resource
# ============================================================================


class TestTraceResource:
    """Test suite for trace://{session_id} resource."""

    @pytest.mark.asyncio
    async def test_trace_basic_functionality(self, sample_session):
        """Test basic trace resource functionality."""
        # Setup
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))

        # Register resources
        register_trace_resources(mcp)

        # Get the resource function
        resource_func = mcp.resources["trace://{session_id}"]
        assert resource_func is not None

        # Call the resource
        result = await resource_func(session_id=sample_session.id)

        # Verify result is JSON string
        assert isinstance(result, str)

        # Parse JSON
        trace = json.loads(result)
        assert isinstance(trace, dict)

        # Verify top-level structure
        assert trace["session_id"] == sample_session.id
        assert "metadata" in trace
        assert "config" in trace
        assert "metrics" in trace
        assert "thoughts" in trace
        assert "branches" in trace
        assert "conclusions" in trace

    @pytest.mark.asyncio
    async def test_trace_metadata_structure(self, sample_session):
        """Test that trace metadata contains all required fields."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        metadata = trace["metadata"]
        assert metadata["method"] == "chain_of_thought"
        assert metadata["status"] == "completed"
        assert metadata["created_at"] is not None
        assert metadata["started_at"] == "2024-01-06T12:00:00+00:00"
        assert metadata["completed_at"] == "2024-01-06T12:02:00+00:00"
        assert metadata["duration_seconds"] == 120.0
        assert metadata["error"] is None

    @pytest.mark.asyncio
    async def test_trace_config_structure(self, sample_session):
        """Test that trace config contains all configuration fields."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        config = trace["config"]
        assert config["max_depth"] == 10
        assert config["max_thoughts"] == 100
        assert config["timeout_seconds"] == 300.0
        assert config["enable_branching"] is True
        assert config["max_branches"] == 5
        assert config["auto_prune"] is False
        assert config["min_confidence_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_trace_metrics_structure(self, sample_session):
        """Test that trace metrics contains all metrics fields."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        metrics = trace["metrics"]
        assert metrics["total_thoughts"] == 5
        assert metrics["total_edges"] == 4
        assert metrics["max_depth_reached"] == 3
        assert metrics["average_confidence"] == 0.8
        assert metrics["average_quality"] == 0.85
        assert metrics["methods_used"] == {"chain_of_thought": 5}
        assert metrics["thought_types"] == {"initial": 1, "continuation": 3, "conclusion": 1}
        assert metrics["elapsed_time"] == 120.5

    @pytest.mark.asyncio
    async def test_trace_thoughts_chronological_order(self, sample_session):
        """Test that thoughts are returned in chronological order."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        thoughts = trace["thoughts"]
        assert len(thoughts) == 5

        # Verify chronological ordering by created_at timestamps
        timestamps = [t["created_at"] for t in thoughts]
        assert timestamps == sorted(timestamps)

        # Verify thought_number is sequential
        thought_numbers = [t["thought_number"] for t in thoughts]
        assert thought_numbers == [1, 2, 3, 4, 5]

        # Verify first thought is the initial one
        assert thoughts[0]["id"] == "thought-1"
        assert thoughts[0]["type"] == "initial"

    @pytest.mark.asyncio
    async def test_trace_thought_structure(self, sample_session):
        """Test that each thought contains all required fields."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        thought = trace["thoughts"][0]
        assert "thought_number" in thought
        assert "id" in thought
        assert "type" in thought
        assert "method_id" in thought
        assert "content" in thought
        assert "summary" in thought
        assert "evidence" in thought
        assert "confidence" in thought
        assert "quality_score" in thought
        assert "is_valid" in thought
        assert "depth" in thought
        assert "step_number" in thought
        assert "parent_id" in thought
        assert "children_ids" in thought
        assert "branch_id" in thought
        assert "created_at" in thought

    @pytest.mark.asyncio
    async def test_trace_branches_structure(self, sample_session):
        """Test that branches section contains correct information."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        branches = trace["branches"]
        assert branches["total_created"] == 2
        assert branches["total_merged"] == 1
        assert branches["total_pruned"] == 0
        assert isinstance(branches["branch_points"], list)

        # Verify branch points structure
        if branches["branch_points"]:
            branch_point = branches["branch_points"][0]
            assert "branch_id" in branch_point
            assert "thought_id" in branch_point
            assert "parent_thought_id" in branch_point
            assert "created_at" in branch_point

    @pytest.mark.asyncio
    async def test_trace_conclusions_detection_high_confidence(self, sample_session):
        """Test that conclusions include thoughts with confidence >= 0.7."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        conclusions = trace["conclusions"]
        assert len(conclusions) > 0

        # Verify conclusion structure
        for conclusion in conclusions:
            assert "thought_id" in conclusion
            assert "content" in conclusion
            assert "confidence" in conclusion
            assert "quality_score" in conclusion
            assert "branch_id" in conclusion

        # The high confidence conclusion should be included
        thought_ids = [c["thought_id"] for c in conclusions]
        assert "thought-5" in thought_ids

        # Find the high confidence conclusion
        high_conf_conclusion = next(c for c in conclusions if c["thought_id"] == "thought-5")
        assert high_conf_conclusion["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_trace_conclusions_include_explicit_type(self, sample_session):
        """Test that conclusions include thoughts with conclusion type."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        conclusions = trace["conclusions"]

        # Find the explicit conclusion type thought
        conclusion_thoughts = [c for c in conclusions if c["thought_id"] == "thought-5"]
        assert len(conclusion_thoughts) == 1

    @pytest.mark.asyncio
    async def test_trace_session_not_found(self):
        """Test that trace resource raises ValueError for non-existent session."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]

        with pytest.raises(ValueError) as exc_info:
            await resource_func(session_id="non-existent-session")

        assert "Session not found" in str(exc_info.value)
        assert "non-existent-session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_trace_empty_session(self, empty_session):
        """Test trace resource with empty session (no thoughts)."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(empty_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=empty_session.id)
        trace = json.loads(result)

        # Verify empty collections
        assert trace["thoughts"] == []
        assert trace["branches"]["branch_points"] == []
        assert trace["conclusions"] == []

        # Verify metrics are all zero
        assert trace["metrics"]["total_thoughts"] == 0
        assert trace["metrics"]["total_edges"] == 0

    @pytest.mark.asyncio
    async def test_trace_json_formatting(self, sample_session):
        """Test that trace JSON is properly formatted."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)

        # Verify it's valid JSON
        trace = json.loads(result)
        assert isinstance(trace, dict)

        # Verify it's pretty-printed (has indentation)
        assert "\n" in result
        assert "  " in result

    @pytest.mark.asyncio
    async def test_trace_all_leaf_nodes_as_fallback(self):
        """Test that all leaf nodes are included if no high-confidence conclusions exist."""
        from reasoning_mcp.resources.trace import register_trace_resources

        # Create session with low-confidence leaf nodes
        config = SessionConfig()
        metrics = SessionMetrics(total_thoughts=2, total_edges=1)
        session = Session(
            id="low-confidence-session",
            config=config,
            metrics=metrics,
            graph=ThoughtGraph(),
            status=SessionStatus.COMPLETED,
            current_method=MethodIdentifier.CHAIN_OF_THOUGHT,
        )

        thought1 = ThoughtNode(
            id="thought-1",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Initial thought",
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            step_number=1,
        )
        session.graph.add_thought(thought1)

        thought2 = ThoughtNode(
            id="thought-2",
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Low confidence leaf",
            confidence=0.4,  # Below 0.7 threshold
            quality_score=0.4,
            depth=1,
            step_number=2,
            parent_id="thought-1",
        )
        session.graph.add_thought(thought2)

        session_manager = MockSessionManager()
        session_manager.add_session(session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=session.id)
        trace = json.loads(result)

        # Should include the low-confidence leaf as fallback
        conclusions = trace["conclusions"]
        assert len(conclusions) == 1
        assert conclusions[0]["thought_id"] == "thought-2"
        assert conclusions[0]["confidence"] == 0.4

    @pytest.mark.asyncio
    async def test_trace_branch_identification(self, sample_session):
        """Test that branch points are correctly identified."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))
        register_trace_resources(mcp)

        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        branch_points = trace["branches"]["branch_points"]

        # Verify we have branch points for our two branches
        branch_ids = [bp["branch_id"] for bp in branch_points]
        assert "branch-main" in branch_ids
        assert "branch-alternative" in branch_ids

        # Verify branch point structure
        for bp in branch_points:
            assert bp["thought_id"] in ["thought-3", "thought-4"]
            assert bp["parent_thought_id"] == "thought-2"
            assert "created_at" in bp


# ============================================================================
# Integration Tests
# ============================================================================


class TestTraceResourceIntegration:
    """Integration tests for trace resource."""

    @pytest.mark.asyncio
    async def test_trace_resource_registration(self):
        """Test that trace resource is properly registered."""
        from reasoning_mcp.resources.trace import register_trace_resources

        session_manager = MockSessionManager()
        mcp = MockFastMCP(MockAppContext(session_manager))

        # Before registration
        assert "trace://{session_id}" not in mcp.resources

        # Register
        register_trace_resources(mcp)

        # After registration
        assert "trace://{session_id}" in mcp.resources
        assert callable(mcp.resources["trace://{session_id}"])

    @pytest.mark.asyncio
    async def test_trace_full_workflow(self, sample_session):
        """Test complete trace workflow from registration to retrieval."""
        from reasoning_mcp.resources.trace import register_trace_resources

        # Setup
        session_manager = MockSessionManager()
        session_manager.add_session(sample_session)
        mcp = MockFastMCP(MockAppContext(session_manager))

        # Register resources
        register_trace_resources(mcp)

        # Retrieve trace
        resource_func = mcp.resources["trace://{session_id}"]
        result = await resource_func(session_id=sample_session.id)
        trace = json.loads(result)

        # Verify complete structure
        assert trace["session_id"] == sample_session.id
        assert len(trace["thoughts"]) == 5
        assert len(trace["conclusions"]) > 0
        assert trace["metadata"]["status"] == "completed"
        assert trace["metrics"]["total_thoughts"] == 5

        # Verify data consistency
        total_thoughts_in_trace = len(trace["thoughts"])
        total_thoughts_in_metrics = trace["metrics"]["total_thoughts"]
        assert total_thoughts_in_trace == total_thoughts_in_metrics
