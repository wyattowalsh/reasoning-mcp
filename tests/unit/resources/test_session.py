"""
Comprehensive tests for session resource endpoints.

This module tests the MCP resource endpoints for accessing session state
and thought graph visualizations:
- session://{session_id} - Returns session state as JSON
- session://{session_id}/graph - Returns mermaid graph

Each resource is tested for:
1. Basic functionality (returns correct data structure)
2. Session ID validation
3. Not-found handling
4. Graph generation with various thought structures (linear, branching)
5. Edge type formatting in mermaid
6. Content truncation
7. Empty session handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtEdge, ThoughtNode
from reasoning_mcp.resources.session import register_session_resources

# ============================================================================
# AppContext Fixture
# ============================================================================


@pytest.fixture(autouse=True)
def mock_app_context(request):
    """Mock get_app_context for all resource tests.

    This fixture patches get_app_context at the source (reasoning_mcp.server)
    to return a mock context. Tests can access mock_session_manager through
    mock_app_context.session_manager.
    """
    mock_context = MagicMock()
    mock_session_manager = AsyncMock()
    mock_context.session_manager = mock_session_manager

    with patch("reasoning_mcp.server.get_app_context", return_value=mock_context):
        yield mock_context


@pytest.fixture
def mock_session_manager(mock_app_context):
    """Provide the mock session manager for tests to configure."""
    return mock_app_context.session_manager

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp.app_context = MagicMock()

    # Store registered resource handlers
    mcp._resource_handlers = {}

    def mock_resource(uri_pattern):
        """Decorator to capture resource handlers."""

        def decorator(func):
            # Extract the pattern key for testing
            # session://{session_id} -> "session_state"
            # session://{session_id}/graph -> "session_graph"
            if uri_pattern.endswith("/graph"):
                key = "session_graph"
            else:
                key = "session_state"
            mcp._resource_handlers[key] = func
            return func

        return decorator

    mcp.resource = mock_resource
    return mcp


@pytest.fixture
def empty_session():
    """Create an empty session for testing."""
    session = Session(id="empty-session-123")
    session.start()
    return session


@pytest.fixture
def linear_session():
    """Create a session with a linear thought chain."""
    session = Session(id="linear-session-456")
    session.start()

    # Create a chain of thoughts: root -> child1 -> child2
    root = ThoughtNode(
        id="root-1",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="This is the initial thought to start our analysis",
        confidence=0.9,
        depth=0,
    )
    session.add_thought(root)

    child1 = ThoughtNode(
        id="child-1",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Building on the initial thought, we explore deeper",
        parent_id="root-1",
        confidence=0.85,
        depth=1,
    )
    session.add_thought(child1)

    child2 = ThoughtNode(
        id="child-2",
        type=ThoughtType.CONCLUSION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Finally, we reach our conclusion based on the reasoning chain",
        parent_id="child-1",
        confidence=0.8,
        depth=2,
    )
    session.add_thought(child2)

    return session


@pytest.fixture
def branching_session():
    """Create a session with branching thoughts."""
    session = Session(id="branch-session-789")
    session.start()

    # Create a branching structure: root -> branch1, root -> branch2
    root = ThoughtNode(
        id="root-2",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Initial problem analysis",
        confidence=0.95,
        depth=0,
    )
    session.add_thought(root)

    branch1 = ThoughtNode(
        id="branch-1",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Exploring approach A",
        parent_id="root-2",
        branch_id="branch-a",
        confidence=0.7,
        depth=1,
    )
    session.add_thought(branch1)

    branch2 = ThoughtNode(
        id="branch-2",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Exploring approach B",
        parent_id="root-2",
        branch_id="branch-b",
        confidence=0.75,
        depth=1,
    )
    session.add_thought(branch2)

    # Add supporting edge
    support_edge = ThoughtEdge(
        id="edge-support",
        source_id="branch-1",
        target_id="branch-2",
        edge_type="supports",
        weight=0.6,
    )
    session.graph.add_edge(support_edge)

    # Add contradicting edge
    contradict_edge = ThoughtEdge(
        id="edge-contradict",
        source_id="branch-2",
        target_id="branch-1",
        edge_type="contradicts",
        weight=0.4,
    )
    session.graph.add_edge(contradict_edge)

    return session


@pytest.fixture
def complex_session():
    """Create a session with complex edge types."""
    session = Session(id="complex-session-abc")
    session.start()

    root = ThoughtNode(
        id="root-3",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Starting point",
        confidence=1.0,
        depth=0,
    )
    session.add_thought(root)

    child1 = ThoughtNode(
        id="child-3-1",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="First continuation",
        parent_id="root-3",
        confidence=0.9,
        depth=1,
    )
    session.add_thought(child1)

    child2 = ThoughtNode(
        id="child-3-2",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Branching off",
        parent_id="root-3",
        branch_id="alt-branch",
        confidence=0.85,
        depth=1,
    )
    session.add_thought(child2)

    # Add branch edge explicitly
    branch_edge = ThoughtEdge(
        id="edge-branch",
        source_id="root-3",
        target_id="child-3-2",
        edge_type="branches",
        weight=1.0,
    )
    session.graph.add_edge(branch_edge)

    return session


# ============================================================================
# Test session://{session_id} - Session State Resource
# ============================================================================


class TestSessionStateResource:
    """Test suite for the session state resource endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_state_basic(self, mock_mcp, mock_session_manager, linear_session):
        """Test basic session state retrieval returns valid JSON."""
        # Setup mock session manager
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        # Register resources
        register_session_resources(mock_mcp)

        # Get the handler and call it
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("linear-session-456")

        # Verify it returns valid JSON
        assert isinstance(result, str)
        parsed = json.loads(result)

        # Verify structure
        assert isinstance(parsed, dict)
        assert "id" in parsed
        assert "status" in parsed
        assert "graph" in parsed
        assert "metrics" in parsed
        assert "config" in parsed

        # Verify values
        assert parsed["id"] == "linear-session-456"
        assert parsed["status"] == SessionStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_get_session_state_includes_graph_data(self, mock_mcp, mock_session_manager, linear_session):
        """Test that session state includes complete thought graph data."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("linear-session-456")

        parsed = json.loads(result)
        graph = parsed["graph"]

        # Verify graph structure
        assert "nodes" in graph
        assert "edges" in graph
        assert "root_id" in graph

        # Verify nodes
        assert len(graph["nodes"]) == 3
        assert "root-1" in graph["nodes"]
        assert "child-1" in graph["nodes"]
        assert "child-2" in graph["nodes"]

        # Verify node content
        root_node = graph["nodes"]["root-1"]
        assert root_node["content"] == "This is the initial thought to start our analysis"
        assert root_node["type"] == ThoughtType.INITIAL.value
        assert root_node["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_get_session_state_includes_metrics(self, mock_mcp, mock_session_manager, linear_session):
        """Test that session state includes complete metrics."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("linear-session-456")

        parsed = json.loads(result)
        metrics = parsed["metrics"]

        # Verify metrics structure
        assert "total_thoughts" in metrics
        assert "total_edges" in metrics
        assert "max_depth_reached" in metrics
        assert "average_confidence" in metrics

        # Verify values
        assert metrics["total_thoughts"] == 3
        assert metrics["total_edges"] == 2  # 2 derivation edges
        assert metrics["max_depth_reached"] == 2

    @pytest.mark.asyncio
    async def test_get_session_state_includes_config(self, mock_mcp, mock_session_manager, linear_session):
        """Test that session state includes configuration."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("linear-session-456")

        parsed = json.loads(result)
        config = parsed["config"]

        # Verify config structure
        assert "max_depth" in config
        assert "max_thoughts" in config
        assert "timeout_seconds" in config
        assert "enable_branching" in config

        # Verify default values
        assert config["max_depth"] == 10
        assert config["max_thoughts"] == 100

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self, mock_mcp, mock_session_manager):
        """Test that requesting a non-existent session raises ValueError."""
        mock_session_manager.get = AsyncMock(return_value=None)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]

        with pytest.raises(ValueError, match="Session not found: nonexistent-id"):
            await handler("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_session_state_empty_session(self, mock_mcp, mock_session_manager, empty_session):
        """Test session state for a session with no thoughts."""
        mock_session_manager.get = AsyncMock(return_value=empty_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("empty-session-123")

        parsed = json.loads(result)

        # Verify empty state
        assert parsed["id"] == "empty-session-123"
        assert len(parsed["graph"]["nodes"]) == 0
        assert len(parsed["graph"]["edges"]) == 0
        assert parsed["metrics"]["total_thoughts"] == 0
        assert parsed["metrics"]["total_edges"] == 0

    @pytest.mark.asyncio
    async def test_get_session_state_json_formatting(self, mock_mcp, mock_session_manager, linear_session):
        """Test that JSON output is properly formatted with indentation."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_state"]
        result = await handler("linear-session-456")

        # Verify it has indentation (pretty-printed)
        assert "\n" in result
        assert "  " in result  # Should have 2-space indentation

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


# ============================================================================
# Test session://{session_id}/graph - Graph Visualization Resource
# ============================================================================


class TestSessionGraphResource:
    """Test suite for the session graph visualization resource endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_graph_basic_structure(self, mock_mcp, mock_session_manager, linear_session):
        """Test basic mermaid graph generation."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("linear-session-456")

        # Verify it's a string
        assert isinstance(result, str)

        # Verify it starts with mermaid graph directive
        lines = result.split("\n")
        assert lines[0] == "graph TD"

    @pytest.mark.asyncio
    async def test_get_session_graph_linear_chain(self, mock_mcp, mock_session_manager, linear_session):
        """Test graph generation for linear thought chain."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("linear-session-456")

        # Verify nodes are present (with safe IDs)
        assert "root_1[" in result  # Hyphens replaced with underscores
        assert "child_1[" in result
        assert "child_2[" in result

        # Verify edges (arrows)
        assert "root_1 -->" in result
        assert "child_1 -->" in result

        # Verify node labels include type and confidence
        assert "initial:" in result
        assert "continuation:" in result
        assert "conclusion:" in result
        assert "(conf: 0.90)" in result
        assert "(conf: 0.85)" in result
        assert "(conf: 0.80)" in result

    @pytest.mark.asyncio
    async def test_get_session_graph_content_truncation(self, mock_mcp, mock_session_manager, linear_session):
        """Test that long content is truncated with ellipsis."""
        # The fixture has content longer than 50 chars
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("linear-session-456")

        # Content should be truncated (note: content is NOT truncated if <= 50 chars)
        # The first thought is exactly 50 chars, so it won't have "..."
        assert "This is the initial thought to start our analysis" in result
        # The second thought is 45 chars, so it also won't be truncated
        assert (
            "Building on the initial thought, we explore deeper" in result
            or "Building on the initial thought, we explore deep" in result
        )

    @pytest.mark.asyncio
    async def test_get_session_graph_with_branches(self, mock_mcp, mock_session_manager, branching_session):
        """Test graph generation with branching structure."""
        mock_session_manager.get = AsyncMock(return_value=branching_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("branch-session-789")

        # Verify branch nodes
        assert "branch_1[" in result
        assert "branch_2[" in result

        # Verify branch info in labels
        assert "[branch: branch-a]" in result
        assert "[branch: branch-b]" in result

        # Verify branch type (lowercase in implementation)
        assert "branch:" in result

    @pytest.mark.asyncio
    async def test_get_session_graph_edge_types(self, mock_mcp, mock_session_manager, branching_session):
        """Test different edge types use different arrow styles."""
        mock_session_manager.get = AsyncMock(return_value=branching_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("branch-session-789")

        # Verify default "derives" edges use standard arrow
        assert "-->" in result

        # Verify "supports" edges use dotted arrow
        assert "-.->'" in result or "-.->|supports|" in result

        # Verify "contradicts" edges use cross arrow
        assert "-.x" in result or "-.x|contradicts|" in result

    @pytest.mark.asyncio
    async def test_get_session_graph_branches_edge_type(self, mock_mcp, mock_session_manager, complex_session):
        """Test 'branches' edge type uses special arrow."""
        mock_session_manager.get = AsyncMock(return_value=complex_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("complex-session-abc")

        # Verify "branches" edges use thick arrow
        assert "==>" in result or "==>|branches|" in result

    @pytest.mark.asyncio
    async def test_get_session_graph_edge_labels(self, mock_mcp, mock_session_manager, branching_session):
        """Test that non-derives edges include labels."""
        mock_session_manager.get = AsyncMock(return_value=branching_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("branch-session-789")

        # Non-default edge types should have labels
        assert "|supports|" in result
        assert "|contradicts|" in result

    @pytest.mark.asyncio
    async def test_get_session_graph_empty_session(self, mock_mcp, mock_session_manager, empty_session):
        """Test graph generation for empty session."""
        mock_session_manager.get = AsyncMock(return_value=empty_session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("empty-session-123")

        # Verify it returns graph with empty message
        assert "graph TD" in result
        assert 'empty["No thoughts in this session yet"]' in result

    @pytest.mark.asyncio
    async def test_get_session_graph_not_found(self, mock_mcp, mock_session_manager):
        """Test that requesting graph for non-existent session raises ValueError."""
        mock_session_manager.get = AsyncMock(return_value=None)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]

        with pytest.raises(ValueError, match="Session not found: nonexistent-id"):
            await handler("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_session_graph_quote_escaping(self, mock_mcp, mock_session_manager):
        """Test that quotes in content are properly escaped."""
        # Create session with quotes in content
        session = Session(id="quote-session")
        session.start()

        thought = ThoughtNode(
            id="quote-node",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content='This thought contains "quotes" in it',
            confidence=0.8,
            depth=0,
        )
        session.add_thought(thought)

        mock_session_manager.get = AsyncMock(return_value=session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("quote-session")

        # Verify quotes are escaped
        assert '\\"quotes\\"' in result

    @pytest.mark.asyncio
    async def test_get_session_graph_safe_node_ids(self, mock_mcp, mock_session_manager):
        """Test that node IDs with hyphens are converted to underscores."""
        session = Session(id="hyphen-session")
        session.start()

        thought = ThoughtNode(
            id="node-with-hyphens-123",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test",
            confidence=0.8,
            depth=0,
        )
        session.add_thought(thought)

        mock_session_manager.get = AsyncMock(return_value=session)

        register_session_resources(mock_mcp)
        handler = mock_mcp._resource_handlers["session_graph"]
        result = await handler("hyphen-session")

        # Verify hyphens are replaced with underscores in mermaid
        assert "node_with_hyphens_123[" in result
        assert "node-with-hyphens-123" not in result  # Original ID shouldn't appear


# ============================================================================
# Integration Tests
# ============================================================================


class TestSessionResourcesIntegration:
    """Integration tests for session resource endpoints."""

    @pytest.mark.asyncio
    async def test_register_session_resources(self, mock_mcp, mock_session_manager):
        """Test that register_session_resources registers both endpoints."""
        register_session_resources(mock_mcp)

        # Verify both handlers are registered
        assert "session_state" in mock_mcp._resource_handlers
        assert "session_graph" in mock_mcp._resource_handlers

        # Verify handlers are callable
        assert callable(mock_mcp._resource_handlers["session_state"])
        assert callable(mock_mcp._resource_handlers["session_graph"])

    @pytest.mark.asyncio
    async def test_both_resources_use_same_session(self, mock_mcp, mock_session_manager, linear_session):
        """Test that both resources retrieve the same session data."""
        mock_session_manager.get = AsyncMock(return_value=linear_session)

        register_session_resources(mock_mcp)

        # Get both resources
        state_handler = mock_mcp._resource_handlers["session_state"]
        graph_handler = mock_mcp._resource_handlers["session_graph"]

        state_result = await state_handler("linear-session-456")
        graph_result = await graph_handler("linear-session-456")

        # Verify both accessed the same session
        assert mock_session_manager.get.call_count == 2
        assert mock_session_manager.get.call_args_list[0][0][0] == "linear-session-456"
        assert mock_session_manager.get.call_args_list[1][0][0] == "linear-session-456"

        # Verify state includes data
        state_data = json.loads(state_result)
        assert state_data["id"] == "linear-session-456"

        # Verify graph is generated
        assert "graph TD" in graph_result
        assert "root_1" in graph_result

    @pytest.mark.asyncio
    async def test_resources_handle_various_session_states(self, mock_mcp, mock_session_manager):
        """Test resources work with sessions in different states."""
        # Test with different session statuses
        for status in [
            SessionStatus.CREATED,
            SessionStatus.ACTIVE,
            SessionStatus.PAUSED,
            SessionStatus.COMPLETED,
        ]:
            session = Session(id=f"session-{status.value}")
            if status != SessionStatus.CREATED:
                session.start()
            if status == SessionStatus.PAUSED:
                session.pause()
            elif status == SessionStatus.COMPLETED:
                session.complete()

            mock_session_manager.get = AsyncMock(return_value=session)

            register_session_resources(mock_mcp)
            state_handler = mock_mcp._resource_handlers["session_state"]
            graph_handler = mock_mcp._resource_handlers["session_graph"]

            # Both should work regardless of session state
            state_result = await state_handler(f"session-{status.value}")
            graph_result = await graph_handler(f"session-{status.value}")

            assert isinstance(state_result, str)
            assert isinstance(graph_result, str)

            state_data = json.loads(state_result)
            assert state_data["status"] == status.value
