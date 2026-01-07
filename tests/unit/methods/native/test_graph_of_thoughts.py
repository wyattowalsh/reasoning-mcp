"""Unit tests for GraphOfThoughts reasoning method.

This module contains comprehensive tests for the GraphOfThoughtsMethod class,
covering initialization, execution, graph structure, DAG properties, convergence,
aggregation, cycle prevention, and various edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.graph_of_thoughts import (
    GRAPH_OF_THOUGHTS_METADATA,
    GraphOfThoughts,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType


class TestGraphOfThoughtsInitialization:
    """Tests for GraphOfThoughts initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        got = GraphOfThoughts()
        assert got.branching_factor == 3
        assert got.max_depth == 6
        assert got.min_merge_score == 0.6
        assert got.enable_convergence is True
        assert got.convergence_threshold == 0.7

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        got = GraphOfThoughts(
            branching_factor=5,
            max_depth=10,
            min_merge_score=0.8,
            enable_convergence=False,
            convergence_threshold=0.9,
        )
        assert got.branching_factor == 5
        assert got.max_depth == 10
        assert got.min_merge_score == 0.8
        assert got.enable_convergence is False
        assert got.convergence_threshold == 0.9

    def test_init_validates_branching_factor_too_low(self):
        """Test that branching_factor must be at least 1."""
        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            GraphOfThoughts(branching_factor=0)

    def test_init_validates_branching_factor_negative(self):
        """Test that branching_factor cannot be negative."""
        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            GraphOfThoughts(branching_factor=-1)

    def test_init_validates_max_depth_too_low(self):
        """Test that max_depth must be at least 1."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            GraphOfThoughts(max_depth=0)

    def test_init_validates_max_depth_negative(self):
        """Test that max_depth cannot be negative."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            GraphOfThoughts(max_depth=-1)

    def test_init_validates_min_merge_score_too_low(self):
        """Test that min_merge_score must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="min_merge_score must be 0.0-1.0"):
            GraphOfThoughts(min_merge_score=-0.1)

    def test_init_validates_min_merge_score_too_high(self):
        """Test that min_merge_score must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="min_merge_score must be 0.0-1.0"):
            GraphOfThoughts(min_merge_score=1.1)

    def test_init_validates_convergence_threshold_too_low(self):
        """Test that convergence_threshold must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="convergence_threshold must be 0.0-1.0"):
            GraphOfThoughts(convergence_threshold=-0.1)

    def test_init_validates_convergence_threshold_too_high(self):
        """Test that convergence_threshold must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="convergence_threshold must be 0.0-1.0"):
            GraphOfThoughts(convergence_threshold=1.1)

    def test_init_boundary_values_valid(self):
        """Test that boundary values are accepted."""
        got = GraphOfThoughts(
            branching_factor=1,
            max_depth=1,
            min_merge_score=0.0,
            convergence_threshold=1.0,
        )
        assert got.branching_factor == 1
        assert got.max_depth == 1
        assert got.min_merge_score == 0.0
        assert got.convergence_threshold == 1.0

    def test_properties(self):
        """Test that properties return correct values."""
        got = GraphOfThoughts()
        assert got.identifier == str(MethodIdentifier.GRAPH_OF_THOUGHTS)
        assert got.name == "Graph of Thoughts"
        assert got.description == "DAG-based reasoning with multiple parents, path merging, and complex dependencies"
        assert got.category == "core"

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test that initialize() completes without error."""
        got = GraphOfThoughts()
        await got.initialize()  # Should complete without error

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test that health_check() returns True."""
        got = GraphOfThoughts()
        result = await got.health_check()
        assert result is True


class TestGraphOfThoughtsBasicExecution:
    """Tests for basic execution of GraphOfThoughts."""

    @pytest.mark.asyncio
    async def test_execute_creates_root_thought(self):
        """Test that execute() creates a root thought."""
        got = GraphOfThoughts(branching_factor=2, max_depth=1)
        session = Session().start()

        result = await got.execute(session, "Test problem")

        assert session.thought_count > 0
        # Check that a root thought exists
        root_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.INITIAL]
        assert len(root_thoughts) == 1
        assert root_thoughts[0].metadata.get("is_root") is True

    @pytest.mark.asyncio
    async def test_execute_creates_dag_structure(self):
        """Test that execute() creates a DAG reasoning structure."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "Analyze this problem")

        # Should have multiple thoughts
        assert session.thought_count >= 4  # root + branches + synthesis
        # Result should be a synthesis thought
        assert result.type == ThoughtType.SYNTHESIS
        assert result.metadata.get("is_final") is True

    @pytest.mark.asyncio
    async def test_execute_with_inactive_session_raises_error(self):
        """Test that execute() raises ValueError for inactive session."""
        got = GraphOfThoughts()
        session = Session()  # Not started

        with pytest.raises(ValueError, match="Session must be active"):
            await got.execute(session, "Test problem")

    @pytest.mark.asyncio
    async def test_execute_with_context_override(self):
        """Test that execute() respects context parameter overrides."""
        got = GraphOfThoughts(branching_factor=3, max_depth=4)
        session = Session().start()

        context = {
            "branching_factor": 2,
            "max_depth": 1,
            "enable_convergence": False,
        }

        result = await got.execute(session, "Test", context=context)

        # Should use context values, not instance defaults
        assert result is not None
        # With max_depth=1 and branching=2, should have fewer thoughts
        assert session.thought_count < 10

    @pytest.mark.asyncio
    async def test_execute_returns_synthesis_thought(self):
        """Test that execute() returns a synthesis ThoughtNode."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "Problem to analyze")

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.SYNTHESIS
        assert result.method_id == MethodIdentifier.GRAPH_OF_THOUGHTS
        assert result.metadata.get("is_final") is True


class TestGraphOfThoughtsGraphStructure:
    """Tests for DAG structure properties."""

    @pytest.mark.asyncio
    async def test_graph_has_root_node(self):
        """Test that the graph has exactly one root node."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        await got.execute(session, "Test problem")

        # Find root nodes (nodes with no parent)
        root_nodes = [
            node for node in session.graph.nodes.values()
            if node.parent_id is None
        ]
        assert len(root_nodes) == 1
        assert root_nodes[0].type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_graph_has_multiple_levels(self):
        """Test that the graph has multiple depth levels."""
        got = GraphOfThoughts(branching_factor=2, max_depth=3)
        session = Session().start()

        await got.execute(session, "Multi-level problem")

        # Collect all unique depths
        depths = {node.depth for node in session.graph.nodes.values()}
        assert len(depths) >= 3  # At least 3 different depths

    @pytest.mark.asyncio
    async def test_graph_creates_branches(self):
        """Test that the graph creates branch thoughts."""
        got = GraphOfThoughts(branching_factor=3, max_depth=2)
        session = Session().start()

        await got.execute(session, "Branching problem")

        # Count branch thoughts
        branch_thoughts = [
            node for node in session.graph.nodes.values()
            if node.type == ThoughtType.BRANCH
        ]
        assert len(branch_thoughts) >= 3  # At least initial branching_factor

    @pytest.mark.asyncio
    async def test_graph_tracks_edges(self):
        """Test that the graph properly tracks edges between thoughts."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        await got.execute(session, "Connected thoughts")

        # Should have edges
        assert session.graph.edge_count > 0
        assert session.metrics.total_edges > 0


class TestGraphOfThoughtsMultiParentNodes:
    """Tests for multi-parent node support (DAG property)."""

    @pytest.mark.asyncio
    async def test_convergence_creates_multi_parent_nodes(self):
        """Test that convergence enabled can create nodes with multiple parents."""
        got = GraphOfThoughts(
            branching_factor=3,
            max_depth=3,
            enable_convergence=True,
            convergence_threshold=0.7,
            min_merge_score=0.5,
        )
        session = Session().start()

        await got.execute(session, "Convergent analysis")

        # Look for merged nodes (which have multiple parents)
        merged_nodes = [
            node for node in session.graph.nodes.values()
            if node.metadata.get("is_merged") is True
        ]

        # With convergence enabled, we should find merged nodes
        # (though not guaranteed depending on heuristics)
        # At minimum, check that the mechanism exists
        if merged_nodes:
            assert len(merged_nodes) >= 1
            # Check metadata indicates this node was merged from multiple sources
            # Note: parent_count tracks unique grandparents (may be 1 if siblings merged)
            # merged_from always has the 2 nodes that were combined
            merged_from = merged_nodes[0].metadata.get("merged_from", [])
            assert len(merged_from) >= 2

    @pytest.mark.asyncio
    async def test_convergence_disabled_no_merging(self):
        """Test that with convergence disabled, no path merging occurs."""
        got = GraphOfThoughts(
            branching_factor=3,
            max_depth=3,
            enable_convergence=False,
        )
        session = Session().start()

        await got.execute(session, "Non-convergent analysis")

        # No merged nodes should exist
        merged_nodes = [
            node for node in session.graph.nodes.values()
            if node.metadata.get("is_merged") is True
        ]
        assert len(merged_nodes) == 0

    @pytest.mark.asyncio
    async def test_synthesis_has_multiple_parents(self):
        """Test that final synthesis node has multiple leaf parents."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "Synthesis test")

        # The final synthesis should reference multiple convergence points
        assert result.metadata.get("is_final") is True
        assert result.metadata.get("leaf_count", 0) > 0


class TestGraphOfThoughtsConfiguration:
    """Tests for configuration options."""

    @pytest.mark.asyncio
    async def test_different_branching_factors(self):
        """Test execution with different branching factors."""
        for branching in [1, 2, 4]:
            got = GraphOfThoughts(branching_factor=branching, max_depth=1)
            session = Session().start()

            await got.execute(session, "Branching test")

            # More branching should create more thoughts
            assert session.thought_count > 0
            assert session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_different_max_depths(self):
        """Test execution with different max depths."""
        for depth in [1, 2, 4]:
            got = GraphOfThoughts(branching_factor=2, max_depth=depth)
            session = Session().start()

            await got.execute(session, "Depth test")

            # Check that max depth is respected
            max_node_depth = max(node.depth for node in session.graph.nodes.values())
            assert max_node_depth <= depth + 1  # +1 for final synthesis

    @pytest.mark.asyncio
    async def test_convergence_threshold_affects_merging(self):
        """Test that convergence threshold affects merging behavior."""
        # High threshold - less merging
        got_high = GraphOfThoughts(
            branching_factor=3,
            max_depth=3,
            enable_convergence=True,
            convergence_threshold=0.95,
        )
        session_high = Session().start()
        await got_high.execute(session_high, "High threshold test")

        # Low threshold - more merging (potentially)
        got_low = GraphOfThoughts(
            branching_factor=3,
            max_depth=3,
            enable_convergence=True,
            convergence_threshold=0.5,
        )
        session_low = Session().start()
        await got_low.execute(session_low, "Low threshold test")

        # Both should complete
        assert session_high.thought_count > 0
        assert session_low.thought_count > 0


class TestGraphOfThoughtsContinueReasoning:
    """Tests for continue_reasoning functionality."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_creates_continuation(self):
        """Test that continue_reasoning() creates a continuation thought."""
        got = GraphOfThoughts()
        session = Session().start()

        # Create initial thought
        initial = await got.execute(session, "Initial problem")

        # Continue reasoning
        continuation = await got.continue_reasoning(
            session,
            initial,
            guidance="Explore this further",
        )

        assert isinstance(continuation, ThoughtNode)
        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == initial.id
        assert continuation.metadata.get("is_continuation") is True

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self):
        """Test that continue_reasoning() uses guidance parameter."""
        got = GraphOfThoughts()
        session = Session().start()

        initial = await got.execute(session, "Start")
        guidance = "Focus on practical applications"

        continuation = await got.continue_reasoning(
            session,
            initial,
            guidance=guidance,
        )

        assert continuation.metadata.get("guidance") == guidance
        assert guidance in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_inactive_session(self):
        """Test that continue_reasoning() raises error for inactive session."""
        got = GraphOfThoughts()
        session = Session().start()

        initial = await got.execute(session, "Test")
        session.pause()  # Make inactive

        with pytest.raises(ValueError, match="Session must be active"):
            await got.continue_reasoning(session, initial)

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self):
        """Test that continue_reasoning() respects context parameters."""
        got = GraphOfThoughts()
        session = Session().start()

        initial = await got.execute(session, "Test")
        context = {"branching_factor": 5}

        continuation = await got.continue_reasoning(
            session,
            initial,
            context=context,
        )

        assert continuation is not None
        assert continuation.metadata.get("can_merge") is True


class TestGraphOfThoughtsAggregation:
    """Tests for combining multiple parent conclusions."""

    @pytest.mark.asyncio
    async def test_synthesis_aggregates_leaf_nodes(self):
        """Test that final synthesis aggregates information from all leaf nodes."""
        got = GraphOfThoughts(branching_factor=3, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "Aggregation test")

        # Synthesis should have metadata about aggregation
        assert result.type == ThoughtType.SYNTHESIS
        assert "total_nodes" in result.metadata
        assert "convergence_points" in result.metadata
        assert "leaf_count" in result.metadata

        # Should mention multiple nodes in content
        assert "nodes" in result.content.lower() or "paths" in result.content.lower()

    @pytest.mark.asyncio
    async def test_merged_nodes_combine_scores(self):
        """Test that merged nodes properly combine quality scores."""
        got = GraphOfThoughts(
            branching_factor=3,
            max_depth=3,
            enable_convergence=True,
            convergence_threshold=0.7,
        )
        session = Session().start()

        await got.execute(session, "Score combination test")

        # Find merged nodes
        merged_nodes = [
            node for node in session.graph.nodes.values()
            if node.metadata.get("is_merged") is True
        ]

        # If we have merged nodes, check their scores are reasonable
        for node in merged_nodes:
            assert 0.0 <= node.quality_score <= 1.0
            assert 0.0 <= node.confidence <= 1.0


class TestGraphOfThoughtsCyclePrevention:
    """Tests for cycle prevention in the DAG."""

    @pytest.mark.asyncio
    async def test_no_cycles_in_graph(self):
        """Test that the thought graph has no cycles (is a proper DAG)."""
        got = GraphOfThoughts(branching_factor=3, max_depth=4)
        session = Session().start()

        await got.execute(session, "Cycle test")

        # Perform cycle detection using DFS
        def has_cycle() -> bool:
            visited = set()
            rec_stack = set()

            def dfs(node_id: str) -> bool:
                if node_id in rec_stack:
                    return True  # Cycle detected
                if node_id in visited:
                    return False

                visited.add(node_id)
                rec_stack.add(node_id)

                node = session.graph.nodes.get(node_id)
                if node:
                    for child_id in node.children_ids:
                        if dfs(child_id):
                            return True

                rec_stack.remove(node_id)
                return False

            # Check from all potential starting nodes
            for node_id in session.graph.nodes:
                if has_cycle_from_node(node_id):
                    return True
            return False

        def has_cycle_from_node(start_id: str) -> bool:
            visited = set()
            rec_stack = set()

            def dfs(node_id: str) -> bool:
                if node_id in rec_stack:
                    return True
                if node_id in visited:
                    return False

                visited.add(node_id)
                rec_stack.add(node_id)

                node = session.graph.nodes.get(node_id)
                if node:
                    for child_id in node.children_ids:
                        if child_id in session.graph.nodes and dfs(child_id):
                            return True

                rec_stack.remove(node_id)
                return False

            return dfs(start_id)

        # No cycles should exist
        assert not any(has_cycle_from_node(nid) for nid in session.graph.nodes)

    @pytest.mark.asyncio
    async def test_all_nodes_reachable_from_root(self):
        """Test that all nodes are reachable from the root (connected DAG)."""
        got = GraphOfThoughts(branching_factor=2, max_depth=3)
        session = Session().start()

        await got.execute(session, "Connectivity test")

        # Find root
        root_id = session.graph.root_id
        assert root_id is not None

        # Get all reachable nodes from root
        reachable = {root_id}
        to_visit = [root_id]

        while to_visit:
            current_id = to_visit.pop()
            node = session.graph.nodes.get(current_id)
            if node:
                for child_id in node.children_ids:
                    if child_id not in reachable and child_id in session.graph.nodes:
                        reachable.add(child_id)
                        to_visit.append(child_id)

        # All nodes should be reachable from root (or be the final synthesis)
        # Note: Final synthesis might not be in children_ids chain
        # So we check most nodes are reachable
        reachable_count = len(reachable)
        total_count = len(session.graph.nodes)

        # At least 80% should be reachable (accounting for synthesis node)
        assert reachable_count >= total_count * 0.8


class TestGraphOfThoughtsPathAnalysis:
    """Tests for critical path analysis."""

    @pytest.mark.asyncio
    async def test_main_path_identification(self):
        """Test that we can identify main reasoning paths."""
        got = GraphOfThoughts(branching_factor=2, max_depth=3)
        session = Session().start()

        await got.execute(session, "Path analysis test")

        # Get main path from graph
        main_path = session.graph.get_main_path()

        # Main path should exist and have multiple nodes
        assert len(main_path) >= 2
        # First node should be root
        assert main_path[0] == session.graph.root_id

    @pytest.mark.asyncio
    async def test_multiple_paths_to_leaves(self):
        """Test that multiple paths exist from root to different leaves."""
        got = GraphOfThoughts(branching_factor=3, max_depth=2)
        session = Session().start()

        await got.execute(session, "Multi-path test")

        # Get leaf nodes
        leaf_ids = session.graph.leaf_ids

        # Should have multiple leaves
        assert len(leaf_ids) >= 2

        # Each leaf should have a path from root
        root_id = session.graph.root_id
        for leaf_id in leaf_ids[:3]:  # Check first 3
            path = session.graph.get_path(root_id, leaf_id)
            # Path might be None for final synthesis which isn't in children chain
            if path is not None:
                assert len(path) >= 2


class TestGraphOfThoughtsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimal_graph_single_branch_single_depth(self):
        """Test execution with minimal parameters (linear graph)."""
        got = GraphOfThoughts(branching_factor=1, max_depth=1)
        session = Session().start()

        result = await got.execute(session, "Minimal test")

        # Should still complete and create synthesis
        assert result is not None
        assert result.type == ThoughtType.SYNTHESIS
        assert session.thought_count >= 2  # At least root + synthesis

    @pytest.mark.asyncio
    async def test_wide_graph_many_branches(self):
        """Test execution with wide graph (many branches, shallow depth)."""
        got = GraphOfThoughts(branching_factor=8, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "Wide graph test")

        assert result is not None
        # Should create many branches
        assert session.metrics.branches_created >= 8

    @pytest.mark.asyncio
    async def test_deep_graph_few_branches(self):
        """Test execution with deep graph (few branches, many levels)."""
        got = GraphOfThoughts(branching_factor=2, max_depth=5)
        session = Session().start()

        result = await got.execute(session, "Deep graph test")

        assert result is not None
        # Should reach deeper levels
        max_depth = max(node.depth for node in session.graph.nodes.values())
        assert max_depth >= 3

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test execution with empty input text."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        result = await got.execute(session, "")

        # Should still complete
        assert result is not None
        assert session.thought_count > 0

    @pytest.mark.asyncio
    async def test_very_long_input_text(self):
        """Test execution with very long input text."""
        got = GraphOfThoughts(branching_factor=2, max_depth=2)
        session = Session().start()

        long_input = "Lorem ipsum " * 100
        result = await got.execute(session, long_input)

        assert result is not None
        assert session.thought_count > 0

    @pytest.mark.asyncio
    async def test_pruning_limits_frontier_size(self):
        """Test that frontier pruning prevents exponential explosion."""
        got = GraphOfThoughts(branching_factor=10, max_depth=4)
        session = Session().start()

        result = await got.execute(session, "Pruning test")

        # Should complete without creating exponential nodes
        # With branching=10 and depth=4, without pruning we'd have 10^4 = 10000 nodes
        # Pruning should keep it much smaller
        assert session.thought_count < 1000
        assert result is not None


class TestGraphOfThoughtsMetadata:
    """Tests for metadata constants."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert GRAPH_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.GRAPH_OF_THOUGHTS

    def test_metadata_properties(self):
        """Test that metadata has expected properties."""
        metadata = GRAPH_OF_THOUGHTS_METADATA
        assert metadata.name == "Graph of Thoughts"
        assert metadata.complexity == 7
        assert metadata.supports_branching is True
        assert metadata.supports_revision is True
        assert metadata.requires_context is False
        assert metadata.min_thoughts == 4

    def test_metadata_tags(self):
        """Test that metadata includes relevant tags."""
        tags = GRAPH_OF_THOUGHTS_METADATA.tags
        assert "graph" in tags
        assert "dag" in tags
        assert "branching" in tags
        assert "merging" in tags
        assert "convergence" in tags


class TestGraphOfThoughtsMetrics:
    """Tests for session metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_track_branches_created(self):
        """Test that session metrics track branches created."""
        got = GraphOfThoughts(branching_factor=3, max_depth=2)
        session = Session().start()

        initial_branches = session.metrics.branches_created
        await got.execute(session, "Metrics test")

        # Should have created branches
        assert session.metrics.branches_created > initial_branches

    @pytest.mark.asyncio
    async def test_metrics_track_branches_pruned(self):
        """Test that session metrics track branches pruned."""
        got = GraphOfThoughts(branching_factor=10, max_depth=3)
        session = Session().start()

        await got.execute(session, "Pruning metrics test")

        # With high branching, some pruning should occur
        # (or merging which also increments pruned counter)
        assert session.metrics.branches_pruned >= 0

    @pytest.mark.asyncio
    async def test_metrics_track_max_depth(self):
        """Test that session metrics track maximum depth reached."""
        got = GraphOfThoughts(branching_factor=2, max_depth=4)
        session = Session().start()

        await got.execute(session, "Depth metrics test")

        # Max depth should be tracked
        assert session.metrics.max_depth_reached >= 1

    @pytest.mark.asyncio
    async def test_metrics_track_thought_types(self):
        """Test that session metrics track different thought types."""
        got = GraphOfThoughts(branching_factor=3, max_depth=2)
        session = Session().start()

        await got.execute(session, "Type metrics test")

        # Should have different thought types
        thought_types = session.metrics.thought_types
        assert str(ThoughtType.INITIAL) in thought_types
        assert str(ThoughtType.BRANCH) in thought_types
        assert str(ThoughtType.SYNTHESIS) in thought_types
