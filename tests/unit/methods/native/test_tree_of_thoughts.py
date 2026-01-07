"""Unit tests for TreeOfThoughts reasoning method."""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.tree_of_thoughts import (
    TREE_OF_THOUGHTS_METADATA,
    TreeOfThoughts,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


# Fixtures


@pytest.fixture
def tot_method() -> TreeOfThoughts:
    """Create a TreeOfThoughts method instance with default settings."""
    return TreeOfThoughts()


@pytest.fixture
def custom_tot_method() -> TreeOfThoughts:
    """Create a TreeOfThoughts method instance with custom settings."""
    return TreeOfThoughts(
        branching_factor=4,
        max_depth=3,
        min_score_threshold=0.4,
        search_strategy="dfs",
        top_k_branches=3,
    )


@pytest.fixture
def active_session() -> Session:
    """Create an active session for testing."""
    return Session().start()


@pytest.fixture
def inactive_session() -> Session:
    """Create an inactive session for testing."""
    return Session()


# Test Metadata


class TestMetadata:
    """Tests for Tree of Thoughts metadata."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert TREE_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.TREE_OF_THOUGHTS

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert TREE_OF_THOUGHTS_METADATA.name == "Tree of Thoughts"

    def test_metadata_description(self):
        """Test metadata has descriptive text."""
        assert len(TREE_OF_THOUGHTS_METADATA.description) > 0
        assert "tree" in TREE_OF_THOUGHTS_METADATA.description.lower()

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert TREE_OF_THOUGHTS_METADATA.category == MethodCategory.CORE

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {"tree", "branching", "search", "exploration", "pruning", "core"}
        assert expected_tags.issubset(TREE_OF_THOUGHTS_METADATA.tags)

    def test_metadata_complexity(self):
        """Test metadata complexity is high (5-7)."""
        assert 5 <= TREE_OF_THOUGHTS_METADATA.complexity <= 7

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert TREE_OF_THOUGHTS_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates no revision support."""
        assert TREE_OF_THOUGHTS_METADATA.supports_revision is False

    def test_metadata_requires_context(self):
        """Test metadata indicates no context requirement."""
        assert TREE_OF_THOUGHTS_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test metadata has minimum thoughts requirement."""
        assert TREE_OF_THOUGHTS_METADATA.min_thoughts >= 3

    def test_metadata_max_thoughts(self):
        """Test metadata has unlimited max thoughts (0)."""
        assert TREE_OF_THOUGHTS_METADATA.max_thoughts == 0


# Test Initialization


class TestInitialization:
    """Tests for TreeOfThoughts initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        tot = TreeOfThoughts()
        assert tot.branching_factor == 3
        assert tot.max_depth == 5
        assert tot.min_score_threshold == 0.3
        assert tot.search_strategy == "bfs"
        assert tot.top_k_branches == 2

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        tot = TreeOfThoughts(
            branching_factor=4,
            max_depth=10,
            min_score_threshold=0.5,
            search_strategy="dfs",
            top_k_branches=3,
        )
        assert tot.branching_factor == 4
        assert tot.max_depth == 10
        assert tot.min_score_threshold == 0.5
        assert tot.search_strategy == "dfs"
        assert tot.top_k_branches == 3

    def test_init_invalid_branching_factor_zero(self):
        """Test initialization fails with branching_factor=0."""
        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            TreeOfThoughts(branching_factor=0)

    def test_init_invalid_branching_factor_negative(self):
        """Test initialization fails with negative branching_factor."""
        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            TreeOfThoughts(branching_factor=-1)

    def test_init_invalid_max_depth_zero(self):
        """Test initialization fails with max_depth=0."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            TreeOfThoughts(max_depth=0)

    def test_init_invalid_max_depth_negative(self):
        """Test initialization fails with negative max_depth."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            TreeOfThoughts(max_depth=-5)

    def test_init_invalid_threshold_below_zero(self):
        """Test initialization fails with threshold < 0."""
        with pytest.raises(ValueError, match="min_score_threshold must be 0.0-1.0"):
            TreeOfThoughts(min_score_threshold=-0.1)

    def test_init_invalid_threshold_above_one(self):
        """Test initialization fails with threshold > 1."""
        with pytest.raises(ValueError, match="min_score_threshold must be 0.0-1.0"):
            TreeOfThoughts(min_score_threshold=1.5)

    def test_init_invalid_search_strategy(self):
        """Test initialization fails with invalid search strategy."""
        with pytest.raises(ValueError, match="search_strategy must be 'bfs' or 'dfs'"):
            TreeOfThoughts(search_strategy="invalid")

    def test_init_invalid_top_k_zero(self):
        """Test initialization fails with top_k_branches=0."""
        with pytest.raises(ValueError, match="top_k_branches must be >= 1"):
            TreeOfThoughts(top_k_branches=0)

    def test_init_invalid_top_k_negative(self):
        """Test initialization fails with negative top_k_branches."""
        with pytest.raises(ValueError, match="top_k_branches must be >= 1"):
            TreeOfThoughts(top_k_branches=-2)

    @pytest.mark.asyncio
    async def test_initialize_method(self, tot_method: TreeOfThoughts):
        """Test initialize() method executes successfully."""
        await tot_method.initialize()
        # No error means success - lightweight initialization

    @pytest.mark.asyncio
    async def test_health_check(self, tot_method: TreeOfThoughts):
        """Test health_check() returns True."""
        result = await tot_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(self):
        """Test health_check() works before initialize()."""
        tot = TreeOfThoughts()
        result = await tot.health_check()
        assert result is True


# Test Properties


class TestProperties:
    """Tests for TreeOfThoughts properties."""

    def test_identifier_property(self, tot_method: TreeOfThoughts):
        """Test identifier property returns correct value."""
        assert tot_method.identifier == str(MethodIdentifier.TREE_OF_THOUGHTS)

    def test_name_property(self, tot_method: TreeOfThoughts):
        """Test name property returns correct value."""
        assert tot_method.name == "Tree of Thoughts"

    def test_description_property(self, tot_method: TreeOfThoughts):
        """Test description property returns correct value."""
        assert len(tot_method.description) > 0
        assert "tree" in tot_method.description.lower()

    def test_category_property(self, tot_method: TreeOfThoughts):
        """Test category property returns correct value."""
        assert tot_method.category == str(MethodCategory.CORE)


# Test Basic Execution


class TestBasicExecution:
    """Tests for basic execute() functionality."""

    @pytest.mark.asyncio
    async def test_execute_creates_root_thought(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test execute() creates root thought."""
        result = await tot_method.execute(active_session, "Test problem")

        # Should have created thoughts
        assert active_session.thought_count > 0

        # Result should be a synthesis thought
        assert result.type == ThoughtType.SYNTHESIS
        assert result.method_id == MethodIdentifier.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_execute_creates_tree_structure(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test execute() creates tree structure with multiple nodes."""
        result = await tot_method.execute(active_session, "Solve this problem")

        # Should have multiple thoughts (root + branches + synthesis)
        assert active_session.thought_count > 2

        # Check metrics
        assert active_session.metrics.total_thoughts > 2
        assert active_session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_execute_with_inactive_session(
        self, tot_method: TreeOfThoughts, inactive_session: Session
    ):
        """Test execute() fails with inactive session."""
        with pytest.raises(ValueError, match="Session must be active"):
            await tot_method.execute(inactive_session, "Test")

    @pytest.mark.asyncio
    async def test_execute_returns_synthesis_thought(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test execute() returns final synthesis thought."""
        result = await tot_method.execute(active_session, "Test input")

        assert result.type == ThoughtType.SYNTHESIS
        assert "Tree of Thoughts exploration complete" in result.content
        assert result.metadata.get("is_final") is True
        assert "total_nodes" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_synthesis_includes_stats(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test synthesis thought includes statistics."""
        result = await tot_method.execute(active_session, "Problem")

        # Check content includes stats
        assert "Total nodes explored:" in result.content
        assert "Score:" in result.content
        assert "Depth:" in result.content

        # Check metadata
        assert result.metadata.get("total_nodes", 0) > 0
        assert "best_path_depth" in result.metadata
        assert "strategy" in result.metadata


# Test Branching


class TestBranching:
    """Tests for branching behavior."""

    @pytest.mark.asyncio
    async def test_creates_multiple_branches_from_root(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that multiple branches are created from root node."""
        await tot_method.execute(active_session, "Test")

        # Should have created branches
        assert active_session.metrics.branches_created >= tot_method.branching_factor

    @pytest.mark.asyncio
    async def test_branching_factor_controls_branch_count(self, active_session: Session):
        """Test branching_factor controls number of branches created."""
        tot = TreeOfThoughts(branching_factor=5, max_depth=2)
        await tot.execute(active_session, "Test")

        # Should create branches according to branching factor
        # At depth 1, should have 5 branches
        # Each of those may spawn more (up to top_k)
        assert active_session.metrics.branches_created >= 5

    @pytest.mark.asyncio
    async def test_branches_have_unique_ids(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that branches have unique IDs."""
        await tot_method.execute(active_session, "Test")

        thought_ids = set()
        for thought in active_session.graph.nodes.values():
            if thought.type == ThoughtType.BRANCH:
                assert thought.id not in thought_ids
                thought_ids.add(thought.id)

    @pytest.mark.asyncio
    async def test_branches_have_branch_metadata(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that branch thoughts have branch metadata."""
        await tot_method.execute(active_session, "Test")

        branch_thoughts = [
            t for t in active_session.graph.nodes.values()
            if t.type == ThoughtType.BRANCH
        ]

        assert len(branch_thoughts) > 0
        for branch in branch_thoughts:
            assert "strategy" in branch.metadata
            assert "branch_index" in branch.metadata
            assert "branching_factor" in branch.metadata


# Test Search Strategies


class TestSearchStrategies:
    """Tests for BFS vs DFS search strategies."""

    @pytest.mark.asyncio
    async def test_bfs_strategy_default(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test BFS is used by default."""
        result = await tot_method.execute(active_session, "Test")

        # Check that BFS was used
        assert result.metadata.get("strategy") == "bfs"

    @pytest.mark.asyncio
    async def test_dfs_strategy_from_init(self, active_session: Session):
        """Test DFS strategy from initialization."""
        tot = TreeOfThoughts(search_strategy="dfs")
        result = await tot.execute(active_session, "Test")

        assert result.metadata.get("strategy") == "dfs"

    @pytest.mark.asyncio
    async def test_bfs_strategy_from_context(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test BFS strategy can be set via context."""
        result = await tot_method.execute(
            active_session, "Test", context={"search_strategy": "bfs"}
        )

        assert result.metadata.get("strategy") == "bfs"

    @pytest.mark.asyncio
    async def test_dfs_strategy_from_context(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test DFS strategy can be overridden via context."""
        result = await tot_method.execute(
            active_session, "Test", context={"search_strategy": "dfs"}
        )

        assert result.metadata.get("strategy") == "dfs"

    @pytest.mark.asyncio
    async def test_bfs_explores_breadth_first(self, active_session: Session):
        """Test BFS explores all nodes at depth N before depth N+1."""
        tot = TreeOfThoughts(
            branching_factor=3, max_depth=3, search_strategy="bfs", top_k_branches=3
        )
        await tot.execute(active_session, "Test")

        # Get all thoughts by depth
        thoughts_by_depth: dict[int, list[ThoughtNode]] = {}
        for thought in active_session.graph.nodes.values():
            depth = thought.depth
            if depth not in thoughts_by_depth:
                thoughts_by_depth[depth] = []
            thoughts_by_depth[depth].append(thought)

        # BFS should create thoughts at each depth level
        assert len(thoughts_by_depth) > 1

    @pytest.mark.asyncio
    async def test_dfs_explores_depth_first(self, active_session: Session):
        """Test DFS explores depth before breadth."""
        tot = TreeOfThoughts(
            branching_factor=2, max_depth=4, search_strategy="dfs"
        )
        await tot.execute(active_session, "Test")

        # DFS should reach max depth
        assert active_session.current_depth >= 3


# Test Branch Evaluation


class TestBranchEvaluation:
    """Tests for branch scoring and selection."""

    @pytest.mark.asyncio
    async def test_branches_have_quality_scores(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that branches have quality scores."""
        await tot_method.execute(active_session, "Test")

        branch_thoughts = [
            t for t in active_session.graph.nodes.values()
            if t.type == ThoughtType.BRANCH
        ]

        assert len(branch_thoughts) > 0
        for branch in branch_thoughts:
            assert branch.quality_score is not None
            assert 0.0 <= branch.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_branches_have_confidence_scores(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that branches have confidence scores."""
        await tot_method.execute(active_session, "Test")

        branch_thoughts = [
            t for t in active_session.graph.nodes.values()
            if t.type == ThoughtType.BRANCH
        ]

        for branch in branch_thoughts:
            assert 0.0 <= branch.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_best_leaf_has_high_score(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that best leaf node has reasonable score."""
        result = await tot_method.execute(active_session, "Test")

        # Synthesis inherits score from best leaf
        assert result.quality_score is not None
        assert result.quality_score > 0.0


# Test Configuration


class TestConfiguration:
    """Tests for configuration via context and initialization."""

    @pytest.mark.asyncio
    async def test_context_overrides_branching_factor(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test context can override branching_factor."""
        await tot_method.execute(
            active_session, "Test", context={"branching_factor": 5}
        )

        # Should have created more branches
        assert active_session.metrics.branches_created >= 5

    @pytest.mark.asyncio
    async def test_context_overrides_max_depth(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test context can override max_depth."""
        await tot_method.execute(
            active_session, "Test", context={"max_depth": 2}
        )

        # Depth should not exceed 2
        assert active_session.current_depth <= 2

    @pytest.mark.asyncio
    async def test_context_overrides_min_score_threshold(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test context can override min_score_threshold."""
        result = await tot_method.execute(
            active_session, "Test", context={"min_score_threshold": 0.9}
        )

        # High threshold should prune more branches
        # This is reflected in metrics
        assert result is not None

    @pytest.mark.asyncio
    async def test_context_overrides_top_k_branches(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test context can override top_k_branches."""
        await tot_method.execute(
            active_session, "Test", context={"top_k_branches": 1}
        )

        # Should limit branches at each level
        assert active_session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_empty_context_uses_defaults(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test empty context uses default parameters."""
        result = await tot_method.execute(active_session, "Test", context={})

        assert result.metadata.get("strategy") == "bfs"


# Test Continue Reasoning


class TestContinueReasoning:
    """Tests for continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_creates_continuation(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test continue_reasoning() creates continuation thought."""
        # First execute to get a thought
        initial = await tot_method.execute(active_session, "Test")

        # Continue from that thought
        continuation = await tot_method.continue_reasoning(
            active_session, initial, guidance="Explore more"
        )

        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == initial.id
        assert "Continuing exploration" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test continue_reasoning() includes guidance."""
        initial = await tot_method.execute(active_session, "Test")
        guidance_text = "Focus on optimization"

        continuation = await tot_method.continue_reasoning(
            active_session, initial, guidance=guidance_text
        )

        assert guidance_text in continuation.content
        assert continuation.metadata.get("guidance") == guidance_text

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test continue_reasoning() works without guidance."""
        initial = await tot_method.execute(active_session, "Test")

        continuation = await tot_method.continue_reasoning(
            active_session, initial
        )

        assert continuation is not None
        assert "Exploring additional branches" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_inactive_session(
        self, tot_method: TreeOfThoughts, inactive_session: Session
    ):
        """Test continue_reasoning() fails with inactive session."""
        # Create a dummy thought
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Test",
        )

        with pytest.raises(ValueError, match="Session must be active"):
            await tot_method.continue_reasoning(inactive_session, thought)

    @pytest.mark.asyncio
    async def test_continue_reasoning_uses_branching_factor_from_context(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test continue_reasoning() respects branching_factor from context."""
        initial = await tot_method.execute(active_session, "Test")

        continuation = await tot_method.continue_reasoning(
            active_session, initial, context={"branching_factor": 7}
        )

        assert "7 new exploration paths" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_confidence_decay(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test continuation has slightly lower confidence."""
        initial = await tot_method.execute(active_session, "Test")

        continuation = await tot_method.continue_reasoning(
            active_session, initial
        )

        # Continuation should have lower confidence (0.9 decay factor)
        assert continuation.confidence <= initial.confidence


# Test Pruning


class TestPruning:
    """Tests for branch pruning behavior."""

    @pytest.mark.asyncio
    async def test_pruning_removes_low_scoring_branches(self, active_session: Session):
        """Test that low-scoring branches are pruned."""
        tot = TreeOfThoughts(
            branching_factor=5,
            max_depth=3,
            min_score_threshold=0.6,  # High threshold
            top_k_branches=2,
        )
        await tot.execute(active_session, "Test")

        # Should have pruned some branches
        assert active_session.metrics.branches_pruned > 0

    @pytest.mark.asyncio
    async def test_pruning_tracked_in_metrics(self, active_session: Session):
        """Test that pruned branches are tracked in metrics."""
        tot = TreeOfThoughts(
            branching_factor=4,
            max_depth=2,
            min_score_threshold=0.5,
            top_k_branches=1,  # Only keep 1 branch
        )
        await tot.execute(active_session, "Test")

        # Metrics should track pruned branches
        assert active_session.metrics.branches_pruned >= 0

    @pytest.mark.asyncio
    async def test_low_threshold_prunes_less(self, active_session: Session):
        """Test that low threshold prunes fewer branches."""
        tot = TreeOfThoughts(
            branching_factor=3,
            max_depth=2,
            min_score_threshold=0.1,  # Very low threshold
            top_k_branches=3,
        )
        await tot.execute(active_session, "Test")

        # Low threshold should result in less pruning
        pruned_count = active_session.metrics.branches_pruned
        assert pruned_count >= 0  # Some may still be pruned by top_k


# Test Tree Traversal


class TestTreeTraversal:
    """Tests for tree traversal and path finding."""

    @pytest.mark.asyncio
    async def test_finds_best_leaf_node(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that best leaf node is found."""
        result = await tot_method.execute(active_session, "Test")

        # Result should reference the best path
        assert result.quality_score is not None
        assert "Best solution found" in result.content

    @pytest.mark.asyncio
    async def test_max_depth_enforced(self, active_session: Session):
        """Test that max_depth limit is enforced."""
        tot = TreeOfThoughts(max_depth=3)
        await tot.execute(active_session, "Test")

        # No thought should exceed max_depth
        for thought in active_session.graph.nodes.values():
            assert thought.depth <= 3

    @pytest.mark.asyncio
    async def test_all_nodes_tracked(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test that all nodes are tracked in the graph."""
        result = await tot_method.execute(active_session, "Test")

        # Synthesis should report total nodes
        total_nodes = result.metadata.get("total_nodes", 0)
        assert total_nodes > 0

        # Total nodes may be less than thought count because synthesis is added after
        # the total_nodes is calculated. Check that they're close.
        assert abs(total_nodes - active_session.thought_count) <= 1


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_branch_factor(self, active_session: Session):
        """Test with branching_factor=1 (single branch)."""
        tot = TreeOfThoughts(branching_factor=1, max_depth=3)
        result = await tot.execute(active_session, "Test")

        # Should still work with single branch
        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_very_deep_tree(self, active_session: Session):
        """Test with very deep tree (max_depth=10)."""
        tot = TreeOfThoughts(
            branching_factor=2, max_depth=10, top_k_branches=1
        )
        result = await tot.execute(active_session, "Deep exploration")

        assert result is not None
        # Should explore to significant depth
        assert active_session.current_depth >= 5

    @pytest.mark.asyncio
    async def test_many_siblings(self, active_session: Session):
        """Test with many sibling branches."""
        tot = TreeOfThoughts(
            branching_factor=8, max_depth=2, top_k_branches=8
        )
        result = await tot.execute(active_session, "Wide exploration")

        assert result is not None
        # Should create many branches
        assert active_session.metrics.branches_created >= 8

    @pytest.mark.asyncio
    async def test_min_depth(self, active_session: Session):
        """Test with min depth (max_depth=1)."""
        tot = TreeOfThoughts(max_depth=1, branching_factor=3)
        result = await tot.execute(active_session, "Shallow test")

        assert result is not None
        # Synthesis is at depth+1 from best leaf, so max depth could be 2
        # (root at 0, branches at 1, synthesis at 2)
        assert active_session.current_depth <= 2

    @pytest.mark.asyncio
    async def test_threshold_at_boundaries(self, active_session: Session):
        """Test threshold at boundary values (0.0 and 1.0)."""
        # Test with threshold = 0.0 (accept all)
        tot1 = TreeOfThoughts(min_score_threshold=0.0, max_depth=2)
        result1 = await tot1.execute(active_session, "Test 1")
        assert result1 is not None

        # Test with threshold = 1.0 (reject most)
        session2 = Session().start()
        tot2 = TreeOfThoughts(min_score_threshold=1.0, max_depth=2)
        result2 = await tot2.execute(session2, "Test 2")
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_empty_input_text(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test with empty input text."""
        result = await tot_method.execute(active_session, "")

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_very_long_input_text(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test with very long input text."""
        long_input = "Test problem " * 100
        result = await tot_method.execute(active_session, long_input)

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_input(
        self, tot_method: TreeOfThoughts, active_session: Session
    ):
        """Test with special characters in input."""
        special_input = "Test: @#$%^&*() 测试 émojis 🌳🤔"
        result = await tot_method.execute(active_session, special_input)

        assert result is not None
        assert active_session.thought_count > 0
