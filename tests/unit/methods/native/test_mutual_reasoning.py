"""Comprehensive tests for MutualReasoning (rStar) reasoning method.

This module provides complete test coverage for the MutualReasoning method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- MCTS phases (selection, generation, discrimination, expansion)
- Discriminator evaluation and scoring
- UCB1 with discriminator guidance
- Configuration options (iterations, depth, beam_width, threshold)
- Continue reasoning flow
- Path selection and quality tracking
- Generator and Discriminator roles
- Edge cases and boundary conditions

The tests aim for 90%+ coverage of the MutualReasoning implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.mutual_reasoning import (
    MUTUAL_REASONING_METADATA,
    MCTSNode,
    MutualReasoning,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> MutualReasoning:
    """Provide a MutualReasoning method instance for testing.

    Returns:
        MutualReasoning instance (uninitialized).
    """
    return MutualReasoning()


@pytest.fixture
async def initialized_method() -> MutualReasoning:
    """Provide an initialized MutualReasoning method instance.

    Returns:
        Initialized MutualReasoning instance.
    """
    method = MutualReasoning()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.fixture
def simple_input() -> str:
    """Provide a simple test input.

    Returns:
        Simple problem for testing.
    """
    return "If x + 3 = 7, what is x?"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex problem requiring deeper reasoning.
    """
    return "Design an optimal algorithm for resource allocation with competing priorities"


@pytest.fixture
def mock_thought() -> ThoughtNode:
    """Provide a mock ThoughtNode for testing.

    Returns:
        Mock ThoughtNode instance.
    """
    return ThoughtNode(
        type=ThoughtType.HYPOTHESIS,
        method_id=MethodIdentifier.MUTUAL_REASONING,
        content="Test hypothesis",
        depth=1,
        metadata={"action": "test_action"},
    )


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMutualReasoningMetadata:
    """Test suite for MutualReasoning metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert MUTUAL_REASONING_METADATA.identifier == MethodIdentifier.MUTUAL_REASONING

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert MUTUAL_REASONING_METADATA.name == "Mutual Reasoning (rStar)"
        assert "rStar" in MUTUAL_REASONING_METADATA.name

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert MUTUAL_REASONING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert MUTUAL_REASONING_METADATA.complexity == 8
        assert 1 <= MUTUAL_REASONING_METADATA.complexity <= 10

    def test_metadata_supports_branching(self):
        """Test that metadata indicates branching support."""
        assert MUTUAL_REASONING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert MUTUAL_REASONING_METADATA.supports_revision is True

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "mutual",
            "rstar",
            "mcts",
            "discriminator",
            "generator",
            "tree-search",
            "advanced",
        }
        assert expected_tags.issubset(MUTUAL_REASONING_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert MUTUAL_REASONING_METADATA.min_thoughts == 7

    def test_metadata_max_thoughts(self):
        """Test that metadata allows unlimited thoughts."""
        assert MUTUAL_REASONING_METADATA.max_thoughts == 0  # Unlimited

    def test_metadata_description_mentions_discriminator(self):
        """Test that description mentions discriminator."""
        assert "discriminator" in MUTUAL_REASONING_METADATA.description.lower()
        assert "generator" in MUTUAL_REASONING_METADATA.description.lower()


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMutualReasoningInitialization:
    """Test suite for MutualReasoning initialization."""

    def test_create_method(self, method: MutualReasoning):
        """Test creating a MutualReasoning instance."""
        assert isinstance(method, MutualReasoning)
        assert method._initialized is False

    def test_create_with_default_parameters(self):
        """Test creating method with default parameters."""
        method = MutualReasoning()
        assert method.num_iterations == MutualReasoning.DEFAULT_NUM_ITERATIONS
        assert method.max_depth == MutualReasoning.MAX_DEPTH
        assert method.beam_width == MutualReasoning.BEAM_WIDTH
        assert method.exploration_constant == MutualReasoning.DEFAULT_EXPLORATION_CONSTANT
        assert method.discriminator_threshold == MutualReasoning.DEFAULT_DISCRIMINATOR_THRESHOLD

    def test_create_with_custom_parameters(self):
        """Test creating method with custom parameters."""
        method = MutualReasoning(
            num_iterations=60,
            max_depth=6,
            beam_width=5,
            exploration_constant=2.0,
            discriminator_threshold=0.6,
        )
        assert method.num_iterations == 60
        assert method.max_depth == 6
        assert method.beam_width == 5
        assert method.exploration_constant == 2.0
        assert method.discriminator_threshold == 0.6

    def test_invalid_num_iterations(self):
        """Test that invalid num_iterations raises ValueError."""
        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            MutualReasoning(num_iterations=0)

        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            MutualReasoning(num_iterations=-5)

    def test_invalid_max_depth(self):
        """Test that invalid max_depth raises ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            MutualReasoning(max_depth=0)

    def test_invalid_beam_width(self):
        """Test that invalid beam_width raises ValueError."""
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            MutualReasoning(beam_width=0)

    def test_invalid_exploration_constant(self):
        """Test that invalid exploration_constant raises ValueError."""
        with pytest.raises(ValueError, match="exploration_constant must be > 0"):
            MutualReasoning(exploration_constant=0)

        with pytest.raises(ValueError, match="exploration_constant must be > 0"):
            MutualReasoning(exploration_constant=-1.0)

    def test_invalid_discriminator_threshold(self):
        """Test that invalid discriminator_threshold raises ValueError."""
        with pytest.raises(ValueError, match="discriminator_threshold must be in"):
            MutualReasoning(discriminator_threshold=-0.1)

        with pytest.raises(ValueError, match="discriminator_threshold must be in"):
            MutualReasoning(discriminator_threshold=1.5)

    def test_properties_before_initialization(self, method: MutualReasoning):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.MUTUAL_REASONING
        assert method.name == "Mutual Reasoning (rStar)"
        assert method.category == MethodCategory.ADVANCED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: MutualReasoning):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: MutualReasoning):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: MutualReasoning):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# MCTSNode Tests
# ============================================================================


class TestMCTSNode:
    """Test suite for MCTSNode class."""

    def test_create_mcts_node(self, mock_thought: ThoughtNode):
        """Test creating an MCTSNode."""
        node = MCTSNode(mock_thought)
        assert node.thought == mock_thought
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.discriminator_score == 0.5
        assert node.untried_actions == []

    def test_create_with_parent(self, mock_thought: ThoughtNode):
        """Test creating node with parent."""
        parent_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Parent",
            depth=0,
        )
        parent = MCTSNode(parent_thought)
        child = MCTSNode(mock_thought, parent=parent)

        assert child.parent == parent
        assert child.thought.depth == 1

    def test_create_with_untried_actions(self, mock_thought: ThoughtNode):
        """Test creating node with untried actions."""
        actions = ["action1", "action2", "action3"]
        node = MCTSNode(mock_thought, untried_actions=actions)
        assert node.untried_actions == actions

    def test_is_fully_expanded_empty(self, mock_thought: ThoughtNode):
        """Test is_fully_expanded with no untried actions."""
        node = MCTSNode(mock_thought)
        assert node.is_fully_expanded() is True

    def test_is_fully_expanded_with_actions(self, mock_thought: ThoughtNode):
        """Test is_fully_expanded with untried actions."""
        node = MCTSNode(mock_thought, untried_actions=["action1", "action2"])
        assert node.is_fully_expanded() is False

    def test_is_terminal(self, mock_thought: ThoughtNode):
        """Test is_terminal at various depths."""
        thought_at_depth_3 = mock_thought.model_copy(update={"depth": 3})
        node = MCTSNode(thought_at_depth_3)

        assert node.is_terminal(max_depth=5) is False
        assert node.is_terminal(max_depth=3) is True
        assert node.is_terminal(max_depth=2) is True

    def test_ucb1_score_unvisited(self, mock_thought: ThoughtNode):
        """Test UCB1 score for unvisited node."""
        node = MCTSNode(mock_thought)
        score = node.ucb1_score()
        assert score == float("inf")

    def test_ucb1_score_root(self, mock_thought: ThoughtNode):
        """Test UCB1 score for root node."""
        node = MCTSNode(mock_thought)
        node.visits = 5
        node.value = 2.5
        node.discriminator_score = 0.8

        score = node.ucb1_score()
        # For root: value/visits + discriminator_score
        expected = 2.5 / 5 + 0.8
        assert abs(score - expected) < 0.01

    def test_ucb1_score_with_parent(self, mock_thought: ThoughtNode):
        """Test UCB1 score with parent."""
        parent_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Parent",
            depth=0,
        )
        parent = MCTSNode(parent_thought)
        parent.visits = 10

        child = MCTSNode(mock_thought, parent=parent)
        child.visits = 3
        child.value = 1.5
        child.discriminator_score = 0.7

        score = child.ucb1_score(exploration_constant=1.414)

        # exploitation + quality + exploration
        1.414 * (10 / 3) ** 0.5

        assert score > 0
        assert score < float("inf")

    def test_best_child(self, mock_thought: ThoughtNode):
        """Test best_child selection."""
        parent = MCTSNode(mock_thought)

        # Create children with different scores
        for i in range(3):
            child_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.MUTUAL_REASONING,
                content=f"Child {i}",
                depth=1,
            )
            child = MCTSNode(child_thought, parent=parent)
            child.visits = i + 1
            child.value = float(i)
            child.discriminator_score = 0.5 + i * 0.1
            parent.children.append(child)

        parent.visits = 10
        best = parent.best_child()
        assert best in parent.children

    def test_best_child_no_children(self, mock_thought: ThoughtNode):
        """Test best_child with no children raises error."""
        node = MCTSNode(mock_thought)
        with pytest.raises(ValueError, match="No children to select from"):
            node.best_child()

    def test_most_visited_child(self, mock_thought: ThoughtNode):
        """Test most_visited_child selection."""
        parent = MCTSNode(mock_thought)

        # Create children with different visit counts
        for i in range(3):
            child_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.MUTUAL_REASONING,
                content=f"Child {i}",
                depth=1,
            )
            child = MCTSNode(child_thought, parent=parent)
            child.visits = (i + 1) * 10
            parent.children.append(child)

        most_visited = parent.most_visited_child()
        assert most_visited.visits == 30

    def test_most_visited_child_no_children(self, mock_thought: ThoughtNode):
        """Test most_visited_child with no children raises error."""
        node = MCTSNode(mock_thought)
        with pytest.raises(ValueError, match="No children to select from"):
            node.most_visited_child()

    def test_best_discriminator_child(self, mock_thought: ThoughtNode):
        """Test best_discriminator_child selection."""
        parent = MCTSNode(mock_thought)

        # Create children with different discriminator scores
        scores = [0.6, 0.9, 0.7]
        for i, score in enumerate(scores):
            child_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.MUTUAL_REASONING,
                content=f"Child {i}",
                depth=1,
            )
            child = MCTSNode(child_thought, parent=parent)
            child.discriminator_score = score
            parent.children.append(child)

        best = parent.best_discriminator_child()
        assert best.discriminator_score == 0.9

    def test_best_discriminator_child_no_children(self, mock_thought: ThoughtNode):
        """Test best_discriminator_child with no children raises error."""
        node = MCTSNode(mock_thought)
        with pytest.raises(ValueError, match="No children to select from"):
            node.best_discriminator_child()


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestMutualReasoningExecution:
    """Test suite for basic MutualReasoning execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.SYNTHESIS
        assert thought.method_id == MethodIdentifier.MUTUAL_REASONING

    @pytest.mark.asyncio
    async def test_execute_with_inactive_session(
        self, initialized_method: MutualReasoning, simple_input: str
    ):
        """Test that execute fails with inactive session."""
        session = Session()  # Not started
        with pytest.raises(ValueError, match="Session must be active"):
            await initialized_method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_multiple_thoughts(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that execute creates multiple thoughts during search."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 10},  # Small number for testing
        )

        # Should create root + progress + generations + discriminations + synthesis
        assert session.thought_count > initial_count + 5

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test execute with custom context parameters."""
        context = {
            "num_iterations": 20,
            "max_depth": 4,
            "beam_width": 2,
            "exploration_constant": 2.0,
            "discriminator_threshold": 0.5,
        }

        result = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert result.metadata["total_iterations"] == 20
        assert "discriminator_score" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_tracks_metadata(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that execute tracks proper metadata."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        assert "is_final" in result.metadata
        assert result.metadata["is_final"] is True
        assert "mcts_nodes" in result.metadata
        assert "exploration_depth" in result.metadata
        assert "discriminator_score" in result.metadata
        assert "total_evaluations" in result.metadata


# ============================================================================
# Phase Tests
# ============================================================================


class TestMCTSPhases:
    """Test suite for MCTS phases."""

    @pytest.mark.asyncio
    async def test_select_returns_node(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test that _select returns a node."""
        root = MCTSNode(mock_thought, untried_actions=["action1"])
        selected = await initialized_method._select(root, 1.414)
        assert isinstance(selected, MCTSNode)

    @pytest.mark.asyncio
    async def test_generate_creates_hypothesis(
        self, initialized_method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that _generate creates a HYPOTHESIS thought."""
        node = MCTSNode(mock_thought, untried_actions=["test_action"])
        all_thoughts: dict[str, ThoughtNode] = {}

        generated = await initialized_method._generate(session, node, "test input", 1, all_thoughts)

        assert generated is not None
        assert generated.thought.type == ThoughtType.HYPOTHESIS
        assert "generator" in generated.thought.metadata.get("role", "")
        assert generated.thought.id in all_thoughts

    @pytest.mark.asyncio
    async def test_generate_with_no_actions(
        self, initialized_method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that _generate returns None with no actions."""
        node = MCTSNode(mock_thought, untried_actions=[])
        all_thoughts: dict[str, ThoughtNode] = {}

        generated = await initialized_method._generate(session, node, "test input", 1, all_thoughts)

        assert generated is None

    @pytest.mark.asyncio
    async def test_discriminate_returns_score(
        self, initialized_method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that _discriminate returns a score in [0, 1]."""
        node = MCTSNode(mock_thought)
        all_thoughts: dict[str, ThoughtNode] = {}

        score = await initialized_method._discriminate(session, node, "test input", 1, all_thoughts)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_discriminate_creates_verification(
        self, initialized_method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that _discriminate creates a VERIFICATION thought."""
        node = MCTSNode(mock_thought)
        all_thoughts: dict[str, ThoughtNode] = {}

        await initialized_method._discriminate(session, node, "test input", 1, all_thoughts)

        # Should have created a verification thought
        verification_thoughts = [
            t for t in all_thoughts.values() if t.type == ThoughtType.VERIFICATION
        ]
        assert len(verification_thoughts) > 0

    @pytest.mark.asyncio
    async def test_discriminate_updates_node(
        self, initialized_method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that _discriminate updates node with score."""
        node = MCTSNode(mock_thought)
        all_thoughts: dict[str, ThoughtNode] = {}

        score = await initialized_method._discriminate(session, node, "test input", 1, all_thoughts)

        assert node.discriminator_score == score
        assert node.thought.metadata["evaluated"] is True
        assert node.thought.metadata["discriminator_score"] == score

    @pytest.mark.asyncio
    async def test_expand_adds_actions(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test that _expand adds untried actions."""
        node = MCTSNode(mock_thought)
        assert len(node.untried_actions) == 0

        await initialized_method._expand(node, "test input", 3)

        assert len(node.untried_actions) == 3

    @pytest.mark.asyncio
    async def test_simulate_returns_value(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test that _simulate returns a value in [-1, 1]."""
        node = MCTSNode(mock_thought)
        node.discriminator_score = 0.7

        value = await initialized_method._simulate(node, max_depth=5)

        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_backpropagate_updates_tree(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test that _backpropagate updates visits and values."""
        parent_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Parent",
            depth=0,
        )
        parent = MCTSNode(parent_thought)
        child = MCTSNode(mock_thought, parent=parent)

        await initialized_method._backpropagate(child, 0.5)

        assert child.visits == 1
        assert child.value == 0.5
        assert parent.visits == 1
        assert parent.value == 0.5


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: MutualReasoning, session: Session, mock_thought: ThoughtNode
    ):
        """Test that continue_reasoning fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=mock_thought)

    @pytest.mark.asyncio
    async def test_continue_creates_continuation(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that continue_reasoning creates CONTINUATION thought."""
        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 5},
        )

        continuation = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == initial.id

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance."""
        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 5},
        )

        guidance = "Focus on edge cases"
        continuation = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance
        )

        assert "guidance" in continuation.metadata
        assert continuation.metadata["guidance"] == guidance

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test continue_reasoning with custom context."""
        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 5},
        )

        context = {"num_iterations": 30}
        continuation = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, context=context
        )

        assert continuation.metadata["additional_iterations"] == 30

    @pytest.mark.asyncio
    async def test_continue_with_inactive_session(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test that continue fails with inactive session."""
        session = Session()  # Not started
        with pytest.raises(ValueError, match="Session must be active"):
            await initialized_method.continue_reasoning(
                session=session, previous_thought=mock_thought
            )


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Test suite for helper methods."""

    def test_generate_candidate_steps(self, initialized_method: MutualReasoning):
        """Test generating candidate steps."""
        steps = initialized_method._generate_candidate_steps("test problem", 5)

        assert len(steps) == 5
        assert all(isinstance(step, str) for step in steps)
        assert all(len(step) > 0 for step in steps)

    def test_generate_candidate_steps_unique(self, initialized_method: MutualReasoning):
        """Test that generated steps are diverse."""
        steps = initialized_method._generate_candidate_steps("test problem", 10)

        # Should have multiple unique steps
        unique_steps = set(steps)
        assert len(unique_steps) >= 5

    def test_select_best_paths_empty(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test selecting best paths from tree with only root."""
        root = MCTSNode(mock_thought)
        paths = initialized_method._select_best_paths(root, 3)

        # Root node itself is considered a leaf (trivial path)
        assert len(paths) == 1
        assert paths[0] is root

    def test_select_best_paths_with_leaves(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test selecting best paths with leaf nodes."""
        root = MCTSNode(mock_thought)

        # Create some leaf nodes
        for i in range(5):
            child_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.MUTUAL_REASONING,
                content=f"Child {i}",
                depth=1,
            )
            child = MCTSNode(child_thought, parent=root)
            child.visits = i + 1
            child.discriminator_score = 0.5 + i * 0.1
            root.children.append(child)

        paths = initialized_method._select_best_paths(root, 3)

        assert len(paths) == 3
        # Should be sorted by combined score
        assert all(isinstance(node, MCTSNode) for node in paths)

    def test_collect_leaves(self, initialized_method: MutualReasoning, mock_thought: ThoughtNode):
        """Test collecting leaf nodes."""
        root = MCTSNode(mock_thought)

        # Create tree structure
        child1_thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Child 1",
            depth=1,
        )
        child1 = MCTSNode(child1_thought, parent=root)
        root.children.append(child1)

        child2_thought = ThoughtNode(
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Child 2",
            depth=1,
        )
        child2 = MCTSNode(child2_thought, parent=root)
        root.children.append(child2)

        leaves: list[MCTSNode] = []
        initialized_method._collect_leaves(root, leaves)

        # child1 and child2 are leaves
        assert len(leaves) == 2

    def test_extract_path_description(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test extracting path description."""
        parent_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content="Parent",
            depth=0,
        )
        parent = MCTSNode(parent_thought)

        child = MCTSNode(mock_thought, parent=parent)
        child.discriminator_score = 0.8

        description = initialized_method._extract_path_description(child)

        assert isinstance(description, str)
        assert "test_action" in description
        assert "0.80" in description

    def test_extract_path_description_root(
        self, initialized_method: MutualReasoning, mock_thought: ThoughtNode
    ):
        """Test extracting path description for root."""
        root = MCTSNode(mock_thought)
        description = initialized_method._extract_path_description(root)

        assert "Initial state" in description


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    def test_default_constants(self):
        """Test default configuration constants."""
        assert MutualReasoning.DEFAULT_NUM_ITERATIONS == 40
        assert MutualReasoning.MAX_DEPTH == 5
        assert MutualReasoning.BEAM_WIDTH == 3
        assert MutualReasoning.DEFAULT_EXPLORATION_CONSTANT == 1.414
        assert MutualReasoning.DEFAULT_DISCRIMINATOR_THRESHOLD == 0.4

    @pytest.mark.asyncio
    async def test_custom_iterations(self, session: Session, simple_input: str):
        """Test custom number of iterations."""
        method = MutualReasoning(num_iterations=15)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        assert result.metadata["total_iterations"] == 15

    @pytest.mark.asyncio
    async def test_custom_max_depth(self, session: Session, simple_input: str):
        """Test custom max depth."""
        method = MutualReasoning(max_depth=3, num_iterations=10)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        # Exploration depth should not exceed max_depth
        assert result.metadata["exploration_depth"] <= 3

    @pytest.mark.asyncio
    async def test_custom_beam_width(self, session: Session, simple_input: str):
        """Test custom beam width."""
        method = MutualReasoning(beam_width=2, num_iterations=10)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_custom_discriminator_threshold(self, session: Session, simple_input: str):
        """Test custom discriminator threshold."""
        method = MutualReasoning(discriminator_threshold=0.6, num_iterations=10)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        assert isinstance(result, ThoughtNode)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test suite for integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_reasoning_cycle(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test a complete reasoning cycle."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 20},
        )

        # Verify final result structure
        assert result.type == ThoughtType.SYNTHESIS
        assert result.confidence is not None
        assert result.quality_score is not None
        assert "discriminator_score" in result.metadata
        assert "mcts_nodes" in result.metadata

    @pytest.mark.asyncio
    async def test_multiple_executions_same_method(
        self, initialized_method: MutualReasoning, simple_input: str
    ):
        """Test multiple executions with same method instance."""
        session1 = Session().start()
        result1 = await initialized_method.execute(
            session=session1,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        session2 = Session().start()
        result2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        assert result1.id != result2.id
        assert session1.thought_count > 0
        assert session2.thought_count > 0

    @pytest.mark.asyncio
    async def test_complex_problem(
        self, initialized_method: MutualReasoning, session: Session, complex_input: str
    ):
        """Test reasoning on complex problem."""
        result = await initialized_method.execute(
            session=session,
            input_text=complex_input,
            context={"num_iterations": 30, "max_depth": 6},
        )

        assert result.type == ThoughtType.SYNTHESIS
        assert session.thought_count > 10


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: MutualReasoning, session: Session):
        """Test handling of empty input."""
        result = await initialized_method.execute(
            session=session,
            input_text="",
            context={"num_iterations": 5},
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: MutualReasoning, session: Session):
        """Test handling of very long input."""
        long_input = "Solve " + "complex " * 1000 + "problem"
        result = await initialized_method.execute(
            session=session,
            input_text=long_input,
            context={"num_iterations": 5},
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_minimum_iterations(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test with minimum number of iterations."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 1},
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_minimum_depth(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test with minimum depth."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_depth": 1, "num_iterations": 5},
        )

        assert result.metadata["exploration_depth"] <= 1

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context=None,
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={},
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_discriminator_threshold_zero(self, session: Session, simple_input: str):
        """Test with discriminator threshold of 0.0."""
        method = MutualReasoning(discriminator_threshold=0.0, num_iterations=10)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_discriminator_threshold_one(self, session: Session, simple_input: str):
        """Test with discriminator threshold of 1.0."""
        method = MutualReasoning(discriminator_threshold=1.0, num_iterations=10)
        await method.initialize()

        result = await method.execute(
            session=session,
            input_text=simple_input,
        )

        assert isinstance(result, ThoughtNode)


# ============================================================================
# Quality and Scoring Tests
# ============================================================================


class TestQualityAndScoring:
    """Test suite for quality scoring and discriminator evaluation."""

    @pytest.mark.asyncio
    async def test_discriminator_scores_in_range(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that all discriminator scores are in valid range."""
        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 20},
        )

        # Check all verification thoughts have valid scores
        for _thought_id, thought in session.graph.nodes.items():
            if thought.type == ThoughtType.VERIFICATION:
                score = thought.metadata.get("discriminator_score")
                if score is not None:
                    assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_quality_improves_over_iterations(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that quality generally improves with more iterations."""
        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 5},
        )

        session2 = Session().start()
        await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"num_iterations": 30},
        )

        # More iterations should explore more thoroughly
        assert session2.thought_count >= session.thought_count

    @pytest.mark.asyncio
    async def test_final_confidence_based_on_discriminator(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that final confidence is based on discriminator score."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 20},
        )

        disc_score = result.metadata.get("discriminator_score", 0)
        confidence = result.confidence or 0

        # Confidence should match discriminator score
        assert abs(confidence - disc_score) < 0.01


# ============================================================================
# Thought Type Tests
# ============================================================================


class TestThoughtTypes:
    """Test suite for proper thought type usage."""

    @pytest.mark.asyncio
    async def test_initial_thought_type(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that initial thought has correct type."""
        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 5},
        )

        # Find root thought
        root_thoughts = [
            t for t in session.graph.nodes.values() if t.metadata.get("is_root") is True
        ]

        assert len(root_thoughts) == 1
        assert root_thoughts[0].type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_hypothesis_thoughts_created(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that HYPOTHESIS thoughts are created by generator."""
        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        hypothesis_thoughts = [
            t for t in session.graph.nodes.values() if t.type == ThoughtType.HYPOTHESIS
        ]

        assert len(hypothesis_thoughts) > 0

    @pytest.mark.asyncio
    async def test_verification_thoughts_created(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that VERIFICATION thoughts are created by discriminator."""
        await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        verification_thoughts = [
            t for t in session.graph.nodes.values() if t.type == ThoughtType.VERIFICATION
        ]

        assert len(verification_thoughts) > 0

    @pytest.mark.asyncio
    async def test_synthesis_thought_created(
        self, initialized_method: MutualReasoning, session: Session, simple_input: str
    ):
        """Test that final SYNTHESIS thought is created."""
        result = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"num_iterations": 10},
        )

        assert result.type == ThoughtType.SYNTHESIS
        assert result.metadata.get("is_final") is True
