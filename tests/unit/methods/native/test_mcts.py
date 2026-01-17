"""Comprehensive unit tests for Monte Carlo Tree Search (MCTS) reasoning method.

This test suite provides extensive coverage for the MCTS implementation, including:
- Initialization and health checks
- Basic execution flow
- Four MCTS phases (selection, expansion, simulation, backpropagation)
- UCB1 selection algorithm
- Configuration parameters
- Continue reasoning functionality
- Edge cases and error handling
"""

from __future__ import annotations

import math
from uuid import uuid4

import pytest

from reasoning_mcp.methods.native.mcts import MCTS, MCTS_METADATA, MCTSNode
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType


class TestMCTSMetadata:
    """Tests for MCTS method metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert MCTS_METADATA.identifier == MethodIdentifier.MCTS

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert MCTS_METADATA.name == "Monte Carlo Tree Search"

    def test_metadata_description(self):
        """Test that metadata has description."""
        assert len(MCTS_METADATA.description) > 0
        assert "simulation" in MCTS_METADATA.description.lower()

    def test_metadata_supports_branching(self):
        """Test that MCTS supports branching."""
        assert MCTS_METADATA.supports_branching is True

    def test_metadata_complexity(self):
        """Test that MCTS has high complexity."""
        assert MCTS_METADATA.complexity == 8

    def test_metadata_tags(self):
        """Test that MCTS has appropriate tags."""
        assert "mcts" in MCTS_METADATA.tags
        assert "tree" in MCTS_METADATA.tags
        assert "search" in MCTS_METADATA.tags
        assert "simulation" in MCTS_METADATA.tags


class TestMCTSNode:
    """Tests for MCTSNode internal tree representation."""

    def test_node_initialization(self):
        """Test MCTSNode initializes with correct defaults."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
        )
        node = MCTSNode(thought)

        assert node.thought == thought
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.untried_actions == []

    def test_node_initialization_with_actions(self):
        """Test MCTSNode initializes with untried actions."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
        )
        actions = ["action1", "action2", "action3"]
        node = MCTSNode(thought, untried_actions=actions)

        assert node.untried_actions == actions
        assert not node.is_fully_expanded()

    def test_node_is_fully_expanded(self):
        """Test is_fully_expanded returns correct state."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
        )
        node = MCTSNode(thought, untried_actions=["action1"])

        assert not node.is_fully_expanded()

        node.untried_actions = []
        assert node.is_fully_expanded()

    def test_node_is_terminal(self):
        """Test is_terminal checks against max depth."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
            depth=3,
        )
        node = MCTSNode(thought)

        assert not node.is_terminal(max_depth=5)
        assert node.is_terminal(max_depth=3)
        assert node.is_terminal(max_depth=2)

    def test_ucb1_score_unvisited_node(self):
        """Test UCB1 score returns infinity for unvisited nodes."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
        )
        node = MCTSNode(thought)

        assert node.ucb1_score() == float("inf")

    def test_ucb1_score_no_parent(self):
        """Test UCB1 score for root node (no parent)."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test thought",
        )
        node = MCTSNode(thought)
        node.visits = 10
        node.value = 5.0

        # Should return exploitation term only (value/visits)
        expected = 5.0 / 10
        assert node.ucb1_score() == expected

    def test_ucb1_score_with_parent(self):
        """Test UCB1 score calculation with parent."""
        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought)
        parent_node.visits = 100

        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child",
            parent_id=parent_thought.id,
        )
        child_node = MCTSNode(child_thought, parent=parent_node)
        child_node.visits = 10
        child_node.value = 5.0

        # UCB1 = value/visits + C * sqrt(ln(parent_visits) / visits)
        # UCB1 = 5.0/10 + 1.414 * sqrt(ln(100) / 10)
        exploitation = 5.0 / 10
        exploration = 1.414 * math.sqrt(math.log(100) / 10)
        expected = exploitation + exploration

        assert abs(child_node.ucb1_score() - expected) < 0.001

    def test_ucb1_score_custom_exploration_constant(self):
        """Test UCB1 score with custom exploration constant."""
        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought)
        parent_node.visits = 100

        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child",
            parent_id=parent_thought.id,
        )
        child_node = MCTSNode(child_thought, parent=parent_node)
        child_node.visits = 10
        child_node.value = 5.0

        # Use custom exploration constant
        exploration_constant = 2.0
        exploitation = 5.0 / 10
        exploration = exploration_constant * math.sqrt(math.log(100) / 10)
        expected = exploitation + exploration

        assert abs(child_node.ucb1_score(exploration_constant) - expected) < 0.001

    def test_best_child_selection(self):
        """Test best_child selects child with highest UCB1 score."""
        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought)
        parent_node.visits = 100

        # Create children with different visit/value statistics
        child1 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 1",
            ),
            parent=parent_node,
        )
        child1.visits = 10
        child1.value = 3.0

        child2 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 2",
            ),
            parent=parent_node,
        )
        child2.visits = 20
        child2.value = 8.0

        # Unvisited child should have highest UCB1 (infinity)
        child3 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 3",
            ),
            parent=parent_node,
        )

        parent_node.children = [child1, child2, child3]

        # Should select unvisited child (UCB1 = inf)
        best = parent_node.best_child()
        assert best == child3

    def test_best_child_no_children_raises_error(self):
        """Test best_child raises error when no children exist."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test",
        )
        node = MCTSNode(thought)

        with pytest.raises(ValueError, match="No children to select from"):
            node.best_child()

    def test_most_visited_child(self):
        """Test most_visited_child selects child with most visits."""
        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought)

        child1 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 1",
            ),
            parent=parent_node,
        )
        child1.visits = 10

        child2 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 2",
            ),
            parent=parent_node,
        )
        child2.visits = 25  # Most visited

        child3 = MCTSNode(
            ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.MCTS,
                content="Child 3",
            ),
            parent=parent_node,
        )
        child3.visits = 15

        parent_node.children = [child1, child2, child3]

        most_visited = parent_node.most_visited_child()
        assert most_visited == child2

    def test_most_visited_child_no_children_raises_error(self):
        """Test most_visited_child raises error when no children exist."""
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Test",
        )
        node = MCTSNode(thought)

        with pytest.raises(ValueError, match="No children to select from"):
            node.most_visited_child()


class TestMCTSInitialization:
    """Tests for MCTS method initialization."""

    def test_default_initialization(self):
        """Test MCTS initializes with default parameters."""
        mcts = MCTS()

        assert mcts.num_iterations == 50
        assert mcts.max_depth == 4
        assert mcts.exploration_constant == 1.414
        assert mcts.branching_factor == 3
        assert mcts.simulation_depth == 3

    def test_custom_initialization(self):
        """Test MCTS initializes with custom parameters."""
        mcts = MCTS(
            num_iterations=100,
            max_depth=6,
            exploration_constant=2.0,
            branching_factor=5,
            simulation_depth=4,
        )

        assert mcts.num_iterations == 100
        assert mcts.max_depth == 6
        assert mcts.exploration_constant == 2.0
        assert mcts.branching_factor == 5
        assert mcts.simulation_depth == 4

    def test_invalid_num_iterations_raises_error(self):
        """Test that invalid num_iterations raises ValueError."""
        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            MCTS(num_iterations=0)

        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            MCTS(num_iterations=-1)

    def test_invalid_max_depth_raises_error(self):
        """Test that invalid max_depth raises ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            MCTS(max_depth=0)

        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            MCTS(max_depth=-1)

    def test_invalid_exploration_constant_raises_error(self):
        """Test that invalid exploration_constant raises ValueError."""
        with pytest.raises(ValueError, match="exploration_constant must be > 0"):
            MCTS(exploration_constant=0)

        with pytest.raises(ValueError, match="exploration_constant must be > 0"):
            MCTS(exploration_constant=-1.0)

    def test_invalid_branching_factor_raises_error(self):
        """Test that invalid branching_factor raises ValueError."""
        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            MCTS(branching_factor=0)

        with pytest.raises(ValueError, match="branching_factor must be >= 1"):
            MCTS(branching_factor=-1)

    def test_invalid_simulation_depth_raises_error(self):
        """Test that invalid simulation_depth raises ValueError."""
        with pytest.raises(ValueError, match="simulation_depth must be >= 1"):
            MCTS(simulation_depth=0)

        with pytest.raises(ValueError, match="simulation_depth must be >= 1"):
            MCTS(simulation_depth=-1)

    def test_properties(self):
        """Test MCTS properties return correct values."""
        mcts = MCTS()

        assert mcts.identifier == str(MethodIdentifier.MCTS)
        assert mcts.name == "Monte Carlo Tree Search"
        assert len(mcts.description) > 0
        assert mcts.category == str(MCTS_METADATA.category)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize method runs without error."""
        mcts = MCTS()
        await mcts.initialize()
        # No exceptions should be raised

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check returns True."""
        mcts = MCTS()
        result = await mcts.health_check()
        assert result is True


class TestMCTSExecution:
    """Tests for MCTS execute method."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic MCTS execution."""
        mcts = MCTS(num_iterations=10, max_depth=3, branching_factor=2)
        session = Session().start()

        result = await mcts.execute(
            session,
            "What is the best strategy for market entry?",
        )

        # Verify result is a ThoughtNode
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.SYNTHESIS
        assert result.method_id == MethodIdentifier.MCTS

        # Verify session contains thoughts
        assert session.thought_count > 0

        # Verify metrics updated
        assert session.metrics.total_thoughts > 0

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test MCTS execution with custom context parameters."""
        mcts = MCTS()
        session = Session().start()

        context = {
            "num_iterations": 20,
            "max_depth": 5,
            "exploration_constant": 2.0,
            "branching_factor": 4,
            "simulation_depth": 3,
        }

        result = await mcts.execute(
            session,
            "Optimize resource allocation",
            context=context,
        )

        # Verify result
        assert isinstance(result, ThoughtNode)
        assert result.metadata["total_iterations"] == 20

    @pytest.mark.asyncio
    async def test_execute_inactive_session_raises_error(self):
        """Test execute raises error for inactive session."""
        mcts = MCTS()
        session = Session()  # Not started

        with pytest.raises(ValueError, match="Session must be active"):
            await mcts.execute(session, "Test query")

    @pytest.mark.asyncio
    async def test_execute_creates_root_thought(self):
        """Test execute creates root thought with INITIAL type."""
        mcts = MCTS(num_iterations=5)
        session = Session().start()

        await mcts.execute(session, "Test query")

        # Find root thought
        initial_thoughts = session.get_thoughts_by_type(ThoughtType.INITIAL)
        assert len(initial_thoughts) > 0

        root = initial_thoughts[0]
        assert root.type == ThoughtType.INITIAL
        assert root.depth == 0
        assert "MCTS" in root.content

    @pytest.mark.asyncio
    async def test_execute_creates_branches(self):
        """Test execute creates branch thoughts during expansion."""
        mcts = MCTS(num_iterations=10, branching_factor=3)
        session = Session().start()

        await mcts.execute(session, "Test query")

        # Verify branches were created
        branch_thoughts = session.get_thoughts_by_type(ThoughtType.BRANCH)
        assert len(branch_thoughts) > 0
        assert session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_execute_creates_progress_observations(self):
        """Test execute creates progress observation thoughts."""
        mcts = MCTS(num_iterations=25)  # Should create progress at iteration 10, 20
        session = Session().start()

        await mcts.execute(session, "Test query")

        # Find progress thoughts
        observation_thoughts = session.get_thoughts_by_type(ThoughtType.OBSERVATION)
        assert len(observation_thoughts) > 0

        # Verify progress metadata
        progress_thoughts = [t for t in observation_thoughts if t.metadata.get("is_progress")]
        assert len(progress_thoughts) > 0

    @pytest.mark.asyncio
    async def test_execute_creates_synthesis(self):
        """Test execute creates final synthesis thought."""
        mcts = MCTS(num_iterations=10)
        session = Session().start()

        result = await mcts.execute(session, "Test query")

        # Verify synthesis
        assert result.type == ThoughtType.SYNTHESIS
        assert result.metadata.get("is_final") is True
        assert "win_rate" in result.metadata
        assert "total_iterations" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_updates_session_metrics(self):
        """Test execute updates session metrics correctly."""
        mcts = MCTS(num_iterations=15)
        session = Session().start()

        initial_count = session.thought_count
        await mcts.execute(session, "Test query")

        # Verify metrics updated
        assert session.thought_count > initial_count
        assert session.metrics.total_thoughts > 0
        assert session.metrics.average_confidence > 0

    @pytest.mark.asyncio
    async def test_execute_small_tree(self):
        """Test execute with minimal tree (edge case)."""
        mcts = MCTS(num_iterations=1, max_depth=1, branching_factor=1)
        session = Session().start()

        result = await mcts.execute(session, "Simple query")

        # Should still complete successfully
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_execute_deep_tree(self):
        """Test execute with deep tree configuration."""
        mcts = MCTS(num_iterations=20, max_depth=10, branching_factor=2)
        session = Session().start()

        result = await mcts.execute(session, "Complex query")

        # Verify execution completed
        assert isinstance(result, ThoughtNode)
        assert session.thought_count > 0


class TestMCTSPhases:
    """Tests for individual MCTS phases."""

    @pytest.mark.asyncio
    async def test_select_phase(self):
        """Test selection phase chooses appropriate node."""
        mcts = MCTS()
        session = Session().start()

        # Create a simple tree
        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        root_node = MCTSNode(root_thought, untried_actions=["action1"])

        # Select should return root since it's not fully expanded
        selected = await mcts._select(root_node, exploration_constant=1.414, session=session)
        assert selected == root_node

    @pytest.mark.asyncio
    async def test_select_phase_fully_expanded(self):
        """Test selection phase with fully expanded node."""
        mcts = MCTS()
        session = Session().start()

        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        root_node = MCTSNode(root_thought, untried_actions=[])
        root_node.visits = 10

        # Add a child
        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child",
            parent_id=root_thought.id,
        )
        child_node = MCTSNode(child_thought, parent=root_node, untried_actions=["action1"])
        root_node.children = [child_node]

        # Should select child since root is fully expanded
        selected = await mcts._select(root_node, exploration_constant=1.414, session=session)
        assert selected == child_node

    @pytest.mark.asyncio
    async def test_expand_phase(self):
        """Test expansion phase creates new child node."""
        mcts = MCTS(branching_factor=3)
        session = Session().start()

        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought, untried_actions=["action1", "action2"])
        all_thoughts: dict[str, ThoughtNode] = {parent_thought.id: parent_thought}

        # Expand should create a child
        child_node = await mcts._expand(
            session, parent_node, "Test input", branching_factor=3, all_thoughts=all_thoughts
        )

        assert child_node is not None
        assert len(parent_node.untried_actions) == 1  # One action consumed
        assert len(parent_node.children) == 1
        assert child_node.parent == parent_node
        assert session.metrics.branches_created == 1

    @pytest.mark.asyncio
    async def test_expand_phase_no_actions_returns_none(self):
        """Test expansion phase returns None when no actions available."""
        mcts = MCTS()
        session = Session().start()

        parent_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Parent",
        )
        parent_node = MCTSNode(parent_thought, untried_actions=[])
        all_thoughts: dict[str, ThoughtNode] = {}

        result = await mcts._expand(
            session, parent_node, "Test input", branching_factor=3, all_thoughts=all_thoughts
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_simulate_phase(self):
        """Test simulation phase returns value in valid range."""
        mcts = MCTS(simulation_depth=5)
        session = Session().start()

        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Test",
            depth=2,
        )
        node = MCTSNode(thought)

        value = await mcts._simulate(
            node, "Test input", simulation_depth=5, max_depth=10, session=session, iteration=1
        )

        # Value should be in [-1, 1] range
        assert -1.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_simulate_phase_respects_max_depth(self):
        """Test simulation phase stops at max depth."""
        mcts = MCTS()
        session = Session().start()

        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Test",
            depth=5,  # Already at max_depth
        )
        node = MCTSNode(thought)

        # Should still return a value even if at max depth
        value = await mcts._simulate(
            node, "Test input", simulation_depth=3, max_depth=5, session=session, iteration=1
        )
        assert -1.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_backpropagate_phase(self):
        """Test backpropagation updates values up the tree."""
        mcts = MCTS()

        # Create a chain: root -> child -> grandchild
        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        root_node = MCTSNode(root_thought)

        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child",
            parent_id=root_thought.id,
        )
        child_node = MCTSNode(child_thought, parent=root_node)
        root_node.children = [child_node]

        grandchild_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Grandchild",
            parent_id=child_thought.id,
        )
        grandchild_node = MCTSNode(grandchild_thought, parent=child_node)
        child_node.children = [grandchild_node]

        # Backpropagate from grandchild
        test_value = 0.7
        await mcts._backpropagate(grandchild_node, test_value)

        # All nodes should be updated
        assert grandchild_node.visits == 1
        assert grandchild_node.value == test_value
        assert child_node.visits == 1
        assert child_node.value == test_value
        assert root_node.visits == 1
        assert root_node.value == test_value

    @pytest.mark.asyncio
    async def test_backpropagate_accumulates_values(self):
        """Test backpropagation accumulates values over multiple calls."""
        mcts = MCTS()

        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        root_node = MCTSNode(root_thought)

        # Backpropagate multiple times
        await mcts._backpropagate(root_node, 0.5)
        await mcts._backpropagate(root_node, 0.3)
        await mcts._backpropagate(root_node, 0.8)

        assert root_node.visits == 3
        assert root_node.value == 0.5 + 0.3 + 0.8


class TestMCTSHelperMethods:
    """Tests for MCTS helper methods."""

    async def test_generate_action_set(self):
        """Test action set generation."""
        mcts = MCTS()

        actions = await mcts._generate_action_set("Test input", count=5)

        assert len(actions) == 5
        assert all(isinstance(action, str) for action in actions)
        assert all(len(action) > 0 for action in actions)

    async def test_generate_action_set_different_counts(self):
        """Test action set generation with different counts."""
        mcts = MCTS()

        actions1 = await mcts._generate_action_set("Test", count=1)
        assert len(actions1) == 1

        actions3 = await mcts._generate_action_set("Test", count=3)
        assert len(actions3) == 3

        actions10 = await mcts._generate_action_set("Test", count=10)
        assert len(actions10) == 10

    def test_extract_path_single_node(self):
        """Test path extraction from single node."""
        mcts = MCTS()

        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        node = MCTSNode(thought)

        path = mcts._extract_path(node)
        assert path == ["Initial state"]

    def test_extract_path_chain(self):
        """Test path extraction from chain of nodes."""
        mcts = MCTS()

        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content="Root",
        )
        root_node = MCTSNode(root_thought)

        child1_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child 1",
            parent_id=root_thought.id,
            metadata={"action": "Action 1"},
        )
        child1_node = MCTSNode(child1_thought, parent=root_node)

        child2_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content="Child 2",
            parent_id=child1_thought.id,
            metadata={"action": "Action 2"},
        )
        child2_node = MCTSNode(child2_thought, parent=child1_node)

        path = mcts._extract_path(child2_node)
        assert path == ["Action 1", "Action 2"]


class TestMCTSContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_basic(self):
        """Test continue_reasoning creates continuation thought."""
        mcts = MCTS()
        session = Session().start()

        previous_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content="Previous analysis",
            confidence=0.8,
        )
        session.add_thought(previous_thought)

        result = await mcts.continue_reasoning(session, previous_thought)

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.CONTINUATION
        assert result.parent_id == previous_thought.id
        assert result.metadata.get("is_continuation") is True

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self):
        """Test continue_reasoning with guidance parameter."""
        mcts = MCTS()
        session = Session().start()

        previous_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content="Previous analysis",
            confidence=0.8,
        )
        session.add_thought(previous_thought)

        guidance = "Focus on strategic aspects"
        result = await mcts.continue_reasoning(session, previous_thought, guidance=guidance)

        assert result.metadata.get("guidance") == guidance
        assert guidance in result.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_custom_iterations(self):
        """Test continue_reasoning with custom iteration count."""
        mcts = MCTS()
        session = Session().start()

        previous_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content="Previous analysis",
            confidence=0.8,
        )
        session.add_thought(previous_thought)

        context = {"num_iterations": 50}
        result = await mcts.continue_reasoning(session, previous_thought, context=context)

        assert result.metadata.get("additional_iterations") == 50

    @pytest.mark.asyncio
    async def test_continue_reasoning_inactive_session_raises_error(self):
        """Test continue_reasoning raises error for inactive session."""
        mcts = MCTS()
        session = Session()  # Not started

        previous_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content="Previous",
            confidence=0.8,
        )

        with pytest.raises(ValueError, match="Session must be active"):
            await mcts.continue_reasoning(session, previous_thought)

    @pytest.mark.asyncio
    async def test_continue_reasoning_preserves_confidence(self):
        """Test continue_reasoning adjusts confidence appropriately."""
        mcts = MCTS()
        session = Session().start()

        previous_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content="Previous",
            confidence=0.9,
        )
        session.add_thought(previous_thought)

        result = await mcts.continue_reasoning(session, previous_thought)

        # Confidence should be slightly reduced (0.9 * 0.95 = 0.855)
        assert result.confidence < previous_thought.confidence
        assert abs(result.confidence - 0.855) < 0.001


class TestMCTSEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_balanced_tree_exploration(self):
        """Test MCTS handles balanced tree exploration."""
        mcts = MCTS(num_iterations=20, branching_factor=3, max_depth=3)
        session = Session().start()

        result = await mcts.execute(session, "Balanced decision tree")

        # Should complete successfully with balanced exploration
        assert isinstance(result, ThoughtNode)
        assert session.thought_count > 0

    @pytest.mark.asyncio
    async def test_single_iteration(self):
        """Test MCTS with single iteration (minimal execution)."""
        mcts = MCTS(num_iterations=1)
        session = Session().start()

        result = await mcts.execute(session, "Single iteration test")

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_large_branching_factor(self):
        """Test MCTS with large branching factor."""
        mcts = MCTS(num_iterations=15, branching_factor=10)
        session = Session().start()

        result = await mcts.execute(session, "Large branching test")

        # Should handle large branching factor
        assert isinstance(result, ThoughtNode)
        assert session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_zero_depth_simulation(self):
        """Test simulation at terminal nodes (edge case)."""
        mcts = MCTS(simulation_depth=1, max_depth=1)
        session = Session().start()

        result = await mcts.execute(session, "Minimal depth test")

        # Should still produce valid result
        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_metadata_propagation(self):
        """Test metadata is properly propagated through tree."""
        mcts = MCTS(num_iterations=10)
        session = Session().start()

        result = await mcts.execute(session, "Metadata test")

        # Check that result has expected metadata fields
        assert "total_iterations" in result.metadata
        assert "total_nodes" in result.metadata
        assert "best_visits" in result.metadata
        assert "win_rate" in result.metadata

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self):
        """Test win rate calculation is within valid bounds."""
        mcts = MCTS(num_iterations=20)
        session = Session().start()

        result = await mcts.execute(session, "Win rate test")

        win_rate = result.metadata.get("win_rate")
        assert win_rate is not None
        assert 0.0 <= win_rate <= 1.0
        assert result.confidence == win_rate

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test MCTS handles empty input text gracefully."""
        mcts = MCTS(num_iterations=5)
        session = Session().start()

        result = await mcts.execute(session, "")

        # Should still execute successfully
        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_multiple_execute_calls_same_method(self):
        """Test multiple execute calls on same MCTS instance."""
        mcts = MCTS(num_iterations=10)

        session1 = Session().start()
        result1 = await mcts.execute(session1, "First query")

        session2 = Session().start()
        result2 = await mcts.execute(session2, "Second query")

        # Both should succeed independently
        assert isinstance(result1, ThoughtNode)
        assert isinstance(result2, ThoughtNode)
        assert result1.id != result2.id

    @pytest.mark.asyncio
    async def test_thought_depth_increases(self):
        """Test that thought depth increases with tree expansion."""
        mcts = MCTS(num_iterations=20, max_depth=5)
        session = Session().start()

        await mcts.execute(session, "Depth test")

        # Find thoughts at different depths
        all_thoughts = list(session.graph.nodes.values())
        depths = [t.depth for t in all_thoughts]

        # Should have thoughts at depth 0 (root)
        assert 0 in depths
        # Should have thoughts at depth > 0 (children)
        assert any(d > 0 for d in depths)
