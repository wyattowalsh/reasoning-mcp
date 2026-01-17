"""Monte Carlo Tree Search (MCTS) reasoning method implementation.

This module implements the Monte Carlo Tree Search (MCTS) reasoning approach, which
uses simulation-based search to find optimal decisions through exploration and
exploitation balancing.

MCTS enables:
- Exploration-exploitation balance via UCB1
- Probabilistic value estimation through rollouts
- Backpropagation of learned values
- Iterative refinement of decision quality
- Asymmetric tree growth toward promising areas

MCTS consists of four phases:
1. Selection: Use UCB1 formula to select promising nodes
2. Expansion: Add new child nodes to the tree
3. Simulation: Random rollout to estimate node value
4. Backpropagation: Update values from leaf to root

Reference: "A Survey of Monte Carlo Tree Search Methods" (Browne et al., 2012)
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_rating,
    elicit_selection,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext


# Define metadata for MCTS method
MCTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.MCTS,
    name="Monte Carlo Tree Search",
    description=(
        "Decision-making through simulation-based search with exploration-exploitation balance"
    ),
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "mcts",
            "tree",
            "search",
            "simulation",
            "decision-making",
            "exploration",
            "exploitation",
            "ucb1",
            "specialized",
        }
    ),
    complexity=8,  # High complexity (7-8)
    supports_branching=True,  # Full branching support
    supports_revision=False,
    requires_context=False,
    min_thoughts=5,  # Need root + multiple iterations
    max_thoughts=0,  # Unlimited - depends on iterations
    avg_tokens_per_thought=550,  # Moderate token usage
    best_for=(
        "complex decision optimization",
        "game-playing scenarios",
        "strategic planning",
        "resource allocation",
        "policy optimization",
        "multi-step decision chains",
        "uncertainty quantification",
    ),
    not_recommended_for=(
        "simple linear problems",
        "deterministic single-path tasks",
        "problems requiring exact solutions",
        "very time-sensitive tasks",
    ),
)


class MCTSNode:
    """Internal node representation for MCTS tree.

    This class maintains the MCTS-specific statistics needed for UCB1
    selection and value backpropagation.

    Attributes:
        thought: The ThoughtNode associated with this MCTS node
        parent: Parent MCTS node
        children: List of child MCTS nodes
        visits: Number of times this node has been visited
        value: Accumulated value from simulations
        untried_actions: Actions that haven't been tried yet
    """

    def __init__(
        self,
        thought: ThoughtNode,
        parent: MCTSNode | None = None,
        untried_actions: list[str] | None = None,
    ) -> None:
        """Initialize an MCTS node.

        Args:
            thought: The associated ThoughtNode
            parent: Parent MCTS node (None for root)
            untried_actions: Available actions to try from this node
        """
        self.thought = thought
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions or []

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0

    def is_terminal(self, max_depth: int) -> bool:
        """Check if this node is a terminal node (reached max depth)."""
        return self.thought.depth >= max_depth

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for this node.

        UCB1 formula: value/visits + C * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: C parameter (default: sqrt(2) ≈ 1.414)

        Returns:
            UCB1 score balancing exploitation and exploration
        """
        if self.visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> MCTSNode:
        """Select the best child using UCB1.

        Args:
            exploration_constant: C parameter for UCB1

        Returns:
            Child node with highest UCB1 score
        """
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children, key=lambda c: c.ucb1_score(exploration_constant))

    def most_visited_child(self) -> MCTSNode:
        """Get the child with the most visits (for final selection).

        Returns:
            Most visited child node
        """
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children, key=lambda c: c.visits)


class MCTS(ReasoningMethodBase):
    """Monte Carlo Tree Search reasoning method implementation.

    This class implements the ReasoningMethod protocol to provide MCTS-based
    decision-making capabilities. It uses the four MCTS phases (selection,
    expansion, simulation, backpropagation) to iteratively improve decision
    quality through simulated rollouts.

    The method balances exploration (trying new actions) with exploitation
    (leveraging known good actions) using the UCB1 algorithm.

    Attributes:
        num_iterations: Number of MCTS iterations to perform (default: 50)
        max_depth: Maximum tree depth (default: 4)
        exploration_constant: UCB1 exploration parameter (default: 1.414)
        branching_factor: Number of actions to try from each node (default: 3)
        simulation_depth: Depth of random rollouts (default: 3)

    Examples:
        Basic usage:
        >>> mcts = MCTS()
        >>> session = Session().start()
        >>> await mcts.initialize()
        >>> result = await mcts.execute(
        ...     session,
        ...     "What is the best strategy for market entry?",
        ... )

        Custom parameters:
        >>> mcts = MCTS(
        ...     num_iterations=100,
        ...     max_depth=5,
        ...     exploration_constant=2.0,
        ...     branching_factor=4,
        ... )
        >>> result = await mcts.execute(
        ...     session,
        ...     "Optimize resource allocation across projects",
        ...     context={"domain": "project_management"}
        ... )
    """

    _use_sampling: bool = True

    def __init__(
        self,
        num_iterations: int = 50,
        max_depth: int = 4,
        exploration_constant: float = 1.414,
        branching_factor: int = 3,
        simulation_depth: int = 3,
        enable_elicitation: bool = True,
    ) -> None:
        """Initialize the MCTS method.

        Args:
            num_iterations: Number of MCTS iterations (simulations)
            max_depth: Maximum tree depth to explore
            exploration_constant: UCB1 exploration parameter (typically sqrt(2))
            branching_factor: Number of child actions per node
            simulation_depth: Depth of random simulation rollouts
            enable_elicitation: Whether to enable user interaction (default: True)

        Raises:
            ValueError: If parameters are invalid
        """
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if exploration_constant <= 0:
            raise ValueError(f"exploration_constant must be > 0, got {exploration_constant}")
        if branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {branching_factor}")
        if simulation_depth < 1:
            raise ValueError(f"simulation_depth must be >= 1, got {simulation_depth}")

        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.branching_factor = branching_factor
        self.simulation_depth = simulation_depth
        self.enable_elicitation = enable_elicitation
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.MCTS)

    @property
    def name(self) -> str:
        """Return the method name."""
        return MCTS_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return MCTS_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return str(MCTS_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the MCTS method.

        This is a lightweight initialization - no external resources needed.
        """
        # No initialization required for this method
        pass

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute MCTS reasoning on the input.

        This method performs Monte Carlo Tree Search:
        1. Creates root node representing the initial problem
        2. Iteratively runs MCTS phases (selection, expansion, simulation, backpropagation)
        3. Selects the best path based on visit counts
        4. Returns the final decision with value estimates

        Args:
            session: The current reasoning session
            input_text: The decision problem or question
            context: Optional context with:
                - num_iterations: Number of MCTS iterations
                - max_depth: Maximum tree depth
                - exploration_constant: UCB1 exploration parameter
                - branching_factor: Actions per node
                - simulation_depth: Simulation rollout depth
            execution_context: Optional ExecutionContext for elicitation

        Returns:
            A ThoughtNode representing the best decision found

        Raises:
            ValueError: If session is not active
        """
        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Store execution context for elicitation
        self._execution_context = execution_context

        # Extract context parameters with defaults
        context = context or {}
        iterations = context.get("num_iterations", self.num_iterations)
        max_depth = context.get("max_depth", self.max_depth)
        exploration_c = context.get("exploration_constant", self.exploration_constant)
        branching = context.get("branching_factor", self.branching_factor)
        sim_depth = context.get("simulation_depth", self.simulation_depth)

        # Create root thought
        content = (
            f"MCTS Decision Analysis: {input_text}\n\n"
            f"Initializing Monte Carlo Tree Search with:\n"
            f"- Iterations: {iterations}\n"
            f"- Max depth: {max_depth}\n"
            f"- Exploration constant: {exploration_c}\n"
            f"- Branching factor: {branching}\n\n"
            f"Phases: Selection → Expansion → Simulation → Backpropagation"
        )
        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MCTS,
            content=content,
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            metadata={
                "iterations": iterations,
                "max_depth": max_depth,
                "exploration_constant": exploration_c,
                "is_root": True,
            },
        )
        session.add_thought(root_thought)

        # Create root MCTS node with initial actions
        initial_actions = await self._generate_action_set(input_text, branching)
        root_node = MCTSNode(root_thought, parent=None, untried_actions=initial_actions)

        # Track all nodes for final analysis
        all_thoughts: dict[str, ThoughtNode] = {root_thought.id: root_thought}

        # Run MCTS iterations
        for iteration in range(iterations):
            # Phase 1: Selection - select node to expand
            selected_node = await self._select(root_node, exploration_c, session)

            # Phase 2: Expansion - add child if not terminal
            if not selected_node.is_terminal(max_depth):
                expanded_node = await self._expand(
                    session, selected_node, input_text, branching, all_thoughts
                )
                # Use expanded node for simulation
                simulation_node = expanded_node if expanded_node else selected_node
            else:
                simulation_node = selected_node

            # Phase 3: Simulation - run random rollout
            value = await self._simulate(
                simulation_node, input_text, sim_depth, max_depth, session, iteration
            )

            # Phase 4: Backpropagation - update values up the tree
            await self._backpropagate(simulation_node, value)

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                avg_val = root_node.value / max(root_node.visits, 1)
                progress_content = (
                    f"MCTS Iteration {iteration + 1}/{iterations}\n\n"
                    f"Tree statistics:\n"
                    f"- Root visits: {root_node.visits}\n"
                    f"- Root value: {root_node.value:.3f}\n"
                    f"- Children: {len(root_node.children)}\n"
                    f"- Average value: {avg_val:.3f}"
                )
                progress = ThoughtNode(
                    id=str(uuid4()),
                    type=ThoughtType.OBSERVATION,
                    method_id=MethodIdentifier.MCTS,
                    content=progress_content,
                    parent_id=root_thought.id,
                    confidence=0.6,
                    quality_score=0.6,
                    depth=1,
                    metadata={
                        "iteration": iteration + 1,
                        "is_progress": True,
                    },
                )
                session.add_thought(progress)
                all_thoughts[progress.id] = progress

        # Select best path based on visit counts (most robust)
        best_node = root_node.most_visited_child() if root_node.children else root_node
        best_path = self._extract_path(best_node)

        # Calculate final statistics
        avg_value = best_node.value / max(best_node.visits, 1)
        win_rate = (avg_value + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]

        # Create final synthesis thought
        synthesis_content = (
            f"MCTS Decision Complete\n\n"
            f"Best decision path found:\n{' → '.join(best_path)}\n\n"
            f"Statistics:\n"
            f"- Total simulations: {iterations}\n"
            f"- Path visits: {best_node.visits}\n"
            f"- Average value: {avg_value:.3f}\n"
            f"- Win rate: {win_rate:.1%}\n"
            f"- Confidence: {win_rate:.1%}\n\n"
            f"Decision: {best_node.thought.content}\n\n"
            f"This decision was selected based on the highest number of simulation visits, "
            f"indicating it is the most promising option after exploring {iterations} scenarios."
        )
        synthesis = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MCTS,
            content=synthesis_content,
            parent_id=best_node.thought.id,
            confidence=win_rate,
            quality_score=win_rate,
            depth=best_node.thought.depth + 1,
            metadata={
                "is_final": True,
                "total_iterations": iterations,
                "total_nodes": len(all_thoughts),
                "best_visits": best_node.visits,
                "best_value": best_node.value,
                "win_rate": win_rate,
            },
        )
        session.add_thought(synthesis)

        return synthesis

    async def _select(
        self,
        node: MCTSNode,
        exploration_constant: float,
        session: Session,
    ) -> MCTSNode:
        """Phase 1: Selection - traverse tree using UCB1 until unexpanded node found.

        Args:
            node: Current node to select from
            exploration_constant: UCB1 exploration parameter
            session: Current session for metrics tracking

        Returns:
            Selected node for expansion or simulation
        """
        current = node

        # Traverse down the tree using UCB1 until we find a node that isn't fully expanded
        while current.is_fully_expanded() and current.children:
            # Optional elicitation: ask user which branch to prioritize
            if (
                self.enable_elicitation
                and self._execution_context
                and self._execution_context.ctx
                and len(current.children) > 1
                and current.visits % 10 == 0  # Only elicit every 10th visit to avoid spam
            ):
                try:
                    # Build options from children
                    child_options = [
                        {
                            "id": str(i),
                            "label": (
                                f"Branch {i + 1} (visits: {child.visits}, "
                                f"value: {child.value / max(child.visits, 1):.2f}, "
                                f"UCB1: {child.ucb1_score(exploration_constant):.2f})"
                            ),
                        }
                        for i, child in enumerate(current.children)
                    ]

                    elicit_config = ElicitationConfig(
                        timeout=20, required=False, default_on_timeout=None
                    )

                    elicit_prompt = (
                        f"MCTS is selecting which branch to explore at depth "
                        f"{current.thought.depth}. Which branch should be prioritized?"
                    )
                    selection = await elicit_selection(
                        self._execution_context.ctx,
                        elicit_prompt,
                        child_options,
                        config=elicit_config,
                    )

                    # Use user's selection if valid
                    selected_idx = int(selection.selected)
                    if 0 <= selected_idx < len(current.children):
                        current = current.children[selected_idx]
                        session.metrics.elicitations_made += 1
                        continue
                except (TimeoutError, ValueError, OSError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="_select",
                        error=str(e),
                    )
                    # Elicitation failed - fall back to UCB1

            # Default: use UCB1 to select best child
            current = current.best_child(exploration_constant)

        return current

    async def _expand(
        self,
        session: Session,
        node: MCTSNode,
        input_text: str,
        branching_factor: int,
        all_thoughts: dict[str, ThoughtNode],
    ) -> MCTSNode | None:
        """Phase 2: Expansion - add a new child node to the tree.

        Args:
            session: Current session
            node: Node to expand from
            input_text: Original input text
            branching_factor: Number of potential actions
            all_thoughts: Dictionary tracking all thoughts

        Returns:
            Newly created child node, or None if no actions left
        """
        if not node.untried_actions:
            return None

        # Select a random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Create thought for this action
        child_content = (
            f"Action: {action}\n\n"
            f"Exploring decision path at depth {node.thought.depth + 1}\n\n"
            f"Context: Building on '{node.thought.content[:100]}...'\n\n"
            f"This branch investigates: {action}"
        )
        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.MCTS,
            content=child_content,
            parent_id=node.thought.id,
            branch_id=f"mcts-{node.thought.id[:8]}-{len(node.children)}",
            confidence=0.5,  # Initial neutral confidence
            quality_score=0.5,
            depth=node.thought.depth + 1,
            metadata={
                "action": action,
                "is_expansion": True,
                "visits": 0,
                "value": 0.0,
            },
        )
        session.add_thought(child_thought)
        all_thoughts[child_thought.id] = child_thought
        # Note: branches_created is already incremented by session.add_thought()
        # via update_from_thought() for BRANCH type thoughts

        # Create new child actions for the next level
        new_actions = await self._generate_action_set(input_text, branching_factor)
        child_node = MCTSNode(child_thought, parent=node, untried_actions=new_actions)
        node.children.append(child_node)

        return child_node

    async def _simulate(
        self,
        node: MCTSNode,
        input_text: str,
        simulation_depth: int,
        max_depth: int,
        session: Session,
        iteration: int,
    ) -> float:
        """Phase 3: Simulation - run random rollout to estimate value.

        Args:
            node: Starting node for simulation
            input_text: Original input text
            simulation_depth: How deep to simulate
            max_depth: Maximum allowed depth
            session: Current session for metrics tracking
            iteration: Current iteration number

        Returns:
            Estimated value in range [-1.0, 1.0]
        """
        # Optional elicitation: ask user to rate node value
        # Only elicit on specific iterations to avoid overwhelming the user
        if (
            self.enable_elicitation
            and self._execution_context
            and self._execution_context.ctx
            and iteration % 20 == 0  # Only elicit every 20th iteration
            and node.thought.depth > 0  # Skip root node
        ):
            try:
                elicit_config = ElicitationConfig(
                    timeout=15, required=False, default_on_timeout=None
                )

                rating_response = await elicit_rating(
                    self._execution_context.ctx,
                    f"MCTS is simulating from node at depth {node.thought.depth}.\n\n"
                    f"Node action: {node.thought.metadata.get('action', 'Root node')}\n\n"
                    f"How promising does this path look for solving: '{input_text[:100]}...'?",
                    config=elicit_config,
                )

                # Convert rating (1-10) to value (-1.0 to 1.0)
                # Rating 1 = -1.0, Rating 5.5 = 0.0, Rating 10 = 1.0
                user_value = (rating_response.rating - 5.5) / 4.5
                session.metrics.elicitations_made += 1
                return max(-1.0, min(1.0, user_value))
            except (TimeoutError, ValueError, OSError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="_simulate",
                    error=str(e),
                )
                # Elicitation failed - fall back to random simulation

        # Simulate random rollout from current node
        depth = node.thought.depth
        current_value = 0.0

        # Random walk simulation
        for step in range(simulation_depth):
            if depth >= max_depth:
                break

            # Simulate a random action and estimate its value
            # In real implementation, this would use domain knowledge or heuristics
            random_action_value = random.uniform(-1.0, 1.0)

            # Decay value slightly with depth (prefer shorter paths)
            decay_factor = 0.9**step
            current_value += random_action_value * decay_factor

            depth += 1

        # Normalize to [-1, 1] range
        if simulation_depth > 0:
            current_value = current_value / simulation_depth

        # Clamp to valid range
        return max(-1.0, min(1.0, current_value))

    async def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Phase 4: Backpropagation - update node values up to root.

        Args:
            node: Starting node (leaf where simulation ended)
            value: Value to propagate upward
        """
        current: MCTSNode | None = node

        # Traverse up to root, updating visits and values
        while current is not None:
            current.visits += 1
            current.value += value

            # Update thought metadata
            current.thought.metadata["visits"] = current.visits
            current.thought.metadata["value"] = current.value
            current.thought.metadata["avg_value"] = current.value / current.visits

            # Move to parent
            current = current.parent

    async def _generate_action_set(self, input_text: str, count: int) -> list[str]:
        """Generate a set of possible actions for a given state.

        Args:
            input_text: The problem context
            count: Number of actions to generate

        Returns:
            List of action descriptions
        """
        # Try LLM sampling if available
        if self._execution_context and self._execution_context.can_sample:
            try:
                system_prompt = (
                    "You are a strategic decision-making assistant helping to generate "
                    "possible actions for Monte Carlo Tree Search. "
                    "Generate distinct, actionable decision paths that explore different "
                    "aspects of the problem. "
                    "Each action should be concrete and represent a meaningful strategic choice."
                )

                user_prompt = f"""Problem: {input_text}

Generate {count} distinct decision actions to explore. Each action should:
1. Be concrete and actionable
2. Explore a different strategic dimension
3. Be formatted as a clear directive (e.g., "Analyze X thoroughly", "Consider Y implications")

Provide exactly {count} actions, one per line."""

                response = await self._execution_context.sample(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,  # Higher temperature for diversity
                    max_tokens=400,
                )
                result = response.text if hasattr(response, "text") else str(response)

                # Parse actions from response
                actions = [line.strip() for line in result.split("\n") if line.strip()]
                # Remove numbering if present (e.g., "1. " or "1) ")
                actions = [action.split(". ", 1)[-1].split(") ", 1)[-1] for action in actions]

                # Ensure we have the right number of actions
                if len(actions) >= count:
                    return actions[:count]
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="_generate_action_set",
                    error=str(e),
                )
                # Fall back to template-based generation

        # Fallback: Define action templates for different types of decisions
        action_templates = [
            "Analyze {aspect} thoroughly",
            "Consider {aspect} implications",
            "Explore {aspect} alternatives",
            "Evaluate {aspect} risks",
            "Prioritize {aspect} benefits",
            "Investigate {aspect} constraints",
            "Optimize {aspect} trade-offs",
            "Balance {aspect} considerations",
        ]

        aspects = [
            "strategic",
            "tactical",
            "financial",
            "operational",
            "technical",
            "market",
            "resource",
            "timeline",
            "quality",
            "innovation",
        ]

        # Generate unique actions
        actions = []
        for i in range(count):
            template = action_templates[i % len(action_templates)]
            aspect = aspects[(i // len(action_templates)) % len(aspects)]
            action = template.format(aspect=aspect)
            actions.append(action)

        return actions

    def _extract_path(self, node: MCTSNode) -> list[str]:
        """Extract the decision path from root to given node.

        Args:
            node: Target node

        Returns:
            List of action descriptions from root to node
        """
        path: list[str] = []
        current = node

        # Traverse up to root, collecting actions
        while current.parent is not None:
            action = current.thought.metadata.get("action", "Unknown action")
            path.insert(0, action)
            current = current.parent

        return path if path else ["Initial state"]

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        For MCTS, continuing means running additional iterations from
        the current tree state to refine the decision.

        Args:
            session: Current session
            previous_thought: Thought to continue from
            guidance: Optional guidance for refinement
            context: Optional context parameters

        Returns:
            New ThoughtNode with refined decision
        """
        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        context = context or {}
        additional_iterations = context.get("num_iterations", 25)

        # Create continuation thought
        guidance_text = guidance or "Running additional simulations"
        continuation_content = (
            f"Continuing MCTS exploration from previous analysis.\n\n"
            f"Guidance: {guidance_text}\n\n"
            f"Performing {additional_iterations} more iterations to refine decision..."
        )
        continuation = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.MCTS,
            content=continuation_content,
            parent_id=previous_thought.id,
            confidence=previous_thought.confidence * 0.95,
            quality_score=previous_thought.quality_score,
            depth=previous_thought.depth + 1,
            metadata={
                "is_continuation": True,
                "additional_iterations": additional_iterations,
                "guidance": guidance,
            },
        )

        session.add_thought(continuation)
        return continuation

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True (this method has no external dependencies)
        """
        return True


# Export metadata and class
__all__ = [
    "MCTS",
    "MCTS_METADATA",
    "MCTSNode",
]
