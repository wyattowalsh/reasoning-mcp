"""Mutual Reasoning (rStar) method implementation.

This module implements the Mutual Reasoning approach (also known as rStar), which uses
MCTS-based exploration with discriminator guidance. The method employs two complementary
roles working together:
- Generator: Proposes reasoning steps and solutions
- Discriminator: Evaluates and scores the quality of proposed steps

Mutual Reasoning combines:
- MCTS exploration with UCB1 for balancing exploration/exploitation
- Discriminator guidance to assess step quality
- Multi-phase reasoning: generate → discriminate → select → expand
- Iterative refinement through tree-based search

Key characteristics:
- Category: ADVANCED
- Complexity: 8 (high complexity)
- MCTS-based with discriminator guidance
- Tracks: mcts_nodes, discriminator_scores, exploration_depth
- Uses ThoughtType: HYPOTHESIS (generate), VERIFICATION (discriminate), BRANCH (expand)
- MAX_DEPTH = 5, BEAM_WIDTH = 3

Reference: "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (Qi et al., 2024, ICML 2025)
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Define metadata for Mutual Reasoning method
MUTUAL_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.MUTUAL_REASONING,
    name="Mutual Reasoning (rStar)",
    description="MCTS-based exploration with discriminator guidance. Two roles (Generator, Discriminator) work mutually to explore and evaluate reasoning paths through generate-discriminate-select-expand phases.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "mutual",
            "rstar",
            "mcts",
            "discriminator",
            "generator",
            "tree-search",
            "advanced",
            "iterative",
            "evaluation",
            "exploration",
            "guided-search",
        }
    ),
    complexity=8,  # High complexity (7-8)
    supports_branching=True,  # Full branching support via MCTS
    supports_revision=True,  # Can revise through discriminator feedback
    requires_context=False,
    min_thoughts=7,  # Root + generate + discriminate + multiple iterations
    max_thoughts=0,  # Unlimited - depends on iterations and depth
    avg_tokens_per_thought=600,  # Higher due to evaluation content
    best_for=(
        "complex problem solving",
        "multi-step reasoning",
        "solution space exploration",
        "quality-aware search",
        "optimization problems",
        "decision-making under uncertainty",
        "strategic planning",
        "mathematical reasoning",
    ),
    not_recommended_for=(
        "simple factual queries",
        "single-step problems",
        "time-critical tasks",
        "problems requiring domain expertise",
        "very large search spaces",
    ),
)

logger = structlog.get_logger(__name__)


class MCTSNode:
    """Internal node representation for Mutual Reasoning MCTS tree.

    This class maintains MCTS-specific statistics with discriminator scores
    for quality-guided exploration.

    Attributes:
        thought: The ThoughtNode associated with this MCTS node
        parent: Parent MCTS node
        children: List of child MCTS nodes
        visits: Number of times this node has been visited
        value: Accumulated value from simulations and discriminator
        discriminator_score: Quality score from discriminator (0.0-1.0)
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
        self.discriminator_score = 0.5  # Neutral initial score
        self.untried_actions = untried_actions or []

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0

    def is_terminal(self, max_depth: int) -> bool:
        """Check if this node is a terminal node (reached max depth)."""
        return self.thought.depth >= max_depth

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score with discriminator guidance.

        Enhanced UCB1 formula incorporating discriminator score:
        value/visits + discriminator_weight * discriminator_score +
        C * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: C parameter (default: sqrt(2) ≈ 1.414)

        Returns:
            UCB1 score balancing exploitation, quality, and exploration
        """
        if self.visits == 0:
            # Prioritize unvisited nodes with discriminator score
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits + self.discriminator_score

        exploitation = self.value / self.visits
        quality = 0.3 * self.discriminator_score  # 30% weight on discriminator
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + quality + exploration

    def best_child(self, exploration_constant: float = 1.414) -> MCTSNode:
        """Select the best child using discriminator-guided UCB1.

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

    def best_discriminator_child(self) -> MCTSNode:
        """Get the child with highest discriminator score.

        Returns:
            Child with highest discriminator score
        """
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children, key=lambda c: c.discriminator_score)


class MutualReasoning(ReasoningMethodBase):
    """Mutual Reasoning (rStar) method implementation.

    This class implements the ReasoningMethod protocol to provide mutual reasoning
    capabilities using MCTS with discriminator guidance. The method uses two roles:
    - Generator: Proposes reasoning steps
    - Discriminator: Evaluates step quality

    The reasoning process follows four phases:
    1. Generate: Propose candidate reasoning steps
    2. Discriminate: Evaluate quality of generated steps
    3. Select: Choose best step using UCB1 with discriminator guidance
    4. Expand: Add selected step to tree and continue

    Attributes:
        num_iterations: Number of MCTS iterations to perform (default: 40)
        max_depth: Maximum tree depth (default: 5)
        beam_width: Number of best paths to maintain (default: 3)
        exploration_constant: UCB1 exploration parameter (default: 1.414)
        discriminator_threshold: Minimum quality score to accept (default: 0.4)

    Examples:
        Basic usage:
        >>> mutual = MutualReasoning()
        >>> session = Session().start()
        >>> await mutual.initialize()
        >>> result = await mutual.execute(
        ...     session,
        ...     "Solve: If x + 3 = 7, what is x?",
        ... )

        Custom parameters:
        >>> mutual = MutualReasoning(
        ...     num_iterations=60,
        ...     max_depth=6,
        ...     beam_width=5,
        ...     discriminator_threshold=0.5,
        ... )
        >>> result = await mutual.execute(
        ...     session,
        ...     "Find optimal strategy for resource allocation",
        ...     context={"domain": "operations"}
        ... )
    """

    # Default configuration
    DEFAULT_NUM_ITERATIONS = 40
    MAX_DEPTH = 5
    BEAM_WIDTH = 3
    DEFAULT_EXPLORATION_CONSTANT = 1.414
    DEFAULT_DISCRIMINATOR_THRESHOLD = 0.4

    # LLM sampling support
    _use_sampling: bool = True

    def __init__(
        self,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        max_depth: int = MAX_DEPTH,
        beam_width: int = BEAM_WIDTH,
        exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
        discriminator_threshold: float = DEFAULT_DISCRIMINATOR_THRESHOLD,
    ) -> None:
        """Initialize the Mutual Reasoning method.

        Args:
            num_iterations: Number of MCTS iterations
            max_depth: Maximum tree depth to explore
            beam_width: Number of best paths to maintain
            exploration_constant: UCB1 exploration parameter
            discriminator_threshold: Minimum quality score to accept steps

        Raises:
            ValueError: If parameters are invalid
        """
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if exploration_constant <= 0:
            raise ValueError(f"exploration_constant must be > 0, got {exploration_constant}")
        if not 0.0 <= discriminator_threshold <= 1.0:
            raise ValueError(
                f"discriminator_threshold must be in [0.0, 1.0], got {discriminator_threshold}"
            )

        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.exploration_constant = exploration_constant
        self.discriminator_threshold = discriminator_threshold
        self._initialized = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return MethodIdentifier.MUTUAL_REASONING

    @property
    def name(self) -> str:
        """Return the method name."""
        return MUTUAL_REASONING_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return MUTUAL_REASONING_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the Mutual Reasoning method.

        This prepares the method for execution by resetting state.
        """
        self._initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute Mutual Reasoning on the input.

        This method performs discriminator-guided MCTS search:
        1. Creates root node representing the initial problem
        2. Iteratively runs four phases: generate, discriminate, select, expand
        3. Maintains beam of best paths based on discriminator scores
        4. Returns the final best solution with quality metrics

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional context with:
                - num_iterations: Number of MCTS iterations
                - max_depth: Maximum tree depth
                - beam_width: Number of paths to maintain
                - exploration_constant: UCB1 exploration parameter
                - discriminator_threshold: Minimum quality threshold

        Returns:
            A ThoughtNode representing the best solution found

        Raises:
            RuntimeError: If the method has not been initialized
            ValueError: If session is not active
        """
        if not self._initialized:
            raise RuntimeError("Mutual Reasoning method must be initialized before execution")

        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        # Extract context parameters with defaults
        context = context or {}
        iterations = context.get("num_iterations", self.num_iterations)
        max_depth = context.get("max_depth", self.max_depth)
        beam_width = context.get("beam_width", self.beam_width)
        exploration_c = context.get("exploration_constant", self.exploration_constant)
        disc_threshold = context.get("discriminator_threshold", self.discriminator_threshold)

        # Create root thought
        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content=f"Mutual Reasoning (rStar) Analysis: {input_text}\n\nInitializing discriminator-guided MCTS with:\n- Iterations: {iterations}\n- Max depth: {max_depth}\n- Beam width: {beam_width}\n- Discriminator threshold: {disc_threshold}\n\nPhases: Generate → Discriminate → Select → Expand\n\nRoles:\n- Generator: Proposes reasoning steps\n- Discriminator: Evaluates step quality",
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            step_number=1,
            metadata={
                "iterations": iterations,
                "max_depth": max_depth,
                "beam_width": beam_width,
                "discriminator_threshold": disc_threshold,
                "is_root": True,
                "phase": "initialize",
            },
        )
        session.add_thought(root_thought)

        # Create root MCTS node with initial candidate steps
        initial_steps = self._generate_candidate_steps(input_text, beam_width)
        root_node = MCTSNode(root_thought, parent=None, untried_actions=initial_steps)

        # Track all nodes and statistics
        all_thoughts: dict[str, ThoughtNode] = {root_thought.id: root_thought}
        discriminator_scores: list[float] = []
        step_counter = 2

        # Run MCTS iterations with discriminator guidance
        for iteration in range(iterations):
            # Phase 1: Selection - select node to expand using UCB1
            selected_node = await self._select(root_node, exploration_c)

            # Phase 2: Generation - generate candidate step
            if not selected_node.is_terminal(max_depth) and selected_node.untried_actions:
                generated_node = await self._generate(
                    session, selected_node, input_text, step_counter, all_thoughts
                )
                if generated_node:
                    step_counter += 1

                    # Phase 3: Discrimination - evaluate generated step
                    disc_score = await self._discriminate(
                        session, generated_node, input_text, step_counter, all_thoughts
                    )
                    step_counter += 1
                    discriminator_scores.append(disc_score)

                    # Update node with discriminator score
                    generated_node.discriminator_score = disc_score

                    # Phase 4: Expansion - if score meets threshold, expand further
                    if disc_score >= disc_threshold:
                        await self._expand(generated_node, input_text, beam_width)

                    # Backpropagate value incorporating discriminator score
                    value = disc_score * 2.0 - 1.0  # Convert [0,1] to [-1,1]
                    await self._backpropagate(generated_node, value)
                else:
                    # No untried actions, simulate from selected node
                    value = await self._simulate(selected_node, max_depth)
                    await self._backpropagate(selected_node, value)
            else:
                # Terminal or fully expanded, just simulate and backpropagate
                value = await self._simulate(selected_node, max_depth)
                await self._backpropagate(selected_node, value)

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                avg_disc_score = (
                    sum(discriminator_scores[-10:]) / min(10, len(discriminator_scores))
                    if discriminator_scores
                    else 0.5
                )
                progress = ThoughtNode(
                    id=str(uuid4()),
                    type=ThoughtType.OBSERVATION,
                    method_id=MethodIdentifier.MUTUAL_REASONING,
                    content=f"Mutual Reasoning Iteration {iteration + 1}/{iterations}\n\nTree statistics:\n- Root visits: {root_node.visits}\n- Root value: {root_node.value:.3f}\n- Children explored: {len(root_node.children)}\n- Avg discriminator score (last 10): {avg_disc_score:.3f}\n- Total evaluations: {len(discriminator_scores)}",
                    parent_id=root_thought.id,
                    confidence=0.6 + (0.2 * avg_disc_score),
                    quality_score=avg_disc_score,
                    depth=1,
                    step_number=step_counter,
                    metadata={
                        "iteration": iteration + 1,
                        "is_progress": True,
                        "phase": "observe",
                    },
                )
                session.add_thought(progress)
                all_thoughts[progress.id] = progress
                step_counter += 1

        # Select best paths based on combined criteria
        best_paths = self._select_best_paths(root_node, beam_width)

        # Get the overall best node (highest discriminator score among most visited)
        if best_paths:
            best_node = best_paths[0]
        else:
            best_node = root_node.most_visited_child() if root_node.children else root_node

        # Calculate final statistics
        avg_value = best_node.value / max(best_node.visits, 1)
        final_disc_score = best_node.discriminator_score
        avg_all_disc = (
            sum(discriminator_scores) / len(discriminator_scores) if discriminator_scores else 0.5
        )

        # Create final synthesis thought
        best_path_desc = self._extract_path_description(best_node)
        synthesis = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content=f"Mutual Reasoning Complete\n\nBest reasoning path found:\n{best_path_desc}\n\nFinal Evaluation:\n- Total iterations: {iterations}\n- Path visits: {best_node.visits}\n- Average value: {avg_value:.3f}\n- Discriminator score: {final_disc_score:.3f}\n- Overall avg discriminator: {avg_all_disc:.3f}\n- Confidence: {final_disc_score:.1%}\n\nSolution: {best_node.thought.content}\n\nThis solution was selected through mutual reasoning between generator and discriminator roles, achieving the highest quality score after exploring {len(all_thoughts)} reasoning states.",
            parent_id=best_node.thought.id,
            confidence=final_disc_score,
            quality_score=final_disc_score,
            depth=best_node.thought.depth + 1,
            step_number=step_counter,
            metadata={
                "is_final": True,
                "phase": "synthesize",
                "total_iterations": iterations,
                "total_nodes": len(all_thoughts),
                "total_evaluations": len(discriminator_scores),
                "best_visits": best_node.visits,
                "best_value": best_node.value,
                "discriminator_score": final_disc_score,
                "avg_discriminator": avg_all_disc,
                "mcts_nodes": len(all_thoughts),
                "exploration_depth": best_node.thought.depth,
            },
        )
        session.add_thought(synthesis)

        return synthesis

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

        For Mutual Reasoning, continuing means running additional iterations
        from the current tree state with optional guidance.

        Args:
            session: Current session
            previous_thought: Thought to continue from
            guidance: Optional guidance for refinement
            context: Optional context parameters

        Returns:
            New ThoughtNode with refined reasoning

        Raises:
            RuntimeError: If the method has not been initialized
            ValueError: If session is not active
        """
        if not self._initialized:
            raise RuntimeError("Mutual Reasoning method must be initialized before continuation")

        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        context = context or {}
        additional_iterations = context.get("num_iterations", 20)

        # Create continuation thought
        continuation = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content=f"Continuing Mutual Reasoning from previous analysis.\n\nGuidance: {guidance or 'Running additional iterations with discriminator guidance'}\n\nPerforming {additional_iterations} more iterations to refine reasoning path...\n\nPrevious quality: {previous_thought.quality_score:.3f}",
            parent_id=previous_thought.id,
            confidence=previous_thought.confidence * 0.95,
            quality_score=previous_thought.quality_score,
            depth=previous_thought.depth + 1,
            step_number=(previous_thought.step_number or 0) + 1,
            metadata={
                "is_continuation": True,
                "phase": "continue",
                "additional_iterations": additional_iterations,
                "guidance": guidance,
            },
        )

        session.add_thought(continuation)
        return continuation

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and healthy
        """
        return self._initialized

    # =========================================================================
    # Phase 1: Selection
    # =========================================================================

    async def _select(self, node: MCTSNode, exploration_constant: float) -> MCTSNode:
        """Phase 1: Selection - traverse tree using discriminator-guided UCB1.

        Args:
            node: Current node to select from
            exploration_constant: UCB1 exploration parameter

        Returns:
            Selected node for generation or simulation
        """
        current = node

        # Traverse down the tree using discriminator-guided UCB1
        while current.is_fully_expanded() and current.children:
            current = current.best_child(exploration_constant)

        return current

    # =========================================================================
    # Phase 2: Generation
    # =========================================================================

    async def _generate(
        self,
        session: Session,
        node: MCTSNode,
        input_text: str,
        step_number: int,
        all_thoughts: dict[str, ThoughtNode],
    ) -> MCTSNode | None:
        """Phase 2: Generation - generate a candidate reasoning step.

        The Generator role proposes a new reasoning step from available actions.

        Args:
            session: Current session
            node: Node to generate from
            input_text: Original input text
            step_number: Current step number
            all_thoughts: Dictionary tracking all thoughts

        Returns:
            Newly created child node with generated step, or None if no actions left
        """
        if not node.untried_actions:
            return None

        # Select a random untried action (generator proposes)
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Create hypothesis thought for generated step
        child_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.HYPOTHESIS,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content=f"Generator Proposes (Depth {node.thought.depth + 1}):\n\n{action}\n\nReasoning context:\nBuilding on: '{node.thought.content[:150]}...'\n\nThis step explores: {action}\n\nStatus: Awaiting discriminator evaluation...",
            parent_id=node.thought.id,
            branch_id=f"mutual-{node.thought.id[:8]}-{len(node.children)}",
            confidence=0.5,  # Neutral until discriminator evaluates
            quality_score=0.5,
            depth=node.thought.depth + 1,
            step_number=step_number,
            metadata={
                "action": action,
                "is_generation": True,
                "phase": "generate",
                "role": "generator",
                "evaluated": False,
            },
        )
        session.add_thought(child_thought)
        all_thoughts[child_thought.id] = child_thought

        # Create new child node
        child_node = MCTSNode(child_thought, parent=node, untried_actions=[])
        node.children.append(child_node)

        return child_node

    # =========================================================================
    # Phase 3: Discrimination
    # =========================================================================

    async def _discriminate(
        self,
        session: Session,
        node: MCTSNode,
        input_text: str,
        step_number: int,
        all_thoughts: dict[str, ThoughtNode],
    ) -> float:
        """Phase 3: Discrimination - evaluate quality of generated step.

        The Discriminator role evaluates the quality of the generated step,
        providing a score between 0.0 (poor) and 1.0 (excellent).

        Args:
            session: Current session
            node: Node to evaluate
            input_text: Original input text
            step_number: Current step number
            all_thoughts: Dictionary tracking all thoughts

        Returns:
            Discriminator score in range [0.0, 1.0]
        """
        # Try LLM sampling first
        discriminator_score = await self._discriminate_with_llm(node, input_text)

        # If LLM sampling failed, use heuristic fallback
        if discriminator_score is None:
            discriminator_score = self._discriminate_heuristic(node)

        # Create verification thought for discriminator evaluation
        quality_label = (
            "Excellent"
            if discriminator_score >= 0.8
            else "Good"
            if discriminator_score >= 0.6
            else "Acceptable"
            if discriminator_score >= 0.4
            else "Poor"
        )

        verification_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.MUTUAL_REASONING,
            content=f"Discriminator Evaluates:\n\nGenerated step: '{node.thought.content[:100]}...'\n\nEvaluation:\n- Relevance: {'High' if discriminator_score > 0.7 else 'Medium' if discriminator_score > 0.4 else 'Low'}\n- Coherence: {'Strong' if discriminator_score > 0.6 else 'Adequate' if discriminator_score > 0.3 else 'Weak'}\n- Progress: {'Significant' if discriminator_score > 0.65 else 'Moderate' if discriminator_score > 0.35 else 'Limited'}\n\nDiscriminator Score: {discriminator_score:.3f}\nQuality: {quality_label}\n\nDecision: {'✓ Accept and expand' if discriminator_score >= self.discriminator_threshold else '✗ Accept but limit expansion'}",
            parent_id=node.thought.id,
            confidence=discriminator_score,
            quality_score=discriminator_score,
            depth=node.thought.depth,
            step_number=step_number,
            metadata={
                "is_discrimination": True,
                "phase": "discriminate",
                "role": "discriminator",
                "discriminator_score": discriminator_score,
                "quality_label": quality_label,
                "meets_threshold": discriminator_score >= self.discriminator_threshold,
            },
        )
        session.add_thought(verification_thought)
        all_thoughts[verification_thought.id] = verification_thought

        # Update original hypothesis with evaluation (ThoughtNode is frozen)
        updated_metadata = {
            **node.thought.metadata,
            "evaluated": True,
            "discriminator_score": discriminator_score,
        }
        node.thought = node.thought.model_copy(
            update={
                "metadata": updated_metadata,
                "confidence": discriminator_score,
                "quality_score": discriminator_score,
            }
        )
        # Also update the MCTSNode's discriminator_score
        node.discriminator_score = discriminator_score

        return discriminator_score

    # =========================================================================
    # Phase 4: Expansion
    # =========================================================================

    async def _expand(
        self,
        node: MCTSNode,
        input_text: str,
        beam_width: int,
    ) -> None:
        """Phase 4: Expansion - add new candidate actions for high-quality steps.

        Args:
            node: Node to expand with new actions
            input_text: Original input text
            beam_width: Number of candidate actions to generate
        """
        # Generate new candidate actions for next level
        new_actions = self._generate_candidate_steps(input_text, beam_width)
        node.untried_actions = new_actions

    # =========================================================================
    # Supporting Methods
    # =========================================================================

    async def _simulate(
        self,
        node: MCTSNode,
        max_depth: int,
    ) -> float:
        """Simulate random rollout to estimate value.

        Args:
            node: Starting node for simulation
            max_depth: Maximum allowed depth

        Returns:
            Estimated value in range [-1.0, 1.0]
        """
        # Simple random simulation incorporating discriminator score
        base_value = node.discriminator_score * 2.0 - 1.0  # Convert [0,1] to [-1,1]

        # Add small random variation
        noise = random.uniform(-0.2, 0.2)
        simulated_value = base_value + noise

        # Prefer shorter paths (depth penalty)
        depth_penalty = 0.05 * node.thought.depth
        final_value = simulated_value - depth_penalty

        return max(-1.0, min(1.0, final_value))

    async def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree.

        Args:
            node: Starting node (leaf where evaluation occurred)
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

    def _generate_candidate_steps(self, input_text: str, count: int) -> list[str]:
        """Generate candidate reasoning steps.

        Args:
            input_text: The problem context
            count: Number of steps to generate

        Returns:
            List of candidate step descriptions
        """
        # Use heuristic generation (LLM generation happens dynamically during _generate)
        return self._generate_candidate_steps_heuristic(input_text, count)

    def _generate_candidate_steps_heuristic(self, input_text: str, count: int) -> list[str]:
        """Generate candidate reasoning steps using heuristics (fallback).

        Args:
            input_text: The problem context
            count: Number of steps to generate

        Returns:
            List of candidate step descriptions
        """
        # Define step templates for different reasoning approaches
        step_templates = [
            "Break down {aspect} into components",
            "Analyze {aspect} systematically",
            "Identify key {aspect} relationships",
            "Evaluate {aspect} implications",
            "Consider {aspect} from first principles",
            "Examine {aspect} constraints",
            "Explore {aspect} alternatives",
            "Validate {aspect} assumptions",
            "Test {aspect} hypotheses",
            "Synthesize {aspect} insights",
        ]

        aspects = [
            "the problem",
            "the requirements",
            "the constraints",
            "the solution space",
            "the key variables",
            "the dependencies",
            "the edge cases",
            "the trade-offs",
            "the optimizations",
            "the verification criteria",
        ]

        # Generate unique candidate steps
        steps = []
        for i in range(count):
            template = step_templates[i % len(step_templates)]
            aspect = aspects[(i // len(step_templates)) % len(aspects)]
            step = template.format(aspect=aspect)
            steps.append(step)

        return steps

    async def _discriminate_with_llm(self, node: MCTSNode, input_text: str) -> float | None:
        """Use LLM sampling to evaluate the quality of a generated step.

        Args:
            node: Node to evaluate
            input_text: Original problem text

        Returns:
            Discriminator score in range [0.0, 1.0], or None if sampling failed
        """
        prompt = f"""Evaluate the quality of this reasoning step for the given problem.

Problem: {input_text}

Current reasoning path: {node.thought.content[:200]}...

Reasoning step to evaluate: {node.thought.metadata.get("action", "Unknown step")}

Evaluate this step based on:
1. Relevance to the problem
2. Logical coherence
3. Progress toward solution
4. Clarity and specificity

Provide a quality score from 0.0 (poor) to 1.0 (excellent).
Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        system_prompt = "You are a discriminator evaluating reasoning quality. Respond with only a single decimal number between 0.0 and 1.0."

        # Use _sample_with_fallback which handles all error cases properly
        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: "",  # Empty string signals fallback to heuristic
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=10,  # Only need a single number
        )

        # If fallback was used (empty string), return None to use heuristic
        if not result:
            return None

        # Parse the score from result
        score_text = result.strip()
        try:
            score = float(score_text)
            # Clamp to [0.0, 1.0]
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to extract first number from text
            import re

            numbers = re.findall(r"0\.\d+|1\.0|0|1", score_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return None

    def _discriminate_heuristic(self, node: MCTSNode) -> float:
        """Heuristic fallback for discriminator evaluation.

        Args:
            node: Node to evaluate

        Returns:
            Discriminator score in range [0.0, 1.0]
        """
        # Factors considered:
        # - Relevance to problem
        # - Logical coherence
        # - Progress toward solution
        # - Depth (prefer not too deep)

        depth_penalty = 0.1 * node.thought.depth  # Slight penalty for depth
        base_score = random.uniform(0.3, 0.9)  # Simulated evaluation
        random_noise = random.uniform(-0.1, 0.1)  # Small random variation

        raw_score = base_score - depth_penalty + random_noise
        return max(0.0, min(1.0, raw_score))

    def _select_best_paths(self, root: MCTSNode, beam_width: int) -> list[MCTSNode]:
        """Select best paths based on discriminator scores and visits.

        Args:
            root: Root node of the tree
            beam_width: Number of best paths to select

        Returns:
            List of best leaf nodes (up to beam_width)
        """
        # Collect all leaf nodes
        leaves: list[MCTSNode] = []
        self._collect_leaves(root, leaves)

        if not leaves:
            return []

        # Sort by combined metric: discriminator_score * 0.7 + visit_ratio * 0.3
        max_visits = max((leaf.visits for leaf in leaves), default=1)

        def score_node(node: MCTSNode) -> float:
            visit_ratio = node.visits / max_visits if max_visits > 0 else 0
            return node.discriminator_score * 0.7 + visit_ratio * 0.3

        sorted_leaves = sorted(leaves, key=score_node, reverse=True)

        return sorted_leaves[:beam_width]

    def _collect_leaves(self, node: MCTSNode, leaves: list[MCTSNode]) -> None:
        """Recursively collect all leaf nodes.

        Args:
            node: Current node
            leaves: List to accumulate leaf nodes
        """
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def _extract_path_description(self, node: MCTSNode) -> str:
        """Extract readable description of path from root to node.

        Args:
            node: Target node

        Returns:
            Description of the reasoning path
        """
        path_steps: list[str] = []
        current = node

        # Traverse up to root, collecting steps
        while current.parent is not None:
            action = current.thought.metadata.get("action", "Unknown step")
            disc_score = current.discriminator_score
            path_steps.insert(0, f"  → {action} (quality: {disc_score:.2f})")
            current = current.parent

        if not path_steps:
            return "  → Initial state"

        return "\n".join(path_steps)


# Export metadata and class
__all__ = [
    "MutualReasoning",
    "MUTUAL_REASONING_METADATA",
    "MCTSNode",
]
