"""Tree of Thoughts reasoning method implementation.

This module implements the Tree of Thoughts (ToT) reasoning approach, which explores
multiple solution branches in a tree structure, evaluating and pruning paths based on
their promise to find optimal solutions.

Tree of Thoughts enables:
- Multiple parallel exploration paths
- Branch scoring and evaluation
- Pruning of unpromising paths
- BFS (breadth-first) or DFS (depth-first) exploration strategies
- Systematic search through the solution space

Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
(Yao et al., 2023)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_selection,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext


# Define metadata for Tree of Thoughts method
TREE_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.TREE_OF_THOUGHTS,
    name="Tree of Thoughts",
    description="Explore multiple reasoning paths in a tree structure with branch evaluation and pruning",
    category=MethodCategory.CORE,
    tags=frozenset(
        {
            "tree",
            "branching",
            "search",
            "exploration",
            "pruning",
            "parallel",
            "core",
        }
    ),
    complexity=6,  # High complexity (5-7)
    supports_branching=True,  # Fully supports branching
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,  # Need at least root + branches
    max_thoughts=0,  # Unlimited - depends on tree depth and branching factor
    avg_tokens_per_thought=600,  # Moderate token usage per thought
    best_for=(
        "complex problem solving",
        "decision making with multiple options",
        "optimization problems",
        "strategic planning",
        "multi-step reasoning tasks",
        "exploring solution spaces",
    ),
    not_recommended_for=(
        "simple linear problems",
        "single-path deterministic tasks",
        "very time-sensitive tasks",
        "problems with clear single solution",
    ),
)


class TreeOfThoughts(ReasoningMethodBase):
    """Tree of Thoughts reasoning method implementation.

    This class implements the ReasoningMethod protocol to provide Tree of Thoughts
    reasoning capabilities. It explores multiple solution branches using either
    breadth-first search (BFS) or depth-first search (DFS) strategies, evaluates
    each branch's promise, and prunes low-scoring paths.

    The method maintains a tree structure where each node represents a partial solution
    or reasoning step, and branches represent alternative approaches or decisions.

    Attributes:
        branching_factor: Number of branches to generate at each decision point (default: 3)
        max_depth: Maximum depth to explore in the tree (default: 5)
        min_score_threshold: Minimum score to keep a branch alive (default: 0.3)
        search_strategy: "bfs" for breadth-first or "dfs" for depth-first (default: "bfs")
        top_k_branches: Number of best branches to keep at each level (default: 2)

    Examples:
        Basic usage with BFS:
        >>> tot = TreeOfThoughts()
        >>> session = Session().start()
        >>> await tot.initialize()
        >>> result = await tot.execute(
        ...     session,
        ...     "How can we improve customer satisfaction?",
        ...     context={"search_strategy": "bfs", "branching_factor": 3}
        ... )

        DFS exploration with custom parameters:
        >>> tot = TreeOfThoughts()
        >>> session = Session().start()
        >>> await tot.initialize()
        >>> result = await tot.execute(
        ...     session,
        ...     "Design a sustainable energy solution",
        ...     context={
        ...         "search_strategy": "dfs",
        ...         "branching_factor": 4,
        ...         "max_depth": 6,
        ...         "min_score_threshold": 0.4
        ...     }
        ... )
    """

    def __init__(
        self,
        branching_factor: int = 3,
        max_depth: int = 5,
        min_score_threshold: float = 0.3,
        search_strategy: str = "bfs",
        top_k_branches: int = 2,
        enable_elicitation: bool = True,
    ) -> None:
        """Initialize the Tree of Thoughts method.

        Args:
            branching_factor: Number of branches to generate at each node
            max_depth: Maximum tree depth to explore
            min_score_threshold: Minimum score to keep a branch (0.0-1.0)
            search_strategy: "bfs" or "dfs" exploration strategy
            top_k_branches: Number of best branches to keep at each level
            enable_elicitation: Whether to enable user interaction (default: True)

        Raises:
            ValueError: If parameters are invalid
        """
        if branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {branching_factor}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if not 0.0 <= min_score_threshold <= 1.0:
            raise ValueError(f"min_score_threshold must be 0.0-1.0, got {min_score_threshold}")
        if search_strategy not in ("bfs", "dfs"):
            raise ValueError(f"search_strategy must be 'bfs' or 'dfs', got {search_strategy}")
        if top_k_branches < 1:
            raise ValueError(f"top_k_branches must be >= 1, got {top_k_branches}")

        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.min_score_threshold = min_score_threshold
        self.search_strategy = search_strategy
        self.top_k_branches = top_k_branches
        self.enable_elicitation = enable_elicitation
        self._use_sampling = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.TREE_OF_THOUGHTS)

    @property
    def name(self) -> str:
        """Return the method name."""
        return TREE_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return TREE_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return str(TREE_OF_THOUGHTS_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the Tree of Thoughts method.

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
        """Execute Tree of Thoughts reasoning on the input.

        This method explores multiple solution paths in a tree structure:
        1. Creates root thought analyzing the problem
        2. Generates multiple branches exploring different approaches
        3. Evaluates each branch with a quality score
        4. Prunes low-scoring branches below threshold
        5. Continues exploration using BFS or DFS until max depth
        6. Returns the best leaf node found

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional context with:
                - search_strategy: "bfs" or "dfs" (overrides default)
                - branching_factor: Number of branches per node
                - max_depth: Maximum tree depth
                - min_score_threshold: Minimum branch score to keep
                - top_k_branches: Number of best branches to keep
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            The best ThoughtNode found (highest scoring leaf)

        Raises:
            ValueError: If session is not active
        """
        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Extract context parameters with defaults
        context = context or {}
        strategy = context.get("search_strategy", self.search_strategy)
        branching = context.get("branching_factor", self.branching_factor)
        depth = context.get("max_depth", self.max_depth)
        threshold = context.get("min_score_threshold", self.min_score_threshold)
        top_k = context.get("top_k_branches", self.top_k_branches)

        # Create root thought (use sampling if available)
        if self._use_sampling:
            root_content = await self._sample_root_analysis(input_text, strategy, branching, depth)
        else:
            root_content = f"Analyzing problem: {input_text}\n\nInitiating tree search with {strategy.upper()} strategy, branching factor {branching}, max depth {depth}."

        root = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content=root_content,
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            metadata={
                "strategy": strategy,
                "branching_factor": branching,
                "is_root": True,
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(root)

        # Track all nodes for evaluation
        all_nodes: dict[str, ThoughtNode] = {root.id: root}

        # Perform tree search based on strategy
        if strategy == "bfs":
            best_leaf = await self._breadth_first_search(
                session, root, input_text, branching, depth, threshold, top_k, all_nodes
            )
        else:  # dfs
            best_leaf = await self._depth_first_search(
                session, root, input_text, branching, depth, threshold, all_nodes
            )

        # Create final synthesis thought
        synthesis = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content=f"Tree of Thoughts exploration complete.\n\nBest solution found:\n{best_leaf.content}\n\nScore: {best_leaf.quality_score:.2f}\nDepth: {best_leaf.depth}\nTotal nodes explored: {len(all_nodes)}",
            parent_id=best_leaf.id,
            confidence=best_leaf.confidence,
            quality_score=best_leaf.quality_score,
            depth=best_leaf.depth + 1,
            metadata={
                "is_final": True,
                "total_nodes": len(all_nodes),
                "best_path_depth": best_leaf.depth,
                "strategy": strategy,
            },
        )
        session.add_thought(synthesis)

        return synthesis

    async def _breadth_first_search(
        self,
        session: Session,
        root: ThoughtNode,
        input_text: str,
        branching_factor: int,
        max_depth: int,
        min_threshold: float,
        top_k: int,
        all_nodes: dict[str, ThoughtNode],
    ) -> ThoughtNode:
        """Perform breadth-first search through the thought tree.

        Args:
            session: Current session
            root: Root thought node
            input_text: Original input
            branching_factor: Branches per node
            max_depth: Maximum depth
            min_threshold: Minimum score threshold
            top_k: Number of best branches to keep per level
            all_nodes: Dictionary to track all created nodes

        Returns:
            Best leaf node found
        """
        # Queue of (node, depth) tuples to explore
        from collections import deque

        queue: deque[tuple[ThoughtNode, int]] = deque([(root, 0)])
        best_leaf = root
        best_score = root.quality_score or 0.0

        while queue:
            current, current_depth = queue.popleft()

            # Stop if we've reached max depth
            if current_depth >= max_depth:
                # Check if this is better than current best
                node_score = current.quality_score or 0.0
                if node_score > best_score:
                    best_score = node_score
                    best_leaf = current
                continue

            # Generate branches for this node
            branches = await self._generate_branches(
                session, current, input_text, branching_factor, current_depth + 1, all_nodes
            )

            # Score and filter branches
            scored_branches = [(branch, branch.quality_score or 0.0) for branch in branches]

            # Keep only branches above threshold
            valid_branches = [branch for branch, score in scored_branches if score >= min_threshold]

            # Keep only top-k branches
            valid_branches.sort(key=lambda b: b.quality_score or 0.0, reverse=True)
            top_branches = valid_branches[:top_k]

            # Optional elicitation: ask user which branch to prioritize
            if (
                self.enable_elicitation
                and self._execution_context
                and self._execution_context.ctx
                and len(valid_branches) > 1
                and current_depth % 2 == 0  # Only elicit every other level
            ):
                try:
                    branch_options = [
                        {
                            "id": str(i),
                            "label": f"Branch {i + 1} (score: {b.quality_score:.2f}): {b.content[:80]}...",
                        }
                        for i, b in enumerate(valid_branches[:5])  # Show top 5 options
                    ]
                    elicit_config = ElicitationConfig(
                        timeout=30, required=False, default_on_timeout=None
                    )
                    selection = await elicit_selection(
                        self._execution_context.ctx,
                        f"Tree of Thoughts is exploring depth {current_depth + 1}. Which branch should be prioritized?",
                        branch_options,
                        config=elicit_config,
                    )
                    # Boost the selected branch's position
                    selected_idx = int(selection.selected)
                    if 0 <= selected_idx < len(valid_branches):
                        selected_branch = valid_branches[selected_idx]
                        # Move to front if not already there
                        if selected_branch in top_branches and top_branches[0] != selected_branch:
                            top_branches.remove(selected_branch)
                            top_branches.insert(0, selected_branch)
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="_breadth_first_search",
                        error=str(e),
                    )
                    # Elicitation failed or timed out - continue without it
                except Exception as e:
                    # Log unexpected exceptions and re-raise to avoid masking bugs
                    logger.error(
                        "elicitation_unexpected_error",
                        method="_breadth_first_search",
                        error_type=type(e).__name__,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            # Track pruned branches
            pruned_count = len(branches) - len(top_branches)
            if pruned_count > 0:
                session.metrics.branches_pruned += pruned_count

            # Add top branches to queue
            for branch in top_branches:
                queue.append((branch, current_depth + 1))

            # Update best leaf if we found a better one
            for branch in top_branches:
                branch_score = branch.quality_score or 0.0
                if branch_score > best_score:
                    best_score = branch_score
                    best_leaf = branch

        return best_leaf

    async def _depth_first_search(
        self,
        session: Session,
        root: ThoughtNode,
        input_text: str,
        branching_factor: int,
        max_depth: int,
        min_threshold: float,
        all_nodes: dict[str, ThoughtNode],
    ) -> ThoughtNode:
        """Perform depth-first search through the thought tree.

        Args:
            session: Current session
            root: Root thought node
            input_text: Original input
            branching_factor: Branches per node
            max_depth: Maximum depth
            min_threshold: Minimum score threshold
            all_nodes: Dictionary to track all created nodes

        Returns:
            Best leaf node found
        """
        best_leaf = root
        best_score = root.quality_score or 0.0

        async def dfs_recursive(node: ThoughtNode, depth: int) -> None:
            nonlocal best_leaf, best_score

            # Base case: max depth reached
            if depth >= max_depth:
                node_score = node.quality_score or 0.0
                if node_score > best_score:
                    best_score = node_score
                    best_leaf = node
                return

            # Generate branches
            branches = await self._generate_branches(
                session, node, input_text, branching_factor, depth + 1, all_nodes
            )

            # Filter branches by threshold and sort by score
            valid_branches = [b for b in branches if (b.quality_score or 0.0) >= min_threshold]
            valid_branches.sort(key=lambda b: b.quality_score or 0.0, reverse=True)

            # Track pruned branches
            pruned_count = len(branches) - len(valid_branches)
            if pruned_count > 0:
                session.metrics.branches_pruned += pruned_count

            # Recursively explore each valid branch
            for branch in valid_branches:
                await dfs_recursive(branch, depth + 1)

                # Update best if this branch or its descendants are better
                branch_score = branch.quality_score or 0.0
                if branch_score > best_score:
                    best_score = branch_score
                    best_leaf = branch

        await dfs_recursive(root, 0)
        return best_leaf

    async def _generate_branches(
        self,
        session: Session,
        parent: ThoughtNode,
        input_text: str,
        branching_factor: int,
        depth: int,
        all_nodes: dict[str, ThoughtNode],
    ) -> list[ThoughtNode]:
        """Generate branch thoughts from a parent node.

        Args:
            session: Current session
            parent: Parent thought to branch from
            input_text: Original input text
            branching_factor: Number of branches to create
            depth: Depth of the new branches
            all_nodes: Dictionary to track all nodes

        Returns:
            List of branch ThoughtNodes
        """
        branches: list[ThoughtNode] = []

        # Define different exploration strategies for branches
        strategies = [
            "analytical approach",
            "creative solution",
            "systematic breakdown",
            "alternative perspective",
            "optimization focus",
            "risk mitigation",
            "innovative method",
            "practical implementation",
        ]

        for i in range(branching_factor):
            # Create unique branch ID
            branch_id = f"branch-{parent.id[:8]}-{i}"

            # Select strategy for this branch
            strategy = strategies[i % len(strategies)]

            # Calculate score (simulated - in real implementation would use LLM evaluation)
            # Scores decay with depth and vary by branch
            base_score = 0.7 - (depth * 0.1)
            variation = (hash(branch_id) % 30) / 100.0  # -0.15 to +0.15 variation
            score = max(0.0, min(1.0, base_score + variation))

            # Create branch content (use sampling if available)
            if self._use_sampling:
                content, sampled_score = await self._sample_branch(
                    input_text, parent.content, strategy, i + 1, branching_factor, depth
                )
                # Use sampled score if provided, otherwise use calculated score
                if sampled_score is not None:
                    score = sampled_score
            else:
                content = f"Branch {i + 1}/{branching_factor}: Exploring {strategy}\n\nBuilding on: {parent.content[:100]}...\n\nThis branch investigates: {strategy} for '{input_text}'\n\nKey considerations:\n- Depth {depth} exploration\n- Building on parent insights\n- Evaluating promise: {score:.2f}"

            # Create branch thought
            branch = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                content=content,
                parent_id=parent.id,
                branch_id=branch_id,
                confidence=score,
                quality_score=score,
                depth=depth,
                metadata={
                    "strategy": strategy,
                    "branch_index": i,
                    "branching_factor": branching_factor,
                    "sampled": self._use_sampling,
                },
            )

            session.add_thought(branch)
            all_nodes[branch.id] = branch
            branches.append(branch)

            # Update metrics
            session.metrics.branches_created += 1

        return branches

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

        For Tree of Thoughts, continuing means exploring additional branches
        from the specified thought node.

        Args:
            session: Current session
            previous_thought: Thought to continue from
            guidance: Optional guidance for exploration
            context: Optional context parameters
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            New ThoughtNode continuing exploration
        """
        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        context = context or {}
        branching = context.get("branching_factor", self.branching_factor)

        # Generate continuation content (use sampling if available)
        if self._use_sampling:
            continuation_content = await self._sample_continuation(
                previous_thought.content, guidance, branching
            )
        else:
            continuation_content = f"Continuing exploration from previous thought.\n\nGuidance: {guidance or 'Exploring additional branches'}\n\nGenerating {branching} new exploration paths..."

        # Create continuation thought
        continuation = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content=continuation_content,
            parent_id=previous_thought.id,
            confidence=previous_thought.confidence * 0.9,  # Slight decay
            quality_score=previous_thought.quality_score,
            depth=previous_thought.depth + 1,
            metadata={
                "is_continuation": True,
                "guidance": guidance,
                "sampled": self._use_sampling,
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

    async def _sample_root_analysis(
        self,
        input_text: str,
        strategy: str,
        branching: int,
        depth: int,
    ) -> str:
        """Sample root analysis using LLM.

        Args:
            input_text: The problem to analyze
            strategy: Search strategy (bfs/dfs)
            branching: Branching factor
            depth: Max depth

        Returns:
            Sampled root analysis content
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_root_analysis")

        system_prompt = """You are a reasoning assistant using Tree of Thoughts methodology.
Analyze the problem and prepare for systematic exploration of solution paths.

Your analysis should:
1. Break down the problem into key components
2. Identify potential solution approaches
3. Note critical decision points that warrant branching
4. Establish evaluation criteria for comparing paths"""

        user_prompt = f"""Problem: {input_text}

Search configuration:
- Strategy: {strategy.upper()}
- Branching factor: {branching}
- Max depth: {depth}

Provide a root analysis for the Tree of Thoughts exploration."""

        def fallback() -> str:
            return f"Analyzing problem: {input_text}\n\nInitiating tree search with {strategy.upper()} strategy, branching factor {branching}, max depth {depth}."

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

    async def _sample_branch(
        self,
        input_text: str,
        parent_content: str,
        strategy: str,
        branch_num: int,
        total_branches: int,
        depth: int,
    ) -> tuple[str, float | None]:
        """Sample a branch exploration using LLM.

        Args:
            input_text: Original problem
            parent_content: Parent thought content
            strategy: Exploration strategy for this branch
            branch_num: This branch's number (1-indexed)
            total_branches: Total branches being generated
            depth: Current depth in tree

        Returns:
            Tuple of (content, score) where score may be None
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_branch")

        system_prompt = """You are a reasoning assistant exploring a solution branch in Tree of Thoughts.

Generate a thoughtful exploration of this branch that:
1. Builds on the parent insight
2. Follows the specified approach/strategy
3. Provides concrete reasoning steps
4. Concludes with a self-evaluation score (0.0-1.0) based on promise

End your response with: SCORE: X.XX"""

        user_prompt = f"""Problem: {input_text}

Parent analysis (truncated): {parent_content[:200]}...

Branch {branch_num}/{total_branches}: Explore using "{strategy}" approach
Depth: {depth}

Generate this branch's exploration and rate its promise."""

        def fallback() -> str:
            return f"Branch {branch_num}/{total_branches}: Exploring {strategy}\n\nBuilding on parent insights...\n\nKey considerations:\n- Depth {depth} exploration\n- Building on parent insights"

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.8,  # Higher temperature for branch diversity
            max_tokens=600,
        )

        # Try to extract score from response
        score = None
        if "SCORE:" in content.upper():
            try:
                score_part = content.upper().split("SCORE:")[-1].strip()
                score = float(score_part.split()[0])
                score = max(0.0, min(1.0, score))  # Clamp to valid range
            except (ValueError, IndexError):
                pass

        return content, score

    async def _sample_continuation(
        self,
        previous_content: str,
        guidance: str | None,
        branching: int,
    ) -> str:
        """Sample continuation using LLM.

        Args:
            previous_content: Previous thought content
            guidance: Optional guidance for continuation
            branching: Number of branches to explore

        Returns:
            Sampled continuation content
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_continuation")

        system_prompt = """You are a reasoning assistant continuing Tree of Thoughts exploration.
Build on the previous analysis to extend the search into new territory."""

        user_prompt = f"""Previous exploration: {previous_content[:300]}...

Guidance: {guidance or "Continue exploring additional branches"}

Plan for generating {branching} new exploration paths from this point."""

        def fallback() -> str:
            return f"Continuing exploration from previous thought.\n\nGuidance: {guidance or 'Exploring additional branches'}\n\nGenerating {branching} new exploration paths..."

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=500,
        )


# Export metadata and class
__all__ = [
    "TreeOfThoughts",
    "TREE_OF_THOUGHTS_METADATA",
]
