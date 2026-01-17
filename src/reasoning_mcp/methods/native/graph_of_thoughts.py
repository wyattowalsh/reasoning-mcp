"""Graph of Thoughts reasoning method implementation.

This module implements the Graph of Thoughts (GoT) reasoning approach, which uses
a Directed Acyclic Graph (DAG) structure to represent reasoning paths. Unlike tree-based
approaches where each node has exactly one parent, GoT allows thoughts to have multiple
parents, enabling path merging and complex dependency relationships.

Graph of Thoughts enables:
- DAG-based reasoning (multiple parents per node)
- Path convergence and merging
- Complex dependency tracking between thoughts
- Multiple reasoning paths leading to shared conclusions
- Cross-path information integration
- More flexible reasoning structures than trees

Reference: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
(Besta et al., 2023)
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


# Define metadata for Graph of Thoughts method
GRAPH_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.GRAPH_OF_THOUGHTS,
    name="Graph of Thoughts",
    description="DAG-based reasoning with multiple parents, path merging, and complex dependencies",
    category=MethodCategory.CORE,
    tags=frozenset(
        {
            "graph",
            "dag",
            "branching",
            "merging",
            "convergence",
            "dependencies",
            "parallel",
            "core",
        }
    ),
    complexity=7,  # High complexity (6-8) - more complex than tree
    supports_branching=True,  # Fully supports branching
    supports_revision=True,  # Supports revision through merging
    requires_context=False,
    min_thoughts=4,  # Need at least root + branches + merge
    max_thoughts=0,  # Unlimited - depends on graph complexity
    avg_tokens_per_thought=650,  # Slightly higher than tree due to merging
    best_for=(
        "multi-path reasoning problems",
        "problems requiring information synthesis",
        "complex dependency tracking",
        "convergent analysis from multiple angles",
        "collaborative reasoning scenarios",
        "problems with multiple valid approaches that converge",
    ),
    not_recommended_for=(
        "simple sequential problems",
        "strictly hierarchical tasks",
        "single-perspective analysis",
        "time-critical simple decisions",
    ),
)


class GraphOfThoughts(ReasoningMethodBase):
    """Graph of Thoughts reasoning method implementation.

    This class implements the ReasoningMethod protocol to provide Graph of Thoughts
    reasoning capabilities. Unlike tree-based methods, it uses a DAG structure where
    thoughts can have multiple parents, allowing for path merging and convergence.

    The method maintains a directed acyclic graph where:
    - Each node represents a thought or reasoning step
    - Edges represent dependencies or derivations
    - Nodes can have multiple parents (converging paths)
    - Nodes can have multiple children (diverging paths)
    - Paths can merge to create unified conclusions

    Attributes:
        branching_factor: Number of initial branches to explore (default: 3)
        max_depth: Maximum depth to explore in the graph (default: 6)
        min_merge_score: Minimum combined score to merge paths (default: 0.6)
        enable_convergence: Whether to enable path convergence/merging (default: True)
        convergence_threshold: Similarity threshold for merging (default: 0.7)

    Examples:
        Basic usage with path merging:
        >>> got = GraphOfThoughts()
        >>> session = Session().start()
        >>> await got.initialize()
        >>> result = await got.execute(
        ...     session,
        ...     "Analyze the impact of climate change on agriculture",
        ...     context={"enable_convergence": True, "branching_factor": 4}
        ... )

        Disabled convergence (tree-like behavior):
        >>> got = GraphOfThoughts()
        >>> session = Session().start()
        >>> await got.initialize()
        >>> result = await got.execute(
        ...     session,
        ...     "Design a new product",
        ...     context={"enable_convergence": False, "max_depth": 5}
        ... )
    """

    def __init__(
        self,
        branching_factor: int = 3,
        max_depth: int = 6,
        min_merge_score: float = 0.6,
        enable_convergence: bool = True,
        convergence_threshold: float = 0.7,
        enable_elicitation: bool = True,
    ) -> None:
        """Initialize the Graph of Thoughts method.

        Args:
            branching_factor: Number of initial branches to explore
            max_depth: Maximum graph depth to explore
            min_merge_score: Minimum combined score for path merging (0.0-1.0)
            enable_convergence: Whether to enable path convergence
            convergence_threshold: Score threshold for considering paths similar (0.0-1.0)
            enable_elicitation: Whether to enable user interaction (default: True)

        Raises:
            ValueError: If parameters are invalid
        """
        if branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {branching_factor}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if not 0.0 <= min_merge_score <= 1.0:
            raise ValueError(f"min_merge_score must be 0.0-1.0, got {min_merge_score}")
        if not 0.0 <= convergence_threshold <= 1.0:
            raise ValueError(f"convergence_threshold must be 0.0-1.0, got {convergence_threshold}")

        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.min_merge_score = min_merge_score
        self.enable_convergence = enable_convergence
        self.convergence_threshold = convergence_threshold
        self.enable_elicitation = enable_elicitation
        self._use_sampling = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.GRAPH_OF_THOUGHTS)

    @property
    def name(self) -> str:
        """Return the method name."""
        return GRAPH_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return GRAPH_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return str(GRAPH_OF_THOUGHTS_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the Graph of Thoughts method.

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
        """Execute Graph of Thoughts reasoning on the input.

        This method explores multiple solution paths in a DAG structure:
        1. Creates root thought analyzing the problem
        2. Generates multiple initial branches (divergence)
        3. Each branch can spawn sub-branches
        4. Similar paths can merge (convergence) if enabled
        5. Merged nodes have multiple parents
        6. Final synthesis combines all paths

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional context with:
                - branching_factor: Number of initial branches
                - max_depth: Maximum graph depth
                - min_merge_score: Minimum score for merging
                - enable_convergence: Whether to enable path convergence
                - convergence_threshold: Similarity threshold for merging

        Returns:
            The final synthesis ThoughtNode

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
        branching = context.get("branching_factor", self.branching_factor)
        depth = context.get("max_depth", self.max_depth)
        min_merge = context.get("min_merge_score", self.min_merge_score)
        enable_conv = context.get("enable_convergence", self.enable_convergence)
        conv_threshold = context.get("convergence_threshold", self.convergence_threshold)

        # Optional elicitation: ask user for graph exploration strategy
        exploration_strategy = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
        ):
            try:
                options = [
                    {
                        "id": "breadth",
                        "label": "Breadth-first (wide exploration across many paths)",
                    },
                    {"id": "depth", "label": "Depth-first (deep analysis of fewer paths)"},
                    {"id": "best", "label": "Best-first (follow highest scoring nodes)"},
                    {"id": "balanced", "label": "Balanced (mix of breadth and depth)"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                if self._execution_context.ctx is None:
                    raise RuntimeError("Execution context has no ctx attribute for elicitation")
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    f"How should the Graph of Thoughts explore '{input_text[:80]}...'?",
                    options,
                    config=config,
                )
                if selection:
                    exploration_strategy = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Fall back to default behavior

        # Create root thought (use sampling if available)
        if self._use_sampling:
            root_content = await self._sample_root_analysis(
                input_text, branching, depth, enable_conv
            )
        else:
            root_content = (
                f"Analyzing problem: {input_text}\n\n"
                f"Initiating Graph of Thoughts reasoning with DAG structure.\n"
                f"Branching factor: {branching}, Max depth: {depth}\n"
                f"Convergence enabled: {enable_conv}"
            )

        root = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GRAPH_OF_THOUGHTS,
            content=root_content,
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            metadata={
                "branching_factor": branching,
                "is_root": True,
                "convergence_enabled": enable_conv,
                "sampled": self._use_sampling,
                "exploration_strategy": exploration_strategy,
            },
        )
        session.add_thought(root)

        # Track all nodes and their parent relationships
        all_nodes: dict[str, ThoughtNode] = {root.id: root}
        # Track parent relationships for DAG (node_id -> list of parent_ids)
        parent_map: dict[str, list[str]] = {root.id: []}
        # Track active frontier nodes at each depth
        frontier: list[ThoughtNode] = [root]

        # Build the graph layer by layer
        for current_depth in range(1, depth + 1):
            next_frontier: list[ThoughtNode] = []

            # Generate branches from each frontier node
            for parent_node in frontier:
                branches = await self._generate_branches(
                    session=session,
                    parent=parent_node,
                    input_text=input_text,
                    branching_factor=branching if current_depth == 1 else 2,  # Fewer sub-branches
                    depth=current_depth,
                    all_nodes=all_nodes,
                    parent_map=parent_map,
                )
                next_frontier.extend(branches)

            # Attempt path convergence if enabled
            if enable_conv and len(next_frontier) > 1:
                merged_frontier = await self._converge_paths(
                    session=session,
                    candidates=next_frontier,
                    input_text=input_text,
                    min_score=min_merge,
                    threshold=conv_threshold,
                    depth=current_depth,
                    all_nodes=all_nodes,
                    parent_map=parent_map,
                )
                frontier = merged_frontier
            else:
                frontier = next_frontier

            # Limit frontier size to prevent explosion, using exploration strategy if specified
            if len(frontier) > branching * 2:
                # Apply exploration strategy to pruning
                if exploration_strategy == "breadth":
                    # Keep more diverse nodes (less aggressive pruning)
                    max_frontier = branching * 3
                elif exploration_strategy == "depth":
                    # Keep fewer nodes (more aggressive pruning)
                    max_frontier = branching
                elif exploration_strategy == "best":
                    # Sort by quality and keep top nodes
                    frontier.sort(key=lambda n: n.quality_score or 0.0, reverse=True)
                    max_frontier = branching * 2
                else:  # "balanced" or None (default)
                    max_frontier = branching * 2

                if len(frontier) > max_frontier:
                    # Sort by quality score if not already sorted
                    if exploration_strategy != "best":
                        frontier.sort(key=lambda n: n.quality_score or 0.0, reverse=True)
                    pruned_count = len(frontier) - max_frontier
                    frontier = frontier[:max_frontier]
                    session.metrics.branches_pruned += pruned_count

        # Create final synthesis from all leaf nodes
        synthesis = await self._create_synthesis(
            session=session,
            leaf_nodes=frontier,
            all_nodes=all_nodes,
            parent_map=parent_map,
        )

        return synthesis

    async def _generate_branches(
        self,
        session: Session,
        parent: ThoughtNode,
        input_text: str,
        branching_factor: int,
        depth: int,
        all_nodes: dict[str, ThoughtNode],
        parent_map: dict[str, list[str]],
    ) -> list[ThoughtNode]:
        """Generate branch thoughts from a parent node.

        Args:
            session: Current session
            parent: Parent thought to branch from
            input_text: Original input text
            branching_factor: Number of branches to create
            depth: Depth of the new branches
            all_nodes: Dictionary tracking all nodes
            parent_map: Dictionary tracking parent relationships

        Returns:
            List of branch ThoughtNodes
        """
        branches: list[ThoughtNode] = []

        # Define different exploration strategies
        strategies = [
            "causal analysis",
            "systematic decomposition",
            "alternative perspective",
            "synthesis approach",
            "critical evaluation",
            "practical application",
            "theoretical framework",
            "empirical evidence",
        ]

        for i in range(branching_factor):
            branch_id = f"branch-{parent.id[:8]}-{i}-d{depth}"
            strategy = strategies[i % len(strategies)]

            # Calculate score with depth decay and variation
            base_score = 0.75 - (depth * 0.08)
            variation = (hash(branch_id) % 25) / 100.0
            score = max(0.0, min(1.0, base_score + variation))

            # Create branch content (use sampling if available)
            if self._use_sampling:
                content, sampled_score = await self._sample_branch(
                    input_text, parent.content, strategy, i + 1, branching_factor, depth
                )
                # Use sampled score if provided
                if sampled_score is not None:
                    score = sampled_score
            else:
                content = (
                    f"Branch {i + 1}: {strategy}\n\n"
                    f"Depth: {depth}, Parent: {parent.id[:8]}...\n\n"
                    f"Exploring '{input_text}' through {strategy}\n\n"
                    f"Key insights:\n"
                    f"- Building on parent analysis\n"
                    f"- Depth {depth} investigation\n"
                    f"- Quality score: {score:.2f}"
                )

            # Create branch thought
            branch = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.GRAPH_OF_THOUGHTS,
                content=content,
                parent_id=parent.id,  # Single parent for now (may merge later)
                branch_id=branch_id,
                confidence=score,
                quality_score=score,
                depth=depth,
                metadata={
                    "strategy": strategy,
                    "branch_index": i,
                    "is_merged": False,
                    "parent_count": 1,
                    "sampled": self._use_sampling,
                },
            )

            session.add_thought(branch)
            all_nodes[branch.id] = branch
            parent_map[branch.id] = [parent.id]
            branches.append(branch)

            session.metrics.branches_created += 1

        return branches

    async def _converge_paths(
        self,
        session: Session,
        candidates: list[ThoughtNode],
        input_text: str,
        min_score: float,
        threshold: float,
        depth: int,
        all_nodes: dict[str, ThoughtNode],
        parent_map: dict[str, list[str]],
    ) -> list[ThoughtNode]:
        """Attempt to merge similar reasoning paths.

        This is the key differentiator from Tree of Thoughts - paths can converge
        to create nodes with multiple parents, forming a DAG structure.

        Args:
            session: Current session
            candidates: Candidate nodes to potentially merge
            input_text: Original input text
            min_score: Minimum combined score for merging
            threshold: Similarity threshold for considering merging
            depth: Current depth in graph
            all_nodes: Dictionary tracking all nodes
            parent_map: Dictionary tracking parent relationships

        Returns:
            List of nodes after convergence (may include merged nodes)
        """
        if len(candidates) < 2:
            return candidates

        result: list[ThoughtNode] = []
        merged_ids: set[str] = set()

        # Try to find pairs that should merge
        for i, node_a in enumerate(candidates):
            if node_a.id in merged_ids:
                continue

            # Look for a compatible merge partner
            merge_partner: ThoughtNode | None = None
            best_combined_score = 0.0

            for _j, node_b in enumerate(candidates[i + 1 :], start=i + 1):
                if node_b.id in merged_ids:
                    continue

                # Calculate similarity (simulated - in real implementation use embeddings)
                similarity = self._calculate_similarity(node_a, node_b)
                combined_score = (node_a.quality_score or 0.0) + (node_b.quality_score or 0.0)

                if (
                    similarity >= threshold
                    and combined_score >= min_score
                    and combined_score > best_combined_score
                ):
                    best_combined_score = combined_score
                    merge_partner = node_b

            # If we found a merge partner, create merged node
            if merge_partner:
                merged = await self._merge_nodes(
                    session=session,
                    node_a=node_a,
                    node_b=merge_partner,
                    depth=depth,
                    all_nodes=all_nodes,
                    parent_map=parent_map,
                )
                result.append(merged)
                merged_ids.add(node_a.id)
                merged_ids.add(merge_partner.id)
                session.metrics.branches_pruned += 1  # Two branches became one
            else:
                # No merge found, keep original
                result.append(node_a)

        return result

    def _calculate_similarity(self, node_a: ThoughtNode, node_b: ThoughtNode) -> float:
        """Calculate similarity between two nodes.

        In a real implementation, this would use embeddings or semantic similarity.
        For now, we use a simple heuristic based on strategy and scores.

        Args:
            node_a: First node
            node_b: Second node

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Check if strategies are compatible
        strategy_a = node_a.metadata.get("strategy", "")
        strategy_b = node_b.metadata.get("strategy", "")

        # If strategies mention similar concepts, increase similarity
        compatible_pairs = [
            ("causal", "systematic"),
            ("alternative", "critical"),
            ("synthesis", "theoretical"),
            ("practical", "empirical"),
        ]

        base_similarity = 0.5
        for term_a, term_b in compatible_pairs:
            if (term_a in strategy_a and term_b in strategy_b) or (
                term_b in strategy_a and term_a in strategy_b
            ):
                base_similarity += 0.2

        # Adjust by score similarity
        score_diff = abs((node_a.quality_score or 0.0) - (node_b.quality_score or 0.0))
        score_similarity = 1.0 - score_diff

        # Combined similarity
        return min(1.0, (base_similarity + score_similarity) / 2.0)

    async def _merge_nodes(
        self,
        session: Session,
        node_a: ThoughtNode,
        node_b: ThoughtNode,
        depth: int,
        all_nodes: dict[str, ThoughtNode],
        parent_map: dict[str, list[str]],
    ) -> ThoughtNode:
        """Merge two nodes into a single node with multiple parents.

        This creates the DAG structure - the merged node will have both
        node_a and node_b as parents.

        Args:
            session: Current session
            node_a: First node to merge
            node_b: Second node to merge
            depth: Current depth
            all_nodes: Dictionary tracking all nodes
            parent_map: Dictionary tracking parent relationships

        Returns:
            New merged ThoughtNode
        """
        # Get all parent IDs from both nodes
        parents_a = parent_map.get(node_a.id, [node_a.parent_id] if node_a.parent_id else [])
        parents_b = parent_map.get(node_b.id, [node_b.parent_id] if node_b.parent_id else [])
        all_parents = list(set(parents_a + parents_b))  # Deduplicate

        # Calculate merged scores
        merged_score = ((node_a.quality_score or 0.0) + (node_b.quality_score or 0.0)) / 2.0
        merged_confidence = ((node_a.confidence or 0.0) + (node_b.confidence or 0.0)) / 2.0

        # Create merged content
        strategy_a = node_a.metadata.get("strategy", "approach A")
        strategy_b = node_b.metadata.get("strategy", "approach B")

        # Create merged content (use sampling if available)
        if self._use_sampling:
            content = await self._sample_merge(
                node_a.content, node_b.content, strategy_a, strategy_b, len(all_parents)
            )
        else:
            content = (
                f"MERGED PATH (Multiple Parents)\n\n"
                f"Combining insights from:\n- {strategy_a}\n- {strategy_b}\n\n"
                f"Parent nodes: {len(all_parents)}\n\n"
                f"Synthesized analysis:\n"
                f"This convergent path integrates multiple reasoning strategies, "
                f"creating a unified understanding that leverages the strengths of "
                f"both approaches.\n\n"
                f"Combined quality: {merged_score:.2f}"
            )

        # Create merged node - note we only set parent_id to first parent
        # The full parent list is tracked in parent_map
        merged = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,  # Merged nodes are synthesis type
            method_id=MethodIdentifier.GRAPH_OF_THOUGHTS,
            content=content,
            parent_id=all_parents[0] if all_parents else None,  # Primary parent
            confidence=merged_confidence,
            quality_score=merged_score,
            depth=depth,
            metadata={
                "is_merged": True,
                "parent_count": len(all_parents),
                "merged_from": [node_a.id, node_b.id],
                "strategies_combined": [strategy_a, strategy_b],
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(merged)
        all_nodes[merged.id] = merged
        parent_map[merged.id] = all_parents  # Store all parents for DAG

        return merged

    async def _create_synthesis(
        self,
        session: Session,
        leaf_nodes: list[ThoughtNode],
        all_nodes: dict[str, ThoughtNode],
        parent_map: dict[str, list[str]],
    ) -> ThoughtNode:
        """Create final synthesis from all leaf nodes.

        Args:
            session: Current session
            leaf_nodes: All leaf nodes in the graph
            all_nodes: Dictionary of all nodes
            parent_map: Parent relationship map

        Returns:
            Final synthesis ThoughtNode
        """
        # Calculate overall statistics
        total_nodes = len(all_nodes)
        merged_nodes = sum(1 for n in all_nodes.values() if n.metadata.get("is_merged", False))
        avg_score = (
            sum(n.quality_score or 0.0 for n in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0.0
        )

        # Find nodes with multiple parents (DAG indicators)
        multi_parent_nodes = [
            node_id for node_id, parents in parent_map.items() if len(parents) > 1
        ]

        # Create synthesis content (use sampling if available)
        if self._use_sampling:
            content = await self._sample_synthesis(
                leaf_nodes, total_nodes, merged_nodes, len(multi_parent_nodes), avg_score
            )
        else:
            content = "Graph of Thoughts - Final Synthesis\n\n"
            content += "DAG Structure Summary:\n"
            content += f"- Total nodes explored: {total_nodes}\n"
            content += f"- Convergence points (multi-parent nodes): {len(multi_parent_nodes)}\n"
            content += f"- Merged reasoning paths: {merged_nodes}\n"
            content += f"- Final leaf nodes: {len(leaf_nodes)}\n"
            content += f"- Average quality score: {avg_score:.2f}\n\n"

            content += "Key Insights:\n"
            for i, leaf in enumerate(leaf_nodes[:3], 1):  # Top 3 leaves
                strategy = leaf.metadata.get("strategy", "analysis")
                content += f"{i}. {strategy}: {leaf.content[:100]}...\n"

            content += "\nConclusion:\n"
            content += "The graph-based reasoning explored multiple interconnected paths, "
            content += f"with {len(multi_parent_nodes)} convergence points where different "
            content += "approaches merged into unified insights. This DAG structure enabled "
            content += "more flexible reasoning than hierarchical methods."

        # Create synthesis thought
        synthesis = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.GRAPH_OF_THOUGHTS,
            content=content,
            parent_id=leaf_nodes[0].id if leaf_nodes else None,
            confidence=avg_score,
            quality_score=avg_score,
            depth=max(n.depth for n in leaf_nodes) + 1 if leaf_nodes else 0,
            metadata={
                "is_final": True,
                "total_nodes": total_nodes,
                "convergence_points": len(multi_parent_nodes),
                "merged_paths": merged_nodes,
                "leaf_count": len(leaf_nodes),
            },
        )

        session.add_thought(synthesis)
        all_nodes[synthesis.id] = synthesis

        # Add all leaves as parents for final synthesis
        parent_map[synthesis.id] = [n.id for n in leaf_nodes]

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

        For Graph of Thoughts, continuing means adding new branches that can
        potentially merge with existing paths.

        Args:
            session: Current session
            previous_thought: Thought to continue from
            guidance: Optional guidance for exploration
            context: Optional context parameters
            execution_context: Optional execution context for LLM sampling

        Returns:
            New ThoughtNode continuing exploration
        """
        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        context = context or {}
        branching = context.get("branching_factor", 2)

        # Generate continuation content (use sampling if available)
        if self._use_sampling:
            continuation_content = await self._sample_continuation(
                previous_thought.content, guidance, branching
            )
        else:
            guidance_text = guidance or "Exploring additional paths with potential convergence"
            continuation_content = (
                f"Continuing graph exploration from previous thought.\n\n"
                f"Guidance: {guidance_text}\n\n"
                f"Generating {branching} new exploration paths that may merge with "
                f"existing graph..."
            )

        # Create continuation thought
        continuation = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.GRAPH_OF_THOUGHTS,
            content=continuation_content,
            parent_id=previous_thought.id,
            confidence=previous_thought.confidence * 0.95,
            quality_score=previous_thought.quality_score,
            depth=previous_thought.depth + 1,
            metadata={
                "is_continuation": True,
                "guidance": guidance,
                "can_merge": True,
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(continuation)
        return continuation

    async def _sample_root_analysis(
        self,
        input_text: str,
        branching: int,
        depth: int,
        enable_convergence: bool,
    ) -> str:
        """Sample root analysis using LLM.

        Args:
            input_text: The problem to analyze
            branching: Branching factor
            depth: Max depth
            enable_convergence: Whether convergence is enabled

        Returns:
            Sampled root analysis content
        """
        system_prompt = """You are a reasoning assistant using Graph of Thoughts methodology.
Analyze the problem and prepare for systematic exploration using a DAG structure.

Your analysis should:
1. Break down the problem into key components
2. Identify potential solution approaches
3. Note where different paths might converge
4. Establish evaluation criteria for comparing paths"""

        user_prompt = f"""Problem: {input_text}

Graph configuration:
- Branching factor: {branching}
- Max depth: {depth}
- Convergence enabled: {enable_convergence}

Provide a root analysis for the Graph of Thoughts exploration."""

        def fallback() -> str:
            return (
                f"Analyzing problem: {input_text}\n\n"
                f"Initiating Graph of Thoughts reasoning with DAG structure.\n"
                f"Branching factor: {branching}, Max depth: {depth}\n"
                f"Convergence enabled: {enable_convergence}"
            )

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
            branch_num: Branch number (1-indexed)
            total_branches: Total number of branches
            depth: Current depth

        Returns:
            Tuple of (content, optional quality score)
        """
        system_prompt = (
            f"You are exploring branch {branch_num} of {total_branches} in a "
            f"Graph of Thoughts.\n"
            f"Use the '{strategy}' strategy to analyze the problem.\n"
            f"Build upon the parent analysis while exploring this specific perspective."
        )

        user_prompt = f"""Problem: {input_text}

Parent analysis:
{parent_content}

Strategy: {strategy}
Branch: {branch_num}/{total_branches}
Depth: {depth}

Provide your analysis for this branch, focusing on the specified strategy."""

        def fallback() -> str:
            return (
                f"Branch {branch_num}: {strategy}\n\n"
                f"Depth: {depth}\n\n"
                f"Exploring '{input_text}' through {strategy}\n\n"
                f"Key insights:\n"
                f"- Building on parent analysis\n"
                f"- Depth {depth} investigation"
            )

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=600,
        )
        # Could extract quality score from content if LLM provides it
        return content, None

    async def _sample_merge(
        self,
        content_a: str,
        content_b: str,
        strategy_a: str,
        strategy_b: str,
        parent_count: int,
    ) -> str:
        """Sample a merge of two paths using LLM.

        Args:
            content_a: Content from first node
            content_b: Content from second node
            strategy_a: Strategy from first node
            strategy_b: Strategy from second node
            parent_count: Number of parent nodes

        Returns:
            Merged content
        """
        system_prompt = """You are merging two reasoning paths in a Graph of Thoughts.
Create a synthesis that integrates insights from both approaches."""

        user_prompt = f"""Merge these two reasoning paths:

Path A ({strategy_a}):
{content_a[:500]}...

Path B ({strategy_b}):
{content_b[:500]}...

Parent nodes: {parent_count}

Create a unified analysis that synthesizes both perspectives."""

        def fallback() -> str:
            return (
                f"MERGED PATH (Multiple Parents)\n\n"
                f"Combining insights from:\n- {strategy_a}\n- {strategy_b}\n\n"
                f"Parent nodes: {parent_count}\n\n"
                f"Synthesized analysis:\n"
                f"This convergent path integrates multiple reasoning strategies."
            )

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=700,
        )

    async def _sample_synthesis(
        self,
        leaf_nodes: list[ThoughtNode],
        total_nodes: int,
        merged_nodes: int,
        convergence_points: int,
        avg_score: float,
    ) -> str:
        """Sample final synthesis using LLM.

        Args:
            leaf_nodes: All leaf nodes in the graph
            total_nodes: Total number of nodes explored
            merged_nodes: Number of merged nodes
            convergence_points: Number of convergence points
            avg_score: Average quality score

        Returns:
            Final synthesis content
        """
        # Gather key insights from top leaf nodes
        top_leaves = sorted(leaf_nodes, key=lambda n: n.quality_score or 0.0, reverse=True)[:3]
        leaf_summaries = []
        for i, leaf in enumerate(top_leaves, 1):
            strategy = leaf.metadata.get("strategy", "analysis")
            preview = leaf.content[:200].replace("\n", " ")
            leaf_summaries.append(f"{i}. {strategy}: {preview}...")

        system_prompt = """You are creating the final synthesis for a Graph of Thoughts exploration.
Integrate the key insights from all reasoning paths into a coherent conclusion."""

        user_prompt = f"""Create a final synthesis from this Graph of Thoughts exploration:

Statistics:
- Total nodes: {total_nodes}
- Convergence points: {convergence_points}
- Merged paths: {merged_nodes}
- Final leaf nodes: {len(leaf_nodes)}
- Average quality: {avg_score:.2f}

Top insights:
{chr(10).join(leaf_summaries)}

Provide a comprehensive synthesis highlighting how the graph structure enabled
multi-path reasoning."""

        def fallback() -> str:
            return (
                f"Graph of Thoughts - Final Synthesis\n\n"
                f"DAG Structure Summary:\n"
                f"- Total nodes: {total_nodes}\n"
                f"- Convergence points: {convergence_points}\n"
                f"- Merged paths: {merged_nodes}\n"
                f"- Average quality: {avg_score:.2f}\n\n"
                f"Conclusion:\n"
                f"The graph-based reasoning explored multiple interconnected paths."
            )

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )

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
            Continuation content
        """
        system_prompt = """You are continuing a Graph of Thoughts exploration.
Build upon the previous analysis while exploring new paths."""

        user_prompt = f"""Previous analysis:
{previous_content[:500]}...

Guidance: {guidance or "Explore additional paths with potential convergence"}
New branches to explore: {branching}

Continue the graph exploration, noting how new paths might merge with existing ones."""

        def fallback() -> str:
            guidance_text = guidance or "Exploring additional paths with potential convergence"
            return (
                f"Continuing graph exploration from previous thought.\n\n"
                f"Guidance: {guidance_text}\n\n"
                f"Generating {branching} new exploration paths that may merge with "
                f"existing graph..."
            )

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=600,
        )

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True (this method has no external dependencies)
        """
        return True


# Export metadata and class
__all__ = [
    "GraphOfThoughts",
    "GRAPH_OF_THOUGHTS_METADATA",
]
