"""Beam Search reasoning method implementation.

This module implements the Beam Search reasoning approach, which maintains multiple
parallel solution candidates and prunes to keep only the most promising paths.

Beam Search enables:
- Parallel exploration of multiple solution candidates
- Heuristic-based scoring and evaluation
- Pruning to maintain computational efficiency
- Breadth-first exploration with quality control
- Balancing exploration breadth with depth

Beam Search consists of iterative steps:
1. Generate multiple candidate solutions at each level
2. Score each candidate using heuristic evaluation
3. Prune to keep only top beam_width candidates
4. Expand from all kept candidates
5. Continue until convergence or max_depth

This method is particularly effective for optimization problems where maintaining
multiple promising paths leads to better solutions than greedy single-path approaches.

Reference: "Beam Search" - A heuristic search algorithm used in speech recognition,
machine translation, and other sequence generation tasks.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.methods.base import MethodMetadata


# Define metadata for Beam Search method
BEAM_SEARCH_METADATA = MethodMetadata(
    identifier=MethodIdentifier.BEAM_SEARCH,
    name="Beam Search Reasoning",
    description="Parallel exploration with pruning to maintain most promising solution paths",
    category=MethodCategory.ADVANCED,
    tags=frozenset({
        "search",
        "parallel",
        "pruning",
        "optimization",
        "heuristic",
        "breadth-first",
        "advanced",
    }),
    complexity=6,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=4,  # Need root + multiple levels with pruning
    max_thoughts=0,  # Unlimited - depends on beam_width and max_depth
    avg_tokens_per_thought=500,
    best_for=(
        "optimization problems",
        "search with constraints",
        "finding multiple good solutions",
        "balancing exploration/exploitation",
        "sequence generation tasks",
        "problems with many valid paths",
    ),
    not_recommended_for=(
        "problems requiring deep single-path reasoning",
        "very simple problems",
        "tasks requiring exhaustive exploration",
    ),
)


class BeamCandidate:
    """Internal candidate representation for beam search.

    This class maintains a candidate solution with its score and metadata
    needed for beam search selection and pruning.

    Attributes:
        thought: The ThoughtNode associated with this candidate
        score: Heuristic score for this candidate
        level: Level in the beam search (depth from root)
        metadata: Additional metadata for tracking
    """

    def __init__(
        self,
        thought: ThoughtNode,
        score: float,
        level: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a beam candidate.

        Args:
            thought: The associated ThoughtNode
            score: Heuristic score for ranking
            level: Level in beam search
            metadata: Optional additional metadata
        """
        self.thought = thought
        self.score = score
        self.level = level
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        """Return string representation of candidate."""
        return f"BeamCandidate(score={self.score:.3f}, level={self.level})"


class BeamSearchMethod:
    """Beam Search reasoning method implementation.

    This class implements the ReasoningMethod protocol to provide beam search-based
    reasoning capabilities. It maintains multiple parallel solution paths and
    prunes to keep only the most promising candidates at each level.

    The method balances exploration breadth (beam width) with computational
    efficiency (pruning) to find high-quality solutions.

    Attributes:
        beam_width: Number of candidates to keep at each step (default: 3)
        max_depth: Maximum reasoning depth (default: 5)
        scoring_strategy: How to score candidates - "heuristic" or "confidence" (default: "heuristic")

    Examples:
        Basic usage:
        >>> beam_search = BeamSearchMethod()
        >>> session = Session().start()
        >>> await beam_search.initialize()
        >>> result = await beam_search.execute(
        ...     session,
        ...     "Find the optimal solution for resource allocation",
        ... )

        Custom parameters:
        >>> beam_search = BeamSearchMethod(
        ...     beam_width=5,
        ...     max_depth=8,
        ...     scoring_strategy="confidence",
        ... )
        >>> result = await beam_search.execute(
        ...     session,
        ...     "Optimize multi-objective decision problem",
        ...     context={"constraints": ["budget", "time", "quality"]}
        ... )
    """

    def __init__(
        self,
        beam_width: int = 3,
        max_depth: int = 5,
        scoring_strategy: str = "heuristic",
    ) -> None:
        """Initialize the Beam Search method.

        Args:
            beam_width: Number of candidates to keep at each level
            max_depth: Maximum depth to explore
            scoring_strategy: Strategy for scoring candidates ("heuristic" or "confidence")

        Raises:
            ValueError: If parameters are invalid
        """
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if scoring_strategy not in {"heuristic", "confidence"}:
            raise ValueError(f"scoring_strategy must be 'heuristic' or 'confidence', got {scoring_strategy}")

        self.beam_width = beam_width
        self.max_depth = max_depth
        self.scoring_strategy = scoring_strategy

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.BEAM_SEARCH)

    @property
    def name(self) -> str:
        """Return the method name."""
        return BEAM_SEARCH_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return BEAM_SEARCH_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return str(BEAM_SEARCH_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the Beam Search method.

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
    ) -> ThoughtNode:
        """Execute beam search reasoning on the input.

        This method performs Beam Search:
        1. Creates initial candidates representing the problem
        2. Iteratively generates and scores new candidates
        3. Prunes to keep only top beam_width candidates
        4. Expands from all remaining candidates
        5. Returns the best solution found

        Args:
            session: The current reasoning session
            input_text: The problem or question to solve
            context: Optional context with:
                - beam_width: Number of candidates per level
                - max_depth: Maximum search depth
                - scoring_strategy: How to score candidates
                - constraints: Problem-specific constraints

        Returns:
            A ThoughtNode representing the best solution found

        Raises:
            ValueError: If session is not active
        """
        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Extract context parameters with defaults
        context = context or {}
        beam_width = context.get("beam_width", self.beam_width)
        max_depth = context.get("max_depth", self.max_depth)
        scoring_strategy = context.get("scoring_strategy", self.scoring_strategy)
        constraints = context.get("constraints", [])

        # Initialize metrics tracking
        metrics = {
            "candidates_generated": 0,
            "candidates_pruned": 0,
            "best_scores_per_level": [],
            "beam_diversity": [],
        }

        # Create root thought
        root_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content=f"Beam Search Analysis: {input_text}\n\nInitializing beam search with:\n- Beam width: {beam_width} candidates per level\n- Max depth: {max_depth} levels\n- Scoring strategy: {scoring_strategy}\n- Constraints: {', '.join(constraints) if constraints else 'None'}\n\nBeam search will maintain {beam_width} most promising solution paths, pruning less promising candidates at each level to balance exploration breadth with computational efficiency.",
            confidence=0.5,
            quality_score=0.5,
            depth=0,
            metadata={
                "beam_width": beam_width,
                "max_depth": max_depth,
                "scoring_strategy": scoring_strategy,
                "is_root": True,
            },
        )
        session.add_thought(root_thought)

        # Initialize beam with root candidate
        current_beam = [
            BeamCandidate(
                thought=root_thought,
                score=0.5,
                level=0,
                metadata={"is_initial": True},
            )
        ]

        # Track all generated thoughts
        all_thoughts: dict[str, ThoughtNode] = {root_thought.id: root_thought}

        # Iteratively expand beam
        for level in range(1, max_depth + 1):
            # Generate candidates from current beam
            new_candidates: list[BeamCandidate] = []

            for beam_candidate in current_beam:
                # Generate multiple expansions from each candidate
                expansions = await self._generate_expansions(
                    session,
                    beam_candidate,
                    input_text,
                    level,
                    beam_width,
                    all_thoughts,
                )
                new_candidates.extend(expansions)
                metrics["candidates_generated"] += len(expansions)

            if not new_candidates:
                # No more candidates to explore
                break

            # Score all candidates
            for candidate in new_candidates:
                candidate.score = await self._score_candidate(
                    candidate,
                    scoring_strategy,
                    input_text,
                )

            # Sort by score descending
            new_candidates.sort(key=lambda c: c.score, reverse=True)

            # Prune to keep only top beam_width candidates
            pruned_count = max(0, len(new_candidates) - beam_width)
            current_beam = new_candidates[:beam_width]
            metrics["candidates_pruned"] += pruned_count

            # Track best score and diversity at this level
            if current_beam:
                best_score = current_beam[0].score
                metrics["best_scores_per_level"].append(best_score)

                # Calculate diversity (standard deviation of scores)
                scores = [c.score for c in current_beam]
                diversity = self._calculate_diversity(scores)
                metrics["beam_diversity"].append(diversity)

            # Log pruning if any candidates were removed
            if pruned_count > 0:
                pruning_thought = ThoughtNode(
                    id=str(uuid4()),
                    type=ThoughtType.OBSERVATION,
                    method_id=MethodIdentifier.BEAM_SEARCH,
                    content=f"Pruning at Level {level}\n\nGenerated {len(new_candidates)} candidates\nKeeping top {beam_width} candidates\nPruned {pruned_count} lower-scoring candidates\n\nBest score: {current_beam[0].score:.3f}\nBeam diversity: {diversity:.3f}\n\nRemaining candidates represent the {beam_width} most promising solution paths based on {scoring_strategy} scoring.",
                    parent_id=root_thought.id,
                    confidence=0.7,
                    quality_score=0.7,
                    depth=level,
                    metadata={
                        "is_pruning": True,
                        "level": level,
                        "pruned_count": pruned_count,
                        "best_score": current_beam[0].score,
                        "diversity": diversity,
                    },
                )
                session.add_thought(pruning_thought)
                all_thoughts[pruning_thought.id] = pruning_thought
                session.metrics.branches_pruned += pruned_count

            # Check for convergence (all candidates very similar)
            if diversity < 0.05 and level > 2:
                convergence_thought = ThoughtNode(
                    id=str(uuid4()),
                    type=ThoughtType.OBSERVATION,
                    method_id=MethodIdentifier.BEAM_SEARCH,
                    content=f"Early Convergence Detected at Level {level}\n\nBeam diversity has dropped to {diversity:.3f}, indicating candidates are converging to similar solutions.\n\nStopping search early to avoid redundant exploration.",
                    parent_id=root_thought.id,
                    confidence=0.8,
                    quality_score=0.8,
                    depth=level,
                    metadata={
                        "is_convergence": True,
                        "level": level,
                        "diversity": diversity,
                    },
                )
                session.add_thought(convergence_thought)
                all_thoughts[convergence_thought.id] = convergence_thought
                break

        # Select best candidate from final beam
        if not current_beam:
            # Fallback to root if no candidates
            best_candidate = BeamCandidate(root_thought, 0.5, 0)
        else:
            best_candidate = current_beam[0]

        # Calculate final metrics
        final_diversity = metrics["beam_diversity"][-1] if metrics["beam_diversity"] else 0.0
        avg_best_score = sum(metrics["best_scores_per_level"]) / max(len(metrics["best_scores_per_level"]), 1)

        # Create final synthesis thought
        synthesis = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content=f"Beam Search Complete\n\nBest solution found:\n{best_candidate.thought.content}\n\nSearch Statistics:\n- Total candidates generated: {metrics['candidates_generated']}\n- Total candidates pruned: {metrics['candidates_pruned']}\n- Levels explored: {len(metrics['best_scores_per_level'])}\n- Final best score: {best_candidate.score:.3f}\n- Average best score: {avg_best_score:.3f}\n- Final beam diversity: {final_diversity:.3f}\n- Beam width maintained: {beam_width}\n\nThe beam search explored {metrics['candidates_generated']} total candidates while maintaining a beam of {beam_width} most promising paths at each level. This balanced approach allows thorough exploration while remaining computationally efficient through strategic pruning.",
            parent_id=best_candidate.thought.id,
            confidence=best_candidate.score,
            quality_score=best_candidate.score,
            depth=best_candidate.level + 1,
            metadata={
                "is_final": True,
                "total_candidates": metrics["candidates_generated"],
                "total_pruned": metrics["candidates_pruned"],
                "levels_explored": len(metrics["best_scores_per_level"]),
                "best_score": best_candidate.score,
                "avg_best_score": avg_best_score,
                "final_diversity": final_diversity,
                "best_scores_per_level": metrics["best_scores_per_level"],
            },
        )
        session.add_thought(synthesis)

        return synthesis

    async def _generate_expansions(
        self,
        session: Session,
        parent_candidate: BeamCandidate,
        input_text: str,
        level: int,
        beam_width: int,
        all_thoughts: dict[str, ThoughtNode],
    ) -> list[BeamCandidate]:
        """Generate expansion candidates from a parent candidate.

        Args:
            session: Current session
            parent_candidate: Parent candidate to expand from
            input_text: Original input text
            level: Current level in search
            beam_width: Number of expansions to generate
            all_thoughts: Dictionary tracking all thoughts

        Returns:
            List of new BeamCandidate objects
        """
        expansions: list[BeamCandidate] = []

        # Generate beam_width expansions from this candidate
        # In practice, these would be domain-specific solution refinements
        expansion_approaches = [
            "optimizing primary objective",
            "balancing multiple constraints",
            "exploring alternative approach",
            "refining current strategy",
            "considering edge cases",
            "maximizing efficiency",
            "minimizing risks",
            "enhancing robustness",
        ]

        for i in range(min(beam_width, len(expansion_approaches))):
            approach = expansion_approaches[i]

            # Create expansion thought
            expansion_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.BEAM_SEARCH,
                content=f"Candidate Expansion: {approach}\n\nLevel {level} - Branch {i + 1}\n\nBuilding on: {parent_candidate.thought.content[:150]}...\n\nThis candidate explores {approach} as a potential solution path.\n\nParent score: {parent_candidate.score:.3f}",
                parent_id=parent_candidate.thought.id,
                branch_id=f"beam-{parent_candidate.thought.id[:8]}-{i}",
                confidence=0.5,
                quality_score=0.5,
                depth=level,
                metadata={
                    "approach": approach,
                    "is_expansion": True,
                    "parent_score": parent_candidate.score,
                    "level": level,
                },
            )
            session.add_thought(expansion_thought)
            all_thoughts[expansion_thought.id] = expansion_thought
            session.metrics.branches_created += 1

            # Create candidate object
            candidate = BeamCandidate(
                thought=expansion_thought,
                score=0.0,  # Will be scored later
                level=level,
                metadata={
                    "approach": approach,
                    "parent_id": parent_candidate.thought.id,
                },
            )
            expansions.append(candidate)

        return expansions

    async def _score_candidate(
        self,
        candidate: BeamCandidate,
        scoring_strategy: str,
        input_text: str,
    ) -> float:
        """Score a candidate using the specified strategy.

        Args:
            candidate: Candidate to score
            scoring_strategy: Scoring strategy to use
            input_text: Original input for context

        Returns:
            Score in range [0.0, 1.0]
        """
        if scoring_strategy == "confidence":
            # Use the thought's confidence as score
            return candidate.thought.confidence

        # Default: heuristic scoring
        # In practice, this would use domain-specific heuristics
        # For now, use a combination of depth, parent score, and randomness

        # Base score from parent
        parent_score = candidate.metadata.get("parent_score", 0.5)

        # Add some variation based on approach
        approach_bonus = hash(candidate.metadata.get("approach", "")) % 100 / 200.0  # 0.0 to 0.5

        # Slight decay with depth (prefer shallower solutions if similar quality)
        depth_penalty = 0.02 * candidate.level

        # Combine factors
        score = parent_score + approach_bonus - depth_penalty

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _calculate_diversity(self, scores: list[float]) -> float:
        """Calculate diversity (standard deviation) of scores.

        Args:
            scores: List of candidate scores

        Returns:
            Standard deviation of scores
        """
        if len(scores) <= 1:
            return 0.0

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        For beam search, continuing means expanding the beam further
        or refining the search with updated parameters.

        Args:
            session: Current session
            previous_thought: Thought to continue from
            guidance: Optional guidance for refinement
            context: Optional context parameters

        Returns:
            New ThoughtNode with continued search
        """
        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        context = context or {}
        additional_depth = context.get("additional_depth", 2)

        # Create continuation thought
        continuation = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content=f"Continuing Beam Search from previous analysis.\n\nGuidance: {guidance or 'Expanding search depth'}\n\nExtending search by {additional_depth} additional levels to explore solution space further...\n\nPrevious best score: {previous_thought.confidence:.3f}",
            parent_id=previous_thought.id,
            confidence=previous_thought.confidence * 0.95,
            quality_score=previous_thought.quality_score,
            depth=previous_thought.depth + 1,
            metadata={
                "is_continuation": True,
                "additional_depth": additional_depth,
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
    "BeamSearchMethod",
    "BEAM_SEARCH_METADATA",
    "BeamCandidate",
]
