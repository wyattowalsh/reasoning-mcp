"""MCP tool functions for method discovery and recommendation.

This module provides the MCP tool interface for:
- Listing available reasoning methods
- Recommending methods for specific problems
- Comparing multiple methods for a problem
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reasoning_mcp.models.core import MethodIdentifier
from reasoning_mcp.models.tools import ComparisonResult, MethodInfo, Recommendation
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.selector import MethodSelector, SelectionConstraint

if TYPE_CHECKING:
    from reasoning_mcp.models.core import MethodCategory


def methods_list(
    category: str | MethodCategory | None = None,
    tags: list[str] | None = None,
) -> list[MethodInfo]:
    """List available reasoning methods.

    This tool allows discovery of available reasoning methods, with optional
    filtering by category and tags. Use this to explore what methods are
    available and understand their capabilities.

    Args:
        category: Filter by method category (e.g., "core", "high_value", "specialized").
                 If None, all categories are included.
        tags: Filter by tags. Methods must have ALL specified tags to be included.
              If None or empty, no tag filtering is applied.

    Returns:
        List of MethodInfo objects containing method metadata.

    Examples:
        List all methods:
        >>> methods = methods_list()

        List only core methods:
        >>> core_methods = methods_list(category="core")

        List methods with specific tags:
        >>> ethical_methods = methods_list(tags=["ethical", "structured"])
    """
    # Get the global registry (in production, this would be injected)
    registry = MethodRegistry()

    # Convert tags to set for filtering
    tags_set = set(tags) if tags else None

    # Query the registry
    metadata_list = registry.list_methods(
        category=category,
        tags=tags_set,
    )

    # Convert to MethodInfo objects
    return [
        MethodInfo(
            id=metadata.identifier,
            name=metadata.name,
            description=metadata.description,
            category=metadata.category,
            parameters={
                "complexity": metadata.complexity,
                "supports_branching": metadata.supports_branching,
                "supports_revision": metadata.supports_revision,
                "min_thoughts": metadata.min_thoughts,
                "max_thoughts": metadata.max_thoughts,
                "avg_tokens_per_thought": metadata.avg_tokens_per_thought,
            },
            tags=list(metadata.tags),
        )
        for metadata in metadata_list
    ]


def methods_recommend(
    problem: str,
    max_results: int = 3,
) -> list[Recommendation]:
    """Recommend reasoning methods for a specific problem.

    This tool analyzes a problem description and recommends the most suitable
    reasoning methods based on detected patterns, problem type, and complexity.

    Args:
        problem: The problem description to analyze. Should be a clear statement
                of what needs to be reasoned about.
        max_results: Maximum number of recommendations to return. Defaults to 3.

    Returns:
        List of Recommendation objects, sorted by score (highest first).
        Each recommendation includes the method, score, reason, and confidence.

    Examples:
        Get recommendations for an ethical problem:
        >>> recommendations = methods_recommend(
        ...     "Should we implement this feature that might compromise user privacy?",
        ...     max_results=3
        ... )

        Get recommendations for a math problem:
        >>> recommendations = methods_recommend(
        ...     "Calculate the optimal solution to this linear programming problem"
        ... )
    """
    # Get the global registry
    registry = MethodRegistry()

    # Create selector
    selector = MethodSelector(registry)

    # Get recommendations
    method_recommendations = selector.recommend(
        problem=problem,
        max_recommendations=max_results,
    )

    # Convert to Recommendation objects
    return [
        Recommendation(
            method_id=MethodIdentifier(rec.identifier),
            score=rec.score,
            reason=rec.reasoning,
            confidence=rec.confidence,
        )
        for rec in method_recommendations
    ]


def methods_compare(
    methods: list[str],
    problem: str,
) -> ComparisonResult:
    """Compare multiple reasoning methods for a specific problem.

    This tool provides a comparative analysis of specified methods for a given
    problem, helping you understand which method is most suitable and why.

    Args:
        methods: List of method identifiers to compare (e.g., ["chain_of_thought",
                "tree_of_thoughts", "ethical_reasoning"]).
        problem: The problem description to evaluate the methods against.

    Returns:
        ComparisonResult containing:
        - methods: The list of compared methods
        - winner: The highest-scoring method (or None if tie)
        - scores: Dict mapping method IDs to their scores
        - analysis: Detailed explanation of the comparison

    Examples:
        Compare methods for an ethical decision:
        >>> result = methods_compare(
        ...     methods=["ethical_reasoning", "dialectic", "socratic"],
        ...     problem="Should we prioritize user privacy over business metrics?"
        ... )

        Compare methods for a technical problem:
        >>> result = methods_compare(
        ...     methods=["chain_of_thought", "react", "code_reasoning"],
        ...     problem="Debug this complex race condition in our async code"
        ... )
    """
    # Get the global registry
    registry = MethodRegistry()

    # Create selector
    selector = MethodSelector(registry)

    # Get recommendations for all methods
    # Create constraints that only allow the specified methods
    constraints = SelectionConstraint(
        allowed_methods=frozenset(methods),
    )

    method_recommendations = selector.recommend(
        problem=problem,
        constraints=constraints,
        max_recommendations=len(methods),
    )

    # Build scores dictionary
    scores: dict[str, float] = {}
    for rec in method_recommendations:
        scores[rec.identifier] = rec.score

    # Add zero scores for methods that weren't recommended
    for method_id in methods:
        if method_id not in scores:
            scores[method_id] = 0.0

    # Find winner (highest score, None if tie)
    if scores:
        max_score = max(scores.values())
        winners = [m for m, s in scores.items() if s == max_score]
        winner = winners[0] if len(winners) == 1 else None
    else:
        winner = None

    # Build analysis
    analysis_parts = []

    if not method_recommendations:
        analysis_parts.append("None of the specified methods are well-suited for this problem.")
    else:
        # Analyze top method
        top_rec = method_recommendations[0]
        analysis_parts.append(
            f"Best method: {top_rec.identifier} (score: {top_rec.score:.2f}). "
            f"{top_rec.reasoning}"
        )

        # Compare others
        if len(method_recommendations) > 1:
            analysis_parts.append("\n\nComparison with alternatives:")
            for rec in method_recommendations[1:]:
                diff = top_rec.score - rec.score
                analysis_parts.append(
                    f"\n- {rec.identifier} (score: {rec.score:.2f}, "
                    f"-{diff:.2f} vs best): {rec.reasoning}"
                )

        # Note unscored methods
        unscored = [m for m in methods if m not in [r.identifier for r in method_recommendations]]
        if unscored:
            analysis_parts.append(
                f"\n\nMethods with insufficient fit: {', '.join(unscored)}"
            )

    analysis = "".join(analysis_parts)

    # Convert method strings to MethodIdentifier
    method_ids = [MethodIdentifier(m) for m in methods]

    return ComparisonResult(
        methods=method_ids,
        winner=MethodIdentifier(winner) if winner else None,
        scores=scores,
        analysis=analysis,
    )
