"""Session evaluation and quality analysis tool.

This module provides the evaluate() function for analyzing the quality of
reasoning sessions, including coherence, depth, and coverage metrics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from reasoning_mcp.models.tools import EvaluationReport

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session
    from reasoning_mcp.models.thought import ThoughtGraph


# Valid evaluation criteria
VALID_CRITERIA = frozenset({
    "coherence",
    "depth",
    "coverage",
    "method_diversity",
    "confidence_trend",
    "branching_quality",
})


def _calculate_coherence_score(graph: ThoughtGraph) -> float:
    """Calculate coherence score based on thought graph connectivity.

    Coherence is measured by:
    - Percentage of thoughts that have parent connections (graph connectivity)
    - Edge-to-node ratio (well-connected graphs have more edges)
    - Whether the graph is a valid DAG (no circular logic)

    Args:
        graph: The thought graph to analyze

    Returns:
        Coherence score between 0.0 and 1.0
    """
    if graph.node_count == 0:
        return 0.0

    # Factor 1: Connectivity ratio (thoughts with parents vs total)
    # Root node doesn't need a parent, so we expect (n-1) connections for n nodes
    nodes_with_parents = sum(
        1 for node in graph.nodes.values() if node.parent_id is not None
    )
    expected_connections = max(graph.node_count - 1, 1)
    connectivity_ratio = min(nodes_with_parents / expected_connections, 1.0)

    # Factor 2: Edge density (edges vs expected edges for a tree)
    # A tree with n nodes has n-1 edges, more edges indicate richer connections
    edge_ratio = min(graph.edge_count / max(graph.node_count, 1), 1.0)

    # Factor 3: DAG validity (no cycles)
    is_dag = graph.validate_dag()
    dag_score = 1.0 if is_dag else 0.5  # Penalize cycles but don't zero out

    # Weighted combination
    coherence = (connectivity_ratio * 0.4) + (edge_ratio * 0.3) + (dag_score * 0.3)
    return min(max(coherence, 0.0), 1.0)


def _calculate_depth_score(graph: ThoughtGraph, session: Session) -> float:
    """Calculate depth score based on reasoning chain depth.

    Depth is measured by:
    - Maximum depth reached in the graph
    - Average depth of leaf nodes (exploration depth)
    - Depth relative to configuration limits

    Args:
        graph: The thought graph to analyze
        session: The session containing configuration

    Returns:
        Depth score between 0.0 and 1.0
    """
    if graph.node_count == 0:
        return 0.0

    max_depth = graph.max_depth
    max_allowed = session.config.max_depth

    # Factor 1: Relative depth reached vs allowed
    depth_utilization = min(max_depth / max_allowed, 1.0) if max_allowed > 0 else 0.0

    # Factor 2: Average depth of leaf nodes
    leaf_ids = graph.leaf_ids
    if leaf_ids:
        leaf_depths = [
            graph.nodes[leaf_id].depth
            for leaf_id in leaf_ids
            if leaf_id in graph.nodes
        ]
        avg_leaf_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0
        # Normalize against max depth
        avg_depth_score = min(avg_leaf_depth / max(max_depth, 1), 1.0)
    else:
        avg_depth_score = 0.0

    # Factor 3: Minimum reasonable depth exploration
    # At least 2-3 levels of depth is good for meaningful reasoning
    min_depth_score = min(max_depth / 3, 1.0) if max_depth > 0 else 0.0

    # Weighted combination
    depth_score = (depth_utilization * 0.4) + (avg_depth_score * 0.3) + (min_depth_score * 0.3)
    return min(max(depth_score, 0.0), 1.0)


def _calculate_coverage_score(graph: ThoughtGraph, session: Session) -> float:
    """Calculate coverage score based on problem space exploration.

    Coverage is measured by:
    - Number of unique branches explored
    - Number of leaf nodes (exploration endpoints)
    - Thought count relative to capacity

    Args:
        graph: The thought graph to analyze
        session: The session containing configuration

    Returns:
        Coverage score between 0.0 and 1.0
    """
    if graph.node_count == 0:
        return 0.0

    # Factor 1: Branch exploration
    branch_count = graph.branch_count
    max_branches = session.config.max_branches
    branch_score = min(branch_count / max(max_branches, 1), 1.0) if branch_count > 0 else 0.3

    # Factor 2: Leaf node spread (more leaves = more exploration endpoints)
    leaf_count = len(graph.leaf_ids)
    # Reasonable number of leaves is around sqrt(n) to n/2
    expected_leaves = max(math.sqrt(graph.node_count), 1)
    leaf_score = min(leaf_count / expected_leaves, 1.0)

    # Factor 3: Thought utilization (relative to capacity)
    thought_utilization = min(graph.node_count / session.config.max_thoughts, 1.0)
    # Apply a scaling factor - moderate utilization (30-70%) is often optimal
    if thought_utilization < 0.3:
        util_score = thought_utilization / 0.3  # Scale up to 1.0 at 30%
    elif thought_utilization <= 0.7:
        util_score = 1.0  # Optimal range
    else:
        util_score = 1.0 - ((thought_utilization - 0.7) * 0.5)  # Slight penalty for over-use

    # Weighted combination
    coverage = (branch_score * 0.35) + (leaf_score * 0.35) + (util_score * 0.3)
    return min(max(coverage, 0.0), 1.0)


def _calculate_method_diversity_score(session: Session) -> float:
    """Calculate method diversity score based on reasoning methods used.

    Args:
        session: The session to analyze

    Returns:
        Diversity score between 0.0 and 1.0
    """
    methods_used = session.metrics.methods_used
    if not methods_used:
        return 0.0

    unique_methods = len(methods_used)
    total_thoughts = session.metrics.total_thoughts

    if total_thoughts == 0:
        return 0.0

    # Score based on unique methods used
    # Using 1-5 unique methods is good, diminishing returns after
    method_score = min(unique_methods / 5, 1.0)

    # Balance score: check if methods are used relatively evenly
    usage_counts = list(methods_used.values())
    avg_usage = total_thoughts / unique_methods
    variance = sum((c - avg_usage) ** 2 for c in usage_counts) / unique_methods
    std_dev = math.sqrt(variance) if variance > 0 else 0
    # Lower variance (more even distribution) is better
    balance_score = max(1.0 - (std_dev / avg_usage), 0.0) if avg_usage > 0 else 0.0

    return (method_score * 0.6) + (balance_score * 0.4)


def _calculate_confidence_trend_score(graph: ThoughtGraph) -> float:
    """Calculate confidence trend score based on how confidence evolves.

    A good trend shows confidence growing or remaining stable as reasoning deepens.

    Args:
        graph: The thought graph to analyze

    Returns:
        Confidence trend score between 0.0 and 1.0
    """
    if graph.node_count < 2:
        return 0.5  # Neutral for insufficient data

    # Get main reasoning path
    main_path = graph.get_main_path()
    if len(main_path) < 2:
        return 0.5

    # Calculate confidence along the main path
    confidences = [
        graph.nodes[node_id].confidence
        for node_id in main_path
        if node_id in graph.nodes
    ]

    if len(confidences) < 2:
        return 0.5

    # Calculate trend: positive trend (increasing confidence) is good
    positive_changes = sum(
        1 for i in range(1, len(confidences)) if confidences[i] >= confidences[i - 1]
    )
    trend_ratio = positive_changes / (len(confidences) - 1)

    # Final confidence level matters too
    final_confidence = confidences[-1] if confidences else 0.0

    return (trend_ratio * 0.5) + (final_confidence * 0.5)


def _calculate_branching_quality_score(graph: ThoughtGraph, session: Session) -> float:
    """Calculate branching quality score based on branch exploration.

    Good branching shows purposeful exploration with reasonable branch depths.

    Args:
        graph: The thought graph to analyze
        session: The session to analyze

    Returns:
        Branching quality score between 0.0 and 1.0
    """
    branch_count = graph.branch_count
    if branch_count == 0:
        # No branches - check if branching was needed based on graph complexity
        if graph.node_count > 5:
            return 0.5  # Could have benefited from branches
        return 0.7  # Small graph doesn't need branches

    # Factor 1: Branch utilization
    max_branches = session.config.max_branches
    utilization = min(branch_count / max_branches, 1.0)

    # Factor 2: Merge operations (merging branches shows synthesis)
    merges = session.metrics.branches_merged
    merge_ratio = min(merges / branch_count, 1.0) if branch_count > 0 else 0.0

    # Factor 3: Pruned branches (some pruning is normal, too much may indicate issues)
    pruned = session.metrics.branches_pruned
    prune_ratio = pruned / branch_count if branch_count > 0 else 0.0
    prune_score = 1.0 - min(prune_ratio, 1.0) * 0.5  # Moderate penalty for pruning

    return (utilization * 0.4) + (merge_ratio * 0.3) + (prune_score * 0.3)


def _generate_insights(
    session: Session,
    coherence: float,
    depth: float,
    coverage: float,
    criteria: set[str] | None,
) -> list[str]:
    """Generate insights based on evaluation scores.

    Args:
        session: The evaluated session
        coherence: Coherence score
        depth: Depth score
        coverage: Coverage score
        criteria: Optional criteria to focus on

    Returns:
        List of insight strings
    """
    insights: list[str] = []
    graph = session.graph

    # General metrics insight
    insights.append(
        f"Session contains {graph.node_count} thoughts across {graph.max_depth} depth levels"
    )

    # Coherence insights
    if criteria is None or "coherence" in criteria:
        if coherence >= 0.8:
            insights.append("Strong logical flow with well-connected reasoning chains")
        elif coherence >= 0.5:
            insights.append("Moderate coherence with some gaps in reasoning connections")
        else:
            insights.append("Weak coherence - reasoning chains may be fragmented")

    # Depth insights
    if criteria is None or "depth" in criteria:
        if depth >= 0.8:
            insights.append("Excellent exploration depth with thorough analysis")
        elif depth >= 0.5:
            insights.append("Moderate depth - some areas could be explored further")
        else:
            insights.append("Shallow exploration - reasoning may lack depth")

    # Coverage insights
    if criteria is None or "coverage" in criteria:
        if coverage >= 0.8:
            insights.append("Comprehensive problem space coverage with diverse exploration")
        elif coverage >= 0.5:
            insights.append("Moderate coverage - some aspects may be underexplored")
        else:
            insights.append("Limited coverage - significant portions of problem space unexplored")

    # Method diversity insights
    if criteria is None or "method_diversity" in criteria:
        methods_used = len(session.metrics.methods_used)
        if methods_used >= 3:
            insights.append(f"Good method diversity with {methods_used} different reasoning approaches")
        elif methods_used >= 1:
            insights.append(f"Limited method diversity - used {methods_used} method(s)")

    # Branching insights
    if criteria is None or "branching_quality" in criteria:
        branch_count = graph.branch_count
        if branch_count > 0:
            insights.append(f"Explored {branch_count} alternative reasoning branches")
        elif graph.node_count > 5:
            insights.append("No branching used despite complex reasoning - consider exploring alternatives")

    return insights


def _generate_recommendations(
    session: Session,
    coherence: float,
    depth: float,
    coverage: float,
    criteria: set[str] | None,
) -> list[str]:
    """Generate recommendations based on evaluation scores.

    Args:
        session: The evaluated session
        coherence: Coherence score
        depth: Depth score
        coverage: Coverage score
        criteria: Optional criteria to focus on

    Returns:
        List of recommendation strings
    """
    recommendations: list[str] = []

    # Coherence recommendations
    if (criteria is None or "coherence" in criteria) and coherence < 0.7:
        recommendations.append(
            "Improve logical connections between thoughts - ensure each step builds on previous ones"
        )
        if not session.graph.validate_dag():
            recommendations.append("Resolve circular logic patterns in reasoning chains")

    # Depth recommendations
    if (criteria is None or "depth" in criteria) and depth < 0.7:
        recommendations.append(
            "Increase reasoning depth - explore key decision points more thoroughly"
        )
        if session.graph.max_depth < 3:
            recommendations.append("Consider using multi-step reasoning methods like Chain-of-Thought")

    # Coverage recommendations
    if (criteria is None or "coverage" in criteria) and coverage < 0.7:
        recommendations.append(
            "Expand problem space coverage - explore alternative approaches"
        )
        if session.graph.branch_count == 0 and session.config.enable_branching:
            recommendations.append("Consider using branching to explore multiple solution paths")

    # Method diversity recommendations
    if criteria is None or "method_diversity" in criteria:
        methods_used = len(session.metrics.methods_used)
        if methods_used < 2:
            recommendations.append(
                "Try incorporating different reasoning methods for more robust analysis"
            )

    # Confidence recommendations
    if criteria is None or "confidence_trend" in criteria:
        avg_confidence = session.metrics.average_confidence
        if avg_confidence < 0.5:
            recommendations.append(
                "Low average confidence - consider validation steps to strengthen conclusions"
            )

    # Branching recommendations
    if criteria is None or "branching_quality" in criteria:
        if session.metrics.branches_pruned > session.metrics.branches_merged:
            recommendations.append(
                "Many branches were pruned - consider more focused branching decisions"
            )

    # Default recommendation if everything looks good
    if not recommendations:
        recommendations.append("Session quality is good - continue with current approach")

    return recommendations


async def evaluate(
    session_id: str,
    criteria: list[str] | None = None,
) -> EvaluationReport:
    """Evaluate the quality of a reasoning session.

    This function analyzes a completed (or in-progress) reasoning session to assess
    the quality of the reasoning process. It examines thought coherence, reasoning
    depth, coverage of the problem space, and provides actionable insights and
    recommendations.

    The evaluation considers:
    - **Coherence**: How well thoughts connect logically and build on each other
    - **Depth**: How thoroughly each aspect of the problem is explored
    - **Coverage**: How comprehensively the problem space is addressed
    - **Quality metrics**: Confidence scores, branching patterns, method diversity

    Args:
        session_id: The unique identifier of the session to evaluate.
        criteria: Optional list of specific evaluation criteria to focus on.
            If None, performs a comprehensive evaluation across all dimensions.
            Possible criteria include:
            - "coherence": Logical flow and consistency
            - "depth": Thoroughness of analysis
            - "coverage": Breadth of exploration
            - "method_diversity": Variety of reasoning methods used
            - "confidence_trend": How confidence evolves over time
            - "branching_quality": Quality of branching and merging decisions

    Returns:
        EvaluationReport containing:
            - session_id: The evaluated session ID
            - overall_score: Overall quality score (0.0-1.0)
            - coherence_score: Logical coherence score (0.0-1.0)
            - depth_score: Reasoning depth score (0.0-1.0)
            - coverage_score: Problem coverage score (0.0-1.0)
            - insights: List of key insights about the session quality
            - recommendations: List of suggestions for improvement

    Raises:
        ValueError: If the session_id is not found or invalid

    Examples:
        Evaluate a completed session:
        >>> report = await evaluate(session_id="session-123")
        >>> print(f"Overall quality: {report.overall_score:.2f}")
        >>> print(f"Coherence: {report.coherence_score:.2f}")
        >>> print(f"Depth: {report.depth_score:.2f}")
        >>> for insight in report.insights:
        ...     print(f"  - {insight}")
        >>> for rec in report.recommendations:
        ...     print(f"  → {rec}")

        Evaluate with specific criteria:
        >>> report = await evaluate(
        ...     session_id="session-456",
        ...     criteria=["coherence", "depth"]
        ... )
        >>> # Focus only on coherence and depth metrics

        Handle evaluation errors:
        >>> try:
        ...     report = await evaluate(session_id="non-existent")
        ... except ValueError as e:
        ...     print(f"Evaluation failed: {e}")
    """
    # Normalize criteria to a set, filtering out invalid ones
    criteria_set: set[str] | None = None
    if criteria is not None:
        # Filter to valid criteria only, deduplicate
        criteria_set = {c for c in criteria if c in VALID_CRITERIA}
        # If empty after filtering, treat as comprehensive evaluation
        if not criteria_set:
            criteria_set = None

    # Try to retrieve session from app context
    session: Session | None = None
    try:
        from reasoning_mcp.server import get_app_context

        ctx = get_app_context()
        session = await ctx.session_manager.get(session_id)
    except RuntimeError:
        # AppContext not initialized - we're running outside server context
        # Return a basic evaluation with the provided session_id
        pass

    # If session not found, return a report indicating no data
    if session is None:
        return EvaluationReport(
            session_id=session_id,
            overall_score=0.0,
            coherence_score=0.0,
            depth_score=0.0,
            coverage_score=0.0,
            insights=[
                f"Session '{session_id}' not found or server not initialized",
                "Unable to perform detailed evaluation without session data",
            ],
            recommendations=[
                "Ensure the session ID is correct and the server is running",
                "Create and populate a session before evaluation",
            ],
        )

    # Get the thought graph
    graph = session.graph

    # Calculate scores based on criteria
    coherence_score = 0.0
    depth_score = 0.0
    coverage_score = 0.0

    # Always calculate core scores (they're required in the report)
    coherence_score = _calculate_coherence_score(graph)
    depth_score = _calculate_depth_score(graph, session)
    coverage_score = _calculate_coverage_score(graph, session)

    # Calculate additional scores if requested
    method_diversity_score = 0.0
    confidence_trend_score = 0.0
    branching_quality_score = 0.0

    if criteria_set is None or "method_diversity" in criteria_set:
        method_diversity_score = _calculate_method_diversity_score(session)

    if criteria_set is None or "confidence_trend" in criteria_set:
        confidence_trend_score = _calculate_confidence_trend_score(graph)

    if criteria_set is None or "branching_quality" in criteria_set:
        branching_quality_score = _calculate_branching_quality_score(graph, session)

    # Calculate overall score as weighted average
    # If specific criteria were requested, weight those more heavily
    if criteria_set is not None:
        score_weights: list[tuple[float, float]] = []
        if "coherence" in criteria_set:
            score_weights.append((coherence_score, 1.0))
        if "depth" in criteria_set:
            score_weights.append((depth_score, 1.0))
        if "coverage" in criteria_set:
            score_weights.append((coverage_score, 1.0))
        if "method_diversity" in criteria_set:
            score_weights.append((method_diversity_score, 0.8))
        if "confidence_trend" in criteria_set:
            score_weights.append((confidence_trend_score, 0.8))
        if "branching_quality" in criteria_set:
            score_weights.append((branching_quality_score, 0.8))

        if score_weights:
            total_weight = sum(w for _, w in score_weights)
            overall_score = sum(s * w for s, w in score_weights) / total_weight
        else:
            # Fallback to core metrics
            overall_score = (coherence_score + depth_score + coverage_score) / 3
    else:
        # Comprehensive evaluation - all metrics with standard weights
        overall_score = (
            coherence_score * 0.25
            + depth_score * 0.25
            + coverage_score * 0.25
            + method_diversity_score * 0.10
            + confidence_trend_score * 0.10
            + branching_quality_score * 0.05
        )

    # Generate insights and recommendations
    insights = _generate_insights(session, coherence_score, depth_score, coverage_score, criteria_set)
    recommendations = _generate_recommendations(
        session, coherence_score, depth_score, coverage_score, criteria_set
    )

    return EvaluationReport(
        session_id=session_id,
        overall_score=min(max(overall_score, 0.0), 1.0),
        coherence_score=coherence_score,
        depth_score=depth_score,
        coverage_score=coverage_score,
        insights=insights,
        recommendations=recommendations,
    )
