"""Best score voting strategy for ensemble decision-making.

This module implements a voting strategy that selects the result with the
highest confidence score. Unlike majority voting which counts votes, this
strategy prioritizes quality over quantity by selecting the single best-performing
member result based on confidence scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class BestScoreVoting:
    """Best score voting strategy.

    Selects the result with the highest confidence score from all ensemble members.
    No aggregation or voting - simply picks the single answer with the highest
    confidence value.

    This strategy is useful when:
    - You trust individual member confidence scores
    - Quality matters more than consensus
    - Members have well-calibrated confidence estimates
    - You want the single best answer rather than a vote

    Ties (multiple results with identical max confidence) are broken by selecting
    the first result encountered with the maximum confidence score.

    Examples:
        Create and use the voting strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = BestScoreVoting()
        >>> member1 = EnsembleMember(method_name="cot")
        >>> member2 = EnsembleMember(method_name="tot")
        >>> member3 = EnsembleMember(method_name="mcts")
        >>> results = [
        ...     MemberResult(
        ...         member=member1, result="Answer A", confidence=0.85, execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=member2, result="Answer B", confidence=0.95, execution_time_ms=200
        ...     ),
        ...     MemberResult(
        ...         member=member3, result="Answer C", confidence=0.75, execution_time_ms=150
        ...     ),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "Answer B"
        >>> assert confidence == 0.95
        >>> assert details["winner_member"] == "tot"
        >>> assert details["score_gap"] == 0.10  # Gap to second-best

        Handle unanimous confidence (all equal):
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.8, execution_time_ms=100),
        ...     MemberResult(member=member2, result="B", confidence=0.8, execution_time_ms=200),
        ...     MemberResult(member=member3, result="C", confidence=0.8, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "A"  # First encountered wins tie
        >>> assert confidence == 0.8
        >>> assert details["score_gap"] == 0.0  # No gap when tied

        Handle single result:
        >>> results = [
        ...     MemberResult(member=member1, result="Only", confidence=0.9, execution_time_ms=100),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "Only"
        >>> assert confidence == 0.9
        >>> assert details["second_best_score"] is None  # No second place
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Select result with highest confidence score.

        Args:
            results: List of member results to evaluate. Must not be empty.

        Returns:
            Tuple of (best_answer, best_confidence, details) where:
            - best_answer: The result text from the highest-confidence member
            - best_confidence: The confidence score of the winning result (0.0 to 1.0)
            - details: Dict containing:
                - winner_member: Name of the member that produced the best result
                - all_scores: Dict mapping member names to their confidence scores
                - score_gap: Difference between best and second-best scores
                - second_best_score: Confidence score of second-place result (or None)
                - total_members: Total number of members evaluated

        Raises:
            ValueError: If results list is empty.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = BestScoreVoting()
            >>> member = EnsembleMember(method_name="test")
            >>> results = [
            ...     MemberResult(
            ...         member=EnsembleMember(method_name="m1"),
            ...         result="Answer 1",
            ...         confidence=0.7,
            ...         execution_time_ms=100
            ...     ),
            ...     MemberResult(
            ...         member=EnsembleMember(method_name="m2"),
            ...         result="Answer 2",
            ...         confidence=0.9,
            ...         execution_time_ms=200
            ...     ),
            ...     MemberResult(
            ...         member=EnsembleMember(method_name="m3"),
            ...         result="Answer 3",
            ...         confidence=0.8,
            ...         execution_time_ms=150
            ...     ),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "Answer 2"
            >>> assert conf == 0.9
            >>> assert details["winner_member"] == "m2"
            >>> assert abs(details["score_gap"] - 0.1) < 0.01
            >>> assert details["second_best_score"] == 0.8
            >>> assert details["total_members"] == 3
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Find the result with maximum confidence
        # Using max() with key function - stable for ties (returns first encountered)
        best_result = max(results, key=lambda r: r.confidence)

        # Extract best answer and confidence
        best_answer = best_result.result
        best_confidence = best_result.confidence
        winner_name = best_result.member.method_name

        # Build a dict of all scores for transparency
        all_scores: dict[str, float] = {
            result.member.method_name: result.confidence for result in results
        }

        # Calculate gap to second-best score
        # Sort confidence scores in descending order
        sorted_scores = sorted((r.confidence for r in results), reverse=True)

        if len(sorted_scores) >= 2:
            second_best_score = sorted_scores[1]
            score_gap = best_confidence - second_best_score
        else:
            # Only one result - no second place
            second_best_score = None
            score_gap = 0.0

        # Build details dict
        details: dict[str, Any] = {
            "winner_member": winner_name,
            "all_scores": all_scores,
            "score_gap": score_gap,
            "second_best_score": second_best_score,
            "total_members": len(results),
        }

        return best_answer, best_confidence, details
