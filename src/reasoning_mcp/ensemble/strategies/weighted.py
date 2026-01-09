"""Weighted voting strategy for ensemble decision-making.

This module implements a weighted voting strategy where votes are multiplied
by both member weight and confidence score. This allows for differentiated
influence among ensemble members based on their reliability and confidence
in their answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class WeightedVoting:
    """Weighted voting strategy.

    Votes are weighted by member weight * confidence score.
    The answer with highest total weighted score wins.

    The confidence of the final answer is calculated as the ratio of the
    winning score to the total weight of all votes, providing a measure of
    both agreement strength and confidence level.

    Examples:
        Create and use the weighted voting strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = WeightedVoting()
        >>> member1 = EnsembleMember(method_name="cot", weight=1.0)
        >>> member2 = EnsembleMember(method_name="tot", weight=2.0)
        >>> member3 = EnsembleMember(method_name="mcts", weight=1.5)
        >>> results = [
        ...     MemberResult(
        ...         member=member1, result="Answer A", confidence=0.9, execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=member2, result="Answer A", confidence=0.95, execution_time_ms=200
        ...     ),
        ...     MemberResult(
        ...         member=member3, result="Answer B", confidence=0.85, execution_time_ms=150
        ...     ),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> # Answer A: (1.0 * 0.9) + (2.0 * 0.95) = 0.9 + 1.9 = 2.8
        >>> # Answer B: (1.5 * 0.85) = 1.275
        >>> assert answer == "Answer A"
        >>> assert details["weighted_scores"]["Answer A"] == 2.8
        >>> assert details["weighted_scores"]["Answer B"] == 1.275

        Handle unanimous agreement with different weights:
        >>> results = [
        ...     MemberResult(member=member1, result="42", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="42", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="42", confidence=0.92, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "42"
        >>> # Total weight: (1.0*0.9) + (2.0*0.95) + (1.5*0.92) = 4.18
        >>> assert abs(details["total_weight"] - 4.18) < 0.01

        Handle zero confidence (contributes zero weight):
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.0, execution_time_ms=100),
        ...     MemberResult(member=member2, result="B", confidence=0.8, execution_time_ms=200),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "B"
        >>> assert details["weighted_scores"]["A"] == 0.0
        >>> assert details["weighted_scores"]["B"] == 1.6
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Aggregate results using weighted voting.

        Each member's vote is weighted by: member.weight * result.confidence
        The answer with the highest total weighted score wins.

        Args:
            results: List of member results to aggregate. Must not be empty.

        Returns:
            Tuple of (winning_answer, confidence, voting_details) where:
            - winning_answer: The answer with highest weighted score
            - confidence: Winning score divided by total weight (0.0 to 1.0)
            - voting_details: Dict containing:
                - weighted_scores: Dict mapping each answer to its total weighted score
                - total_weight: Sum of all weighted votes across all answers
                - winner: The winning answer (same as first return value)

        Raises:
            ValueError: If results list is empty or all weights are zero.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = WeightedVoting()
            >>> member1 = EnsembleMember(method_name="cot", weight=1.0)
            >>> member2 = EnsembleMember(method_name="tot", weight=2.0)
            >>> results = [
            ...     MemberResult(
            ...         member=member1, result="X", confidence=0.8, execution_time_ms=100
            ...     ),
            ...     MemberResult(
            ...         member=member2, result="Y", confidence=0.9, execution_time_ms=200
            ...     ),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> # X: 1.0 * 0.8 = 0.8
            >>> # Y: 2.0 * 0.9 = 1.8
            >>> # Total: 0.8 + 1.8 = 2.6
            >>> assert answer == "Y"
            >>> assert details["weighted_scores"]["X"] == 0.8
            >>> assert details["weighted_scores"]["Y"] == 1.8
            >>> assert details["total_weight"] == 2.6
            >>> assert abs(conf - 1.8/2.6) < 0.01
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Build weighted scores dict: answer -> total_weighted_score
        weighted_scores: dict[str, float] = {}

        for result in results:
            answer = result.result
            # Calculate weight for this vote: member_weight * confidence
            vote_weight = result.member.weight * result.confidence

            # Accumulate weighted score for this answer
            if answer in weighted_scores:
                weighted_scores[answer] += vote_weight
            else:
                weighted_scores[answer] = vote_weight

        # Calculate total weight across all votes
        total_weight = sum(weighted_scores.values())

        # Handle edge case: all zero weights
        if total_weight == 0.0:
            raise ValueError("Cannot aggregate when all weights are zero")

        # Find answer with highest weighted score
        # In case of tie, max() returns the first encountered (dict maintains insertion order)
        winning_answer = max(weighted_scores, key=weighted_scores.get)  # type: ignore[arg-type]
        winning_score = weighted_scores[winning_answer]

        # Calculate confidence as ratio of winning score to total weight
        confidence = winning_score / total_weight if total_weight > 0 else 0.0

        # Build voting details dict
        voting_details: dict[str, Any] = {
            "weighted_scores": weighted_scores,
            "total_weight": total_weight,
            "winner": winning_answer,
        }

        return winning_answer, confidence, voting_details
