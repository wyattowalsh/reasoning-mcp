"""Majority voting strategy for ensemble decision-making.

This module implements a simple majority voting strategy where the answer
that appears most frequently among ensemble members is selected as the
final result. The strategy is transparent, interpretable, and works well
when members are relatively independent.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class MajorityVoting:
    """Simple majority voting strategy.

    Selects the answer that appears most frequently among member results.
    Ties are broken by selecting the first answer that reached the majority count.

    The confidence score is calculated as the proportion of votes received by
    the winning answer. This provides a measure of agreement strength.

    Examples:
        Create and use the voting strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = MajorityVoting()
        >>> member1 = EnsembleMember(method_name="cot")
        >>> member2 = EnsembleMember(method_name="tot")
        >>> member3 = EnsembleMember(method_name="mcts")
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
        >>> assert answer == "Answer A"
        >>> assert confidence == 2/3  # 2 out of 3 votes
        >>> assert details["total_votes"] == 3
        >>> assert details["vote_counts"]["Answer A"] == 2

        Handle unanimous agreement:
        >>> results = [
        ...     MemberResult(member=member1, result="42", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="42", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="42", confidence=0.92, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "42"
        >>> assert confidence == 1.0  # Unanimous
        >>> assert details["margin"] == 3

        Handle tie (picks first encountered):
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="B", confidence=0.95, execution_time_ms=200),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "A"  # First encountered wins tie
        >>> assert confidence == 0.5
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Aggregate results using simple majority voting.

        Args:
            results: List of member results to aggregate. Must not be empty.

        Returns:
            Tuple of (winning_answer, confidence, voting_details) where:
            - winning_answer: The most common answer among all results
            - confidence: Proportion of votes for the winning answer (0.0 to 1.0)
            - voting_details: Dict containing:
                - vote_counts: Dict mapping each answer to its vote count
                - total_votes: Total number of votes cast
                - margin: Number of votes for the winner

        Raises:
            ValueError: If results list is empty.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = MajorityVoting()
            >>> member = EnsembleMember(method_name="test")
            >>> results = [
            ...     MemberResult(member=member, result="X", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="X", confidence=0.8, execution_time_ms=100),
            ...     MemberResult(member=member, result="Y", confidence=0.7, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "X"
            >>> assert abs(conf - 0.6667) < 0.01
            >>> assert details["vote_counts"] == {"X": 2, "Y": 1}
            >>> assert details["total_votes"] == 3
            >>> assert details["margin"] == 2
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Extract result strings from each MemberResult
        answers = [result.result for result in results]

        # Count occurrences using Counter
        vote_counts = Counter(answers)

        # Select the most common answer
        # most_common(1) returns [(answer, count)] for the top answer
        # In case of tie, Counter.most_common() returns the first encountered
        winning_answer, winning_count = vote_counts.most_common(1)[0]

        # Calculate total votes
        total_votes = len(answers)

        # Calculate confidence as proportion of votes
        confidence = winning_count / total_votes if total_votes > 0 else 0.0

        # Build voting details dict
        voting_details: dict[str, Any] = {
            "vote_counts": dict(vote_counts),  # Convert Counter to regular dict
            "total_votes": total_votes,
            "margin": winning_count,
        }

        return winning_answer, confidence, voting_details
