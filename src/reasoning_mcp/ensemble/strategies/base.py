"""Base protocol for voting strategies in ensemble reasoning.

This module defines the VotingStrategyProtocol that all voting strategy
implementations must follow. The protocol specifies the interface for
aggregating results from multiple ensemble members into a final decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class VotingStrategyProtocol(Protocol):
    """Protocol for voting strategies in ensemble reasoning.

    A voting strategy defines how to aggregate results from multiple ensemble
    members into a final answer. Each strategy implements different logic for
    combining member results, confidence scores, and voting weights.

    The aggregate method is the core interface that all strategies must implement.
    It takes a list of MemberResult objects and produces a final answer with
    confidence and detailed voting information.

    Examples:
        Implementing a simple majority voting strategy:
        >>> class MajorityVoting:
        ...     def aggregate(
        ...         self, results: list[MemberResult]
        ...     ) -> tuple[str, float, dict[str, Any]]:
        ...         # Count votes for each unique answer
        ...         from collections import Counter
        ...         votes = Counter(r.result for r in results)
        ...         # Get the most common answer
        ...         answer, count = votes.most_common(1)[0]
        ...         # Calculate confidence based on agreement
        ...         confidence = count / len(results)
        ...         # Prepare details
        ...         details = {
        ...             "strategy": "majority",
        ...             "votes": dict(votes),
        ...             "total_members": len(results)
        ...         }
        ...         return answer, confidence, details

        Using a voting strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember
        >>> strategy = MajorityVoting()
        >>> results = [
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="cot"),
        ...         result="Answer A",
        ...         confidence=0.9,
        ...         execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="tot"),
        ...         result="Answer A",
        ...         confidence=0.95,
        ...         execution_time_ms=200
        ...     ),
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="mcts"),
        ...         result="Answer B",
        ...         confidence=0.85,
        ...         execution_time_ms=150
        ...     ),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "Answer A"
        >>> assert confidence == 2/3  # 2 out of 3 members agreed
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Aggregate results from multiple ensemble members.

        This method combines the results from all ensemble members into a single
        final answer. The aggregation logic varies by strategy (majority vote,
        weighted vote, consensus, etc.).

        Args:
            results: List of MemberResult objects from ensemble member executions.
                Each result contains the member's answer, confidence score, and
                execution time.

        Returns:
            A tuple containing:
                - answer (str): The final aggregated answer from the ensemble
                - confidence (float): Confidence score for the final answer (0.0 to 1.0)
                - details (dict[str, Any]): Detailed voting information including:
                    - strategy: Name of the voting strategy used
                    - votes: Vote counts or weights for each unique answer
                    - Any additional strategy-specific information

        Raises:
            ValueError: If results list is empty or contains invalid data
            TypeError: If results contain objects that are not MemberResult instances

        Examples:
            Simple majority voting:
            >>> results = [...]  # List of MemberResult objects
            >>> answer, confidence, details = strategy.aggregate(results)
            >>> print(f"Answer: {answer}, Confidence: {confidence:.2f}")
            Answer: 42, Confidence: 0.85

            Weighted voting:
            >>> # Higher weight members have more influence
            >>> answer, confidence, details = weighted_strategy.aggregate(results)
            >>> print(details["votes"])
            {'Answer A': 3.5, 'Answer B': 1.0}

            Consensus voting:
            >>> # Requires minimum agreement threshold
            >>> answer, confidence, details = consensus_strategy.aggregate(results)
            >>> print(details["agreement_score"])
            0.92
        """
        ...
