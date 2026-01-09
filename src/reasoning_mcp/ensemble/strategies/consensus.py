"""Consensus voting strategy for ensemble decision-making.

This module implements a consensus voting strategy that requires a minimum
agreement threshold to be met before a decision is made. If the threshold
is not met, the strategy returns an uncertainty indicator.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class ConsensusVoting:
    """Consensus voting strategy.

    Requires a minimum agreement threshold to be met for a decision.
    If threshold is not met, returns an uncertainty indicator.

    The threshold represents the minimum proportion of votes that the winning
    answer must receive. For example, threshold=0.5 means the winning answer
    must have at least 50% of the votes.

    Special threshold values:
    - 0.0: Always passes (any answer with votes wins)
    - 1.0: Requires unanimity (all members must agree)

    Examples:
        Create with default threshold (50%):
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = ConsensusVoting()
        >>> assert strategy.threshold == 0.5

        Create with custom threshold:
        >>> strategy = ConsensusVoting(threshold=0.75)
        >>> assert strategy.threshold == 0.75

        Consensus reached:
        >>> strategy = ConsensusVoting(threshold=0.6)
        >>> member1 = EnsembleMember(method_name="cot")
        >>> member2 = EnsembleMember(method_name="tot")
        >>> member3 = EnsembleMember(method_name="mcts")
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="A", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="A", confidence=0.85, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "A"
        >>> assert confidence == 1.0  # Unanimous
        >>> assert details["consensus_reached"] is True

        No consensus:
        >>> strategy = ConsensusVoting(threshold=0.8)
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="A", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="B", confidence=0.85, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "[NO_CONSENSUS]"
        >>> assert confidence == 0.0
        >>> assert details["consensus_reached"] is False
        >>> assert details["reason"].startswith("Consensus not reached")

        Unanimity required:
        >>> strategy = ConsensusVoting(threshold=1.0)
        >>> results = [
        ...     MemberResult(member=member1, result="42", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="42", confidence=0.95, execution_time_ms=200),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "42"
        >>> assert confidence == 1.0
        >>> assert details["consensus_reached"] is True

        Always passes (threshold 0.0):
        >>> strategy = ConsensusVoting(threshold=0.0)
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="B", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="C", confidence=0.85, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "A"  # First in Counter.most_common()
        >>> assert details["consensus_reached"] is True
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize with agreement threshold.

        Args:
            threshold: Minimum agreement threshold (0.0-1.0). The winning answer
                must receive at least this proportion of votes to achieve consensus.

        Raises:
            ValueError: If threshold is not in the range [0.0, 1.0].

        Examples:
            >>> strategy = ConsensusVoting()
            >>> assert strategy.threshold == 0.5

            >>> strategy = ConsensusVoting(threshold=0.75)
            >>> assert strategy.threshold == 0.75

            >>> strategy = ConsensusVoting(threshold=0.0)
            >>> assert strategy.threshold == 0.0

            >>> strategy = ConsensusVoting(threshold=1.0)
            >>> assert strategy.threshold == 1.0

            >>> try:
            ...     ConsensusVoting(threshold=1.5)
            ...     assert False, "Should raise ValueError"
            ... except ValueError as e:
            ...     assert "between 0.0 and 1.0" in str(e)

            >>> try:
            ...     ConsensusVoting(threshold=-0.1)
            ...     assert False, "Should raise ValueError"
            ... except ValueError as e:
            ...     assert "between 0.0 and 1.0" in str(e)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Aggregate requiring consensus above threshold.

        Counts votes for each unique answer and checks if the top answer meets
        or exceeds the consensus threshold. If yes, returns that answer with
        its confidence. If no, returns a special "[NO_CONSENSUS]" marker with
        confidence 0.0.

        Args:
            results: List of member results to aggregate. Must not be empty.

        Returns:
            If consensus reached:
                Tuple of (winning_answer, confidence, voting_details)
            If no consensus:
                Tuple of ("[NO_CONSENSUS]", 0.0, voting_details_with_reason)

            The voting_details dict contains:
            - vote_counts: Dict mapping each answer to its vote count
            - total_votes: Total number of votes cast
            - agreement_proportion: Proportion of votes for the winning answer
            - threshold: The threshold that was required
            - consensus_reached: Boolean indicating if consensus was achieved
            - reason: (Only if no consensus) Explanation of why consensus failed

        Raises:
            ValueError: If results list is empty.

        Examples:
            Consensus reached (2/3 >= 0.5):
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = ConsensusVoting(threshold=0.5)
            >>> member = EnsembleMember(method_name="test")
            >>> results = [
            ...     MemberResult(member=member, result="A", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="A", confidence=0.8, execution_time_ms=100),
            ...     MemberResult(member=member, result="B", confidence=0.7, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "A"
            >>> assert abs(conf - 0.6667) < 0.01
            >>> assert details["consensus_reached"] is True

            No consensus (1/3 < 0.5):
            >>> strategy = ConsensusVoting(threshold=0.5)
            >>> results = [
            ...     MemberResult(member=member, result="A", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="B", confidence=0.8, execution_time_ms=100),
            ...     MemberResult(member=member, result="C", confidence=0.7, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "[NO_CONSENSUS]"
            >>> assert conf == 0.0
            >>> assert details["consensus_reached"] is False
            >>> assert "Consensus not reached" in details["reason"]

            Unanimity required:
            >>> strategy = ConsensusVoting(threshold=1.0)
            >>> results = [
            ...     MemberResult(member=member, result="X", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="X", confidence=0.8, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "X"
            >>> assert conf == 1.0
            >>> assert details["consensus_reached"] is True

            Unanimity failed:
            >>> strategy = ConsensusVoting(threshold=1.0)
            >>> results = [
            ...     MemberResult(member=member, result="X", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="Y", confidence=0.8, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer == "[NO_CONSENSUS]"
            >>> assert conf == 0.0
            >>> assert details["consensus_reached"] is False
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Extract result strings from each MemberResult
        answers = [result.result for result in results]

        # Count occurrences using Counter
        vote_counts = Counter(answers)

        # Get the most common answer
        winning_answer, winning_count = vote_counts.most_common(1)[0]

        # Calculate total votes
        total_votes = len(answers)

        # Calculate agreement proportion
        agreement_proportion = winning_count / total_votes if total_votes > 0 else 0.0

        # Check if consensus threshold is met
        consensus_reached = agreement_proportion >= self.threshold

        # Build voting details dict
        voting_details: dict[str, Any] = {
            "vote_counts": dict(vote_counts),
            "total_votes": total_votes,
            "agreement_proportion": agreement_proportion,
            "threshold": self.threshold,
            "consensus_reached": consensus_reached,
        }

        if consensus_reached:
            # Consensus achieved - return the winning answer
            return winning_answer, agreement_proportion, voting_details
        else:
            # No consensus - return uncertainty indicator
            voting_details["reason"] = (
                f"Consensus not reached: top answer '{winning_answer}' "
                f"received {winning_count}/{total_votes} votes "
                f"({agreement_proportion:.1%}), threshold is {self.threshold:.1%}"
            )
            return "[NO_CONSENSUS]", 0.0, voting_details
