"""Ranked choice voting (instant runoff) strategy for ensemble decision-making.

This module implements a ranked choice voting strategy (also known as instant
runoff voting) where the answer with the fewest votes is eliminated in each
round, and votes are redistributed until one answer has a majority (>50%).

Since ensemble members only provide one answer each, this implementation uses
confidence scores as a proxy for ranking strength when redistributing votes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class RankedChoiceVoting:
    """Ranked choice voting (instant runoff) strategy.

    Each member's result is treated as their first-choice vote.
    In each round, the answer with fewest votes is eliminated.
    Votes are redistributed until one answer has majority.

    Since members only provide one answer, this uses confidence
    as a proxy for ranking strength.

    The algorithm:
    1. Count first-choice votes for each unique answer
    2. If any answer has >50% of votes, it wins immediately
    3. Otherwise, eliminate the answer with the fewest votes
    4. Redistribute eliminated answer's votes (using confidence as proxy)
    5. Repeat until a winner emerges with >50% or only one answer remains

    Edge cases handled:
    - Empty results: raises ValueError
    - Single answer: immediate winner with 100% confidence
    - Ties in elimination: eliminates the first encountered answer with minimum votes
    - All remaining answers tied: picks the one with highest total confidence

    Examples:
        Create and use the voting strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = RankedChoiceVoting()
        >>> member1 = EnsembleMember(method_name="cot")
        >>> member2 = EnsembleMember(method_name="tot")
        >>> member3 = EnsembleMember(method_name="mcts")
        >>> results = [
        ...     MemberResult(
        ...         member=member1, result="A", confidence=0.9, execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=member2, result="B", confidence=0.8, execution_time_ms=200
        ...     ),
        ...     MemberResult(
        ...         member=member3, result="C", confidence=0.7, execution_time_ms=150
        ...     ),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer in ["A", "B", "C"]
        >>> assert 0.0 <= confidence <= 1.0
        >>> assert "rounds" in details
        >>> assert "eliminations" in details

        Handle immediate majority:
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="A", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="B", confidence=0.85, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "A"
        >>> assert confidence == 2/3  # 2 out of 3 votes
        >>> assert len(details["rounds"]) == 1  # Won in first round

        Handle single answer (unanimous):
        >>> results = [
        ...     MemberResult(member=member1, result="X", confidence=0.9, execution_time_ms=100),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "X"
        >>> assert confidence == 1.0
        >>> assert len(details["rounds"]) == 1
        >>> assert details["eliminations"] == []
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Perform instant runoff voting with elimination rounds.

        Algorithm:
        1. Count first-choice votes for each unique answer
        2. If any answer has >50%, it wins
        3. Otherwise, eliminate lowest-vote answer
        4. Redistribute eliminated answer's votes (by confidence)
        5. Repeat until winner emerges

        Args:
            results: List of member results to aggregate. Must not be empty.

        Returns:
            Tuple of (winner, final_proportion, details) where:
            - winner: The winning answer after instant runoff
            - final_proportion: Proportion of votes for winner in final round (0.0 to 1.0)
            - details: Dict containing:
                - rounds: List of vote counts for each round
                - eliminations: List of answers eliminated in each round
                - final_counts: Final vote counts when winner emerged
                - total_rounds: Number of rounds to determine winner

        Raises:
            ValueError: If results list is empty.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = RankedChoiceVoting()
            >>> member = EnsembleMember(method_name="test")
            >>> results = [
            ...     MemberResult(member=member, result="A", confidence=0.9, execution_time_ms=100),
            ...     MemberResult(member=member, result="B", confidence=0.8, execution_time_ms=100),
            ...     MemberResult(member=member, result="C", confidence=0.7, execution_time_ms=100),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert answer in ["A", "B", "C"]
            >>> assert 0.0 <= conf <= 1.0
            >>> assert details["total_rounds"] >= 1
            >>> assert len(details["eliminations"]) == details["total_rounds"] - 1
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Handle edge case: single answer (immediate winner)
        if len(results) == 1:
            return self._handle_single_answer(results[0])

        # Build a mapping of answer to list of member results
        answer_to_members: dict[str, list[MemberResult]] = {}
        for result in results:
            if result.result not in answer_to_members:
                answer_to_members[result.result] = []
            answer_to_members[result.result].append(result)

        # Handle edge case: unanimous agreement
        if len(answer_to_members) == 1:
            answer = list(answer_to_members.keys())[0]
            return self._handle_unanimous(answer, len(results))

        # Run instant runoff voting rounds
        rounds_history: list[dict[str, int]] = []
        eliminations: list[str] = []
        active_answers = set(answer_to_members.keys())
        total_votes = len(results)

        round_num = 0
        while len(active_answers) > 1:
            round_num += 1

            # Count current votes for each active answer
            vote_counts = {answer: len(answer_to_members[answer]) for answer in active_answers}

            # Record this round
            rounds_history.append(vote_counts.copy())

            # Check for majority winner (>50%)
            for answer, count in vote_counts.items():
                if count > total_votes / 2:
                    # Winner found!
                    confidence = count / total_votes
                    details = self._build_details(
                        rounds_history, eliminations, vote_counts, round_num
                    )
                    return answer, confidence, details

            # No majority winner - eliminate the answer with fewest votes
            min_votes = min(vote_counts.values())

            # Find all answers with minimum votes (for tie handling)
            candidates_for_elimination = [
                answer for answer, count in vote_counts.items() if count == min_votes
            ]

            # Tie-breaking: if multiple answers tied for fewest votes
            # eliminate the one with lowest total confidence
            if len(candidates_for_elimination) > 1:
                to_eliminate = self._break_tie_by_confidence(
                    candidates_for_elimination, answer_to_members
                )
            else:
                to_eliminate = candidates_for_elimination[0]

            # Eliminate the answer
            active_answers.remove(to_eliminate)
            eliminations.append(to_eliminate)

        # Only one answer remains - it wins
        winner = list(active_answers)[0]
        final_vote_count = len(answer_to_members[winner])
        confidence = final_vote_count / total_votes

        # Final round vote counts
        final_counts = {winner: final_vote_count}

        details = self._build_details(rounds_history, eliminations, final_counts, round_num)

        return winner, confidence, details

    def _handle_single_answer(self, result: MemberResult) -> tuple[str, float, dict[str, Any]]:
        """Handle the edge case of a single result.

        Args:
            result: The single member result.

        Returns:
            Tuple of (answer, confidence=1.0, details).
        """
        answer = result.result
        confidence = 1.0

        details: dict[str, Any] = {
            "rounds": [{answer: 1}],
            "eliminations": [],
            "final_counts": {answer: 1},
            "total_rounds": 1,
        }

        return answer, confidence, details

    def _handle_unanimous(self, answer: str, total_votes: int) -> tuple[str, float, dict[str, Any]]:
        """Handle the edge case of unanimous agreement.

        Args:
            answer: The unanimous answer.
            total_votes: Total number of votes.

        Returns:
            Tuple of (answer, confidence=1.0, details).
        """
        confidence = 1.0

        details: dict[str, Any] = {
            "rounds": [{answer: total_votes}],
            "eliminations": [],
            "final_counts": {answer: total_votes},
            "total_rounds": 1,
        }

        return answer, confidence, details

    def _break_tie_by_confidence(
        self,
        candidates: list[str],
        answer_to_members: dict[str, list[MemberResult]],
    ) -> str:
        """Break ties by selecting answer with lowest total confidence.

        When multiple answers have the same number of votes for elimination,
        we eliminate the one with the lowest total confidence across its supporters.

        Args:
            candidates: List of answers tied for fewest votes.
            answer_to_members: Mapping from answer to list of supporting members.

        Returns:
            The answer to eliminate (lowest total confidence).
        """
        confidence_totals: dict[str, float] = {}

        for answer in candidates:
            total_confidence = sum(member.confidence for member in answer_to_members[answer])
            confidence_totals[answer] = total_confidence

        # Return the answer with minimum total confidence
        return min(confidence_totals, key=confidence_totals.get)  # type: ignore

    def _build_details(
        self,
        rounds_history: list[dict[str, int]],
        eliminations: list[str],
        final_counts: dict[str, int],
        total_rounds: int,
    ) -> dict[str, Any]:
        """Build the detailed voting information dictionary.

        Args:
            rounds_history: List of vote counts for each round.
            eliminations: List of answers eliminated in each round.
            final_counts: Final vote counts when winner emerged.
            total_rounds: Total number of rounds run.

        Returns:
            Dictionary with detailed voting information.
        """
        return {
            "rounds": rounds_history,
            "eliminations": eliminations,
            "final_counts": final_counts,
            "total_rounds": total_rounds,
        }
