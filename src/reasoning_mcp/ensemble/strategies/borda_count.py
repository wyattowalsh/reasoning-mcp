"""Borda count voting strategy for ensemble decision-making.

This module implements a Borda count voting strategy where answers are ranked
and assigned points based on their position. This provides a fair aggregation
method that considers both preference strength and distribution across members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.ensemble import MemberResult


class BordaCountVoting:
    """Borda count voting strategy.

    Ranks answers by confidence and assigns points inversely to rank.
    For N unique answers: 1st place gets N-1 points, 2nd gets N-2, etc.

    Uses each member's confidence to implicitly rank all answers.
    Each member assigns maximum points to their own answer based on confidence,
    and distributes remaining points to other answers inversely proportional
    to their distance from the member's answer.

    The Borda count provides a more nuanced aggregation than simple majority
    voting by capturing preference intensity across all options.

    Examples:
        Create and use the Borda count strategy:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> strategy = BordaCountVoting()
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
        >>> # 2 unique answers: A and B
        >>> # Points scale: 1st = 1 point, 2nd = 0 points
        >>> # Member 1: A gets 0.9 points (confidence * 1), B gets 0 (not member's answer)
        >>> # Member 2: A gets 0.95 points, B gets 0
        >>> # Member 3: B gets 0.85 points, A gets 0
        >>> # Total: A = 1.85, B = 0.85
        >>> assert answer == "Answer A"
        >>> assert details["borda_scores"]["Answer A"] == 1.85
        >>> assert details["borda_scores"]["Answer B"] == 0.85

        Handle unanimous agreement:
        >>> results = [
        ...     MemberResult(member=member1, result="42", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member2, result="42", confidence=0.95, execution_time_ms=200),
        ...     MemberResult(member=member3, result="42", confidence=0.92, execution_time_ms=150),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert answer == "42"
        >>> # Only one unique answer, all members agree
        >>> assert confidence == 1.0  # Perfect agreement

        Handle diverse opinions with multiple answers:
        >>> member4 = EnsembleMember(method_name="react")
        >>> results = [
        ...     MemberResult(member=member1, result="A", confidence=0.8, execution_time_ms=100),
        ...     MemberResult(member=member2, result="B", confidence=0.9, execution_time_ms=100),
        ...     MemberResult(member=member3, result="C", confidence=0.7, execution_time_ms=100),
        ...     MemberResult(member=member4, result="A", confidence=0.85, execution_time_ms=100),
        ... ]
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> # 3 unique answers: A, B, C
        >>> # Points scale: 1st = 2, 2nd = 1, 3rd = 0
        >>> # Answer A appears 2 times (0.8 + 0.85) * 2 = 3.3
        >>> assert answer in ["A", "B", "C"]
        >>> assert "borda_scores" in details
    """

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Aggregate using Borda count scoring.

        Algorithm:
        1. Get all unique answers
        2. For each member, rank answers by closeness to member's answer
           (same answer = highest points, others proportionally lower)
        3. Sum Borda points for each answer
        4. Winner is answer with highest total points

        Args:
            results: List of member results to aggregate. Must not be empty.

        Returns:
            Tuple of (winner, normalized_score, details) where:
            - winner: Answer with highest Borda count score
            - normalized_score: Winner's score divided by max possible score (0.0 to 1.0)
            - details: Dict containing:
                - borda_scores: Dict mapping each answer to its total Borda points
                - points_breakdown: List of dicts showing point assignment per member
                - max_possible: Maximum possible score (for normalization)
                - unique_answers: Number of unique answers

        Raises:
            ValueError: If results list is empty.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = BordaCountVoting()
            >>> member1 = EnsembleMember(method_name="cot")
            >>> member2 = EnsembleMember(method_name="tot")
            >>> results = [
            ...     MemberResult(
            ...         member=member1, result="X", confidence=0.8, execution_time_ms=100
            ...     ),
            ...     MemberResult(
            ...         member=member2, result="Y", confidence=0.9, execution_time_ms=200
            ...     ),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> # 2 unique answers: X and Y
            >>> # Points scale: 1st = 1 point, 2nd = 0 points
            >>> # Member 1: X gets 0.8 * 1 = 0.8, Y gets 0
            >>> # Member 2: Y gets 0.9 * 1 = 0.9, X gets 0
            >>> # Total: X = 0.8, Y = 0.9
            >>> assert answer == "Y"
            >>> assert details["borda_scores"]["X"] == 0.8
            >>> assert details["borda_scores"]["Y"] == 0.9
        """
        # Handle edge case: empty results
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Handle edge case: single result
        if len(results) == 1:
            single_result = results[0]
            return (
                single_result.result,
                single_result.confidence,
                {
                    "borda_scores": {single_result.result: single_result.confidence},
                    "points_breakdown": [
                        {
                            "member": single_result.member.method_name,
                            "points": {single_result.result: single_result.confidence},
                        }
                    ],
                    "max_possible": single_result.confidence,
                    "unique_answers": 1,
                },
            )

        # Step 1: Collect all unique answers
        unique_answers = list({result.result for result in results})
        num_answers = len(unique_answers)

        # Handle edge case: single unique answer (unanimous)
        if num_answers == 1:
            unanimous_answer = unique_answers[0]
            total_confidence = sum(result.confidence for result in results)
            breakdown = [
                {
                    "member": result.member.method_name,
                    "points": {unanimous_answer: result.confidence},
                }
                for result in results
            ]
            return (
                unanimous_answer,
                1.0,  # Perfect agreement
                {
                    "borda_scores": {unanimous_answer: total_confidence},
                    "points_breakdown": breakdown,
                    "max_possible": total_confidence,
                    "unique_answers": 1,
                },
            )

        # Step 2: Initialize Borda scores for each answer
        borda_scores: dict[str, float] = {answer: 0.0 for answer in unique_answers}

        # Track points breakdown for transparency
        points_breakdown: list[dict[str, Any]] = []

        # Step 3: For each member, assign Borda points
        # The points scale is: N-1, N-2, ..., 1, 0 for N unique answers
        # Each member gives max points (N-1) to their answer, scaled by confidence
        # Other answers receive 0 points from this member
        max_points_per_member = num_answers - 1

        for result in results:
            member_answer = result.result
            member_confidence = result.confidence

            # Points awarded by this member
            member_points: dict[str, float] = {answer: 0.0 for answer in unique_answers}

            # The member's own answer gets maximum Borda points, scaled by confidence
            # This represents the member's strongest preference
            borda_points = max_points_per_member * member_confidence
            member_points[member_answer] = borda_points
            borda_scores[member_answer] += borda_points

            # Record this member's contribution
            points_breakdown.append(
                {
                    "member": result.member.method_name,
                    "answer": member_answer,
                    "confidence": member_confidence,
                    "points": dict(member_points),  # Copy for immutability
                }
            )

        # Step 4: Find the winner (answer with highest Borda count)
        if not borda_scores:
            # This should never happen due to earlier checks, but defensive programming
            raise ValueError("No Borda scores computed")

        winning_answer = max(borda_scores, key=borda_scores.get)  # type: ignore[arg-type]
        winning_score = borda_scores[winning_answer]

        # Step 5: Calculate maximum possible score for normalization
        # Max possible is if all members gave their full confidence-weighted points to one answer
        max_possible_score = sum(max_points_per_member * result.confidence for result in results)

        # Step 6: Normalize confidence score
        # Edge case: if max_possible is 0 (all zero confidence), return 0
        if max_possible_score == 0.0:
            normalized_confidence = 0.0
        else:
            normalized_confidence = winning_score / max_possible_score

        # Step 7: Build detailed voting information
        voting_details: dict[str, Any] = {
            "borda_scores": borda_scores,
            "points_breakdown": points_breakdown,
            "max_possible": max_possible_score,
            "unique_answers": num_answers,
        }

        return winning_answer, normalized_confidence, voting_details
