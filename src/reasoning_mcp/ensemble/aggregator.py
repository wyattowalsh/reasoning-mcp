"""Ensemble result aggregator for meta-aggregation of multiple runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from reasoning_mcp.models.ensemble import EnsembleResult, MemberResult


class EnsembleAggregator:
    """Aggregates results from multiple ensemble executions.

    Provides meta-level aggregation when running ensembles multiple times
    or combining results from different ensemble configurations.

    Useful for:
    - Running the same ensemble multiple times for stability
    - Combining results from different ensemble strategies
    - Hierarchical ensemble architectures

    Examples:
        Basic usage with multiple ensemble runs:
        >>> aggregator = EnsembleAggregator()
        >>> for _ in range(3):
        ...     result = run_ensemble(query)  # Returns EnsembleResult
        ...     aggregator.add_result(result)
        >>> final = aggregator.aggregate_results()
        >>> assert final.final_answer is not None
        >>> assert 0.0 <= final.confidence <= 1.0

        Combining different ensemble strategies:
        >>> aggregator = EnsembleAggregator()
        >>> majority_result = run_ensemble(query, VotingStrategy.MAJORITY)
        >>> weighted_result = run_ensemble(query, VotingStrategy.WEIGHTED)
        >>> consensus_result = run_ensemble(query, VotingStrategy.CONSENSUS)
        >>> aggregator.add_result(majority_result)
        >>> aggregator.add_result(weighted_result)
        >>> aggregator.add_result(consensus_result)
        >>> meta_result = aggregator.aggregate_results()

        Using explicit results list:
        >>> results = [result1, result2, result3]
        >>> aggregator = EnsembleAggregator()
        >>> final = aggregator.aggregate_results(results=results)
    """

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self._ensemble_results: list[EnsembleResult] = []

    def add_result(self, result: EnsembleResult) -> None:
        """Add an ensemble result to be aggregated.

        Args:
            result: EnsembleResult to include in meta-aggregation

        Examples:
            >>> aggregator = EnsembleAggregator()
            >>> result = EnsembleResult(
            ...     final_answer="42",
            ...     confidence=0.95,
            ...     agreement_score=0.9,
            ...     member_results=[],
            ...     voting_details={}
            ... )
            >>> aggregator.add_result(result)
            >>> assert len(aggregator._ensemble_results) == 1
        """
        self._ensemble_results.append(result)

    def aggregate_results(self, results: list[EnsembleResult] | None = None) -> EnsembleResult:
        """Aggregate multiple ensemble results into a single result.

        Args:
            results: Optional list of results (uses internal if None)

        Returns:
            Meta-aggregated EnsembleResult

        Raises:
            ValueError: If no results are available to aggregate

        Algorithm:
        1. Collect all final_answers from ensemble results
        2. Vote on most common answer (or use weighted by confidence)
        3. Average confidences and agreement scores
        4. Combine all member_results
        5. Return meta-result with combined details

        Examples:
            With internally stored results:
            >>> aggregator = EnsembleAggregator()
            >>> aggregator.add_result(result1)
            >>> aggregator.add_result(result2)
            >>> final = aggregator.aggregate_results()

            With explicit results:
            >>> aggregator = EnsembleAggregator()
            >>> final = aggregator.aggregate_results([result1, result2])
        """
        # Use provided results or internal storage
        ensemble_results = results if results is not None else self._ensemble_results

        if not ensemble_results:
            msg = "No ensemble results to aggregate"
            raise ValueError(msg)

        # Step 1: Collect all final answers with their confidences
        answers_with_confidence: list[tuple[str, float]] = [
            (result.final_answer, result.confidence) for result in ensemble_results
        ]

        # Step 2: Vote on most common answer weighted by confidence
        final_answer = self._vote_weighted_by_confidence(answers_with_confidence)

        # Step 3: Calculate average confidence and agreement
        total_confidence = sum(result.confidence for result in ensemble_results)
        avg_confidence = total_confidence / len(ensemble_results)

        total_agreement = sum(result.agreement_score for result in ensemble_results)
        avg_agreement = total_agreement / len(ensemble_results)

        # Step 4: Combine all member results from all ensemble runs
        all_member_results: list[MemberResult] = []
        for ensemble_result in ensemble_results:
            all_member_results.extend(ensemble_result.member_results)

        # Step 5: Build voting details showing the meta-aggregation process
        answer_counts = Counter(result.final_answer for result in ensemble_results)
        voting_details: dict[str, Any] = {
            "strategy": "meta_aggregation",
            "num_ensemble_runs": len(ensemble_results),
            "answer_distribution": dict(answer_counts),
            "weighted_by_confidence": True,
            "individual_confidences": [result.confidence for result in ensemble_results],
            "individual_agreements": [result.agreement_score for result in ensemble_results],
        }

        # Return the meta-aggregated result
        return EnsembleResult(
            final_answer=final_answer,
            confidence=avg_confidence,
            agreement_score=avg_agreement,
            member_results=all_member_results,
            voting_details=voting_details,
        )

    def _vote_weighted_by_confidence(self, answers_with_confidence: list[tuple[str, float]]) -> str:
        """Select answer using confidence-weighted voting.

        Args:
            answers_with_confidence: List of (answer, confidence) tuples

        Returns:
            The answer with highest total weighted confidence

        Examples:
            >>> aggregator = EnsembleAggregator()
            >>> answers = [("A", 0.9), ("A", 0.8), ("B", 0.7)]
            >>> result = aggregator._vote_weighted_by_confidence(answers)
            >>> assert result == "A"  # Total: A=1.7, B=0.7
        """
        # Accumulate confidence scores for each unique answer
        weighted_votes: dict[str, float] = {}
        for answer, confidence in answers_with_confidence:
            weighted_votes[answer] = weighted_votes.get(answer, 0.0) + confidence

        # Return answer with highest total confidence
        return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def clear(self) -> None:
        """Clear all stored ensemble results.

        Useful for reusing the aggregator with a fresh state.

        Examples:
            >>> aggregator = EnsembleAggregator()
            >>> aggregator.add_result(result1)
            >>> aggregator.add_result(result2)
            >>> assert len(aggregator._ensemble_results) == 2
            >>> aggregator.clear()
            >>> assert len(aggregator._ensemble_results) == 0
        """
        self._ensemble_results.clear()

    @property
    def num_results(self) -> int:
        """Get the number of ensemble results currently stored.

        Returns:
            Number of ensemble results in the aggregator

        Examples:
            >>> aggregator = EnsembleAggregator()
            >>> assert aggregator.num_results == 0
            >>> aggregator.add_result(result)
            >>> assert aggregator.num_results == 1
        """
        return len(self._ensemble_results)
