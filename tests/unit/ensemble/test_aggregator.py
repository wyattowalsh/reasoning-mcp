"""Unit tests for EnsembleAggregator."""

import pytest

from reasoning_mcp.ensemble import EnsembleAggregator
from reasoning_mcp.models.ensemble import EnsembleMember, EnsembleResult, MemberResult


class TestEnsembleAggregator:
    """Test suite for EnsembleAggregator class."""

    @pytest.fixture
    def sample_member(self) -> EnsembleMember:
        """Create a sample ensemble member."""
        return EnsembleMember(method_name="chain_of_thought", weight=1.0)

    @pytest.fixture
    def sample_member_results(self, sample_member: EnsembleMember) -> list[MemberResult]:
        """Create sample member results."""
        return [
            MemberResult(
                member=sample_member,
                result="42",
                confidence=0.9,
                execution_time_ms=100,
            ),
            MemberResult(
                member=sample_member,
                result="42",
                confidence=0.95,
                execution_time_ms=150,
            ),
        ]

    @pytest.fixture
    def sample_ensemble_result(self, sample_member_results: list[MemberResult]) -> EnsembleResult:
        """Create a sample ensemble result."""
        return EnsembleResult(
            final_answer="42",
            confidence=0.92,
            agreement_score=0.88,
            member_results=sample_member_results,
            voting_details={"strategy": "majority", "votes": {"42": 2}},
        )

    def test_initialization(self) -> None:
        """Test aggregator initialization."""
        aggregator = EnsembleAggregator()
        assert aggregator.num_results == 0

    def test_add_result(self, sample_ensemble_result: EnsembleResult) -> None:
        """Test adding a single result."""
        aggregator = EnsembleAggregator()
        aggregator.add_result(sample_ensemble_result)
        assert aggregator.num_results == 1

    def test_add_multiple_results(self, sample_ensemble_result: EnsembleResult) -> None:
        """Test adding multiple results."""
        aggregator = EnsembleAggregator()
        aggregator.add_result(sample_ensemble_result)
        aggregator.add_result(sample_ensemble_result)
        aggregator.add_result(sample_ensemble_result)
        assert aggregator.num_results == 3

    def test_clear(self, sample_ensemble_result: EnsembleResult) -> None:
        """Test clearing stored results."""
        aggregator = EnsembleAggregator()
        aggregator.add_result(sample_ensemble_result)
        aggregator.add_result(sample_ensemble_result)
        assert aggregator.num_results == 2

        aggregator.clear()
        assert aggregator.num_results == 0

    def test_aggregate_results_with_internal_storage(self, sample_member: EnsembleMember) -> None:
        """Test aggregation using internally stored results."""
        aggregator = EnsembleAggregator()

        # Create three ensemble results with different answers
        result1 = EnsembleResult(
            final_answer="A",
            confidence=0.9,
            agreement_score=0.85,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="A",
                    confidence=0.9,
                    execution_time_ms=100,
                )
            ],
            voting_details={"strategy": "majority"},
        )

        result2 = EnsembleResult(
            final_answer="A",
            confidence=0.95,
            agreement_score=0.9,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="A",
                    confidence=0.95,
                    execution_time_ms=110,
                )
            ],
            voting_details={"strategy": "weighted"},
        )

        result3 = EnsembleResult(
            final_answer="B",
            confidence=0.8,
            agreement_score=0.75,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="B",
                    confidence=0.8,
                    execution_time_ms=105,
                )
            ],
            voting_details={"strategy": "consensus"},
        )

        aggregator.add_result(result1)
        aggregator.add_result(result2)
        aggregator.add_result(result3)

        final = aggregator.aggregate_results()

        # Answer "A" should win (0.9 + 0.95 = 1.85 vs 0.8)
        assert final.final_answer == "A"
        assert final.confidence == pytest.approx((0.9 + 0.95 + 0.8) / 3)
        assert final.agreement_score == pytest.approx((0.85 + 0.9 + 0.75) / 3)
        assert len(final.member_results) == 3
        assert final.voting_details["strategy"] == "meta_aggregation"
        assert final.voting_details["num_ensemble_runs"] == 3
        assert final.voting_details["weighted_by_confidence"] is True

    def test_aggregate_results_with_explicit_list(self, sample_member: EnsembleMember) -> None:
        """Test aggregation with explicit results list."""
        aggregator = EnsembleAggregator()

        result1 = EnsembleResult(
            final_answer="X",
            confidence=0.85,
            agreement_score=0.8,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.85,
                    execution_time_ms=90,
                )
            ],
            voting_details={},
        )

        result2 = EnsembleResult(
            final_answer="X",
            confidence=0.9,
            agreement_score=0.85,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.9,
                    execution_time_ms=95,
                )
            ],
            voting_details={},
        )

        # Use explicit results without adding to internal storage
        final = aggregator.aggregate_results(results=[result1, result2])

        assert final.final_answer == "X"
        assert final.confidence == pytest.approx((0.85 + 0.9) / 2)
        assert final.agreement_score == pytest.approx((0.8 + 0.85) / 2)
        assert aggregator.num_results == 0  # Internal storage untouched

    def test_aggregate_results_empty_raises_error(self) -> None:
        """Test that aggregating with no results raises ValueError."""
        aggregator = EnsembleAggregator()

        with pytest.raises(ValueError, match="No ensemble results to aggregate"):
            aggregator.aggregate_results()

    def test_weighted_voting_algorithm(self, sample_member: EnsembleMember) -> None:
        """Test that weighted voting correctly handles confidence scores."""
        aggregator = EnsembleAggregator()

        # Create results where "B" has higher confidence but fewer votes
        result1 = EnsembleResult(
            final_answer="A",
            confidence=0.6,
            agreement_score=0.5,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="A",
                    confidence=0.6,
                    execution_time_ms=100,
                )
            ],
            voting_details={},
        )

        result2 = EnsembleResult(
            final_answer="A",
            confidence=0.65,
            agreement_score=0.55,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="A",
                    confidence=0.65,
                    execution_time_ms=105,
                )
            ],
            voting_details={},
        )

        result3 = EnsembleResult(
            final_answer="B",
            confidence=0.99,
            agreement_score=0.95,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="B",
                    confidence=0.99,
                    execution_time_ms=110,
                )
            ],
            voting_details={},
        )

        aggregator.add_result(result1)
        aggregator.add_result(result2)
        aggregator.add_result(result3)

        final = aggregator.aggregate_results()

        # "A" has total confidence 1.25, "B" has 0.99, so "A" wins
        assert final.final_answer == "A"

    def test_member_results_combination(self, sample_member: EnsembleMember) -> None:
        """Test that member results from all ensemble runs are combined."""
        aggregator = EnsembleAggregator()

        result1 = EnsembleResult(
            final_answer="X",
            confidence=0.9,
            agreement_score=0.85,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.9,
                    execution_time_ms=100,
                ),
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.88,
                    execution_time_ms=110,
                ),
            ],
            voting_details={},
        )

        result2 = EnsembleResult(
            final_answer="X",
            confidence=0.95,
            agreement_score=0.9,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.95,
                    execution_time_ms=105,
                ),
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.92,
                    execution_time_ms=115,
                ),
                MemberResult(
                    member=sample_member,
                    result="X",
                    confidence=0.93,
                    execution_time_ms=120,
                ),
            ],
            voting_details={},
        )

        aggregator.add_result(result1)
        aggregator.add_result(result2)

        final = aggregator.aggregate_results()

        # Should have 2 + 3 = 5 member results total
        assert len(final.member_results) == 5

    def test_voting_details_structure(self, sample_member: EnsembleMember) -> None:
        """Test that voting details contain expected information."""
        aggregator = EnsembleAggregator()

        result1 = EnsembleResult(
            final_answer="A",
            confidence=0.9,
            agreement_score=0.85,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="A",
                    confidence=0.9,
                    execution_time_ms=100,
                )
            ],
            voting_details={},
        )

        result2 = EnsembleResult(
            final_answer="B",
            confidence=0.8,
            agreement_score=0.75,
            member_results=[
                MemberResult(
                    member=sample_member,
                    result="B",
                    confidence=0.8,
                    execution_time_ms=105,
                )
            ],
            voting_details={},
        )

        aggregator.add_result(result1)
        aggregator.add_result(result2)

        final = aggregator.aggregate_results()

        assert "strategy" in final.voting_details
        assert "num_ensemble_runs" in final.voting_details
        assert "answer_distribution" in final.voting_details
        assert "weighted_by_confidence" in final.voting_details
        assert "individual_confidences" in final.voting_details
        assert "individual_agreements" in final.voting_details

        assert final.voting_details["strategy"] == "meta_aggregation"
        assert final.voting_details["num_ensemble_runs"] == 2
        assert final.voting_details["answer_distribution"] == {"A": 1, "B": 1}
        assert final.voting_details["individual_confidences"] == [0.9, 0.8]
        assert final.voting_details["individual_agreements"] == [0.85, 0.75]

    def test_reuse_after_clear(self, sample_ensemble_result: EnsembleResult) -> None:
        """Test that aggregator can be reused after clearing."""
        aggregator = EnsembleAggregator()

        # First use
        aggregator.add_result(sample_ensemble_result)
        result1 = aggregator.aggregate_results()
        assert result1.final_answer == "42"

        # Clear and reuse
        aggregator.clear()
        aggregator.add_result(sample_ensemble_result)
        result2 = aggregator.aggregate_results()
        assert result2.final_answer == "42"
        assert aggregator.num_results == 1
