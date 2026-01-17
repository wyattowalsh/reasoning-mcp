"""Tests for BestScoreVoting ensemble strategy.

This module tests the BestScoreVoting class, which selects the ensemble member
result with the highest confidence score rather than counting votes.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemble.strategies.best_score import BestScoreVoting
from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult


class TestBestScoreVoting:
    """Test suite for BestScoreVoting strategy."""

    def test_basic_best_score_selection(self) -> None:
        """Test basic functionality - selects result with highest confidence."""
        strategy = BestScoreVoting()

        # Create members
        member1 = EnsembleMember(method_name="cot")
        member2 = EnsembleMember(method_name="tot")
        member3 = EnsembleMember(method_name="mcts")

        # Create results with different confidence scores
        results = [
            MemberResult(member=member1, result="Answer A", confidence=0.85, execution_time_ms=100),
            MemberResult(member=member2, result="Answer B", confidence=0.95, execution_time_ms=200),
            MemberResult(member=member3, result="Answer C", confidence=0.75, execution_time_ms=150),
        ]

        # Aggregate
        answer, confidence, details = strategy.aggregate(results)

        # Assertions
        assert answer == "Answer B"
        assert confidence == 0.95
        assert details["winner_member"] == "tot"
        assert abs(details["score_gap"] - 0.10) < 0.01
        assert details["second_best_score"] == 0.85
        assert details["total_members"] == 3
        assert len(details["all_scores"]) == 3

    def test_unanimous_confidence(self) -> None:
        """Test when all members have equal confidence - should pick first."""
        strategy = BestScoreVoting()

        member1 = EnsembleMember(method_name="m1")
        member2 = EnsembleMember(method_name="m2")
        member3 = EnsembleMember(method_name="m3")

        results = [
            MemberResult(member=member1, result="A", confidence=0.8, execution_time_ms=100),
            MemberResult(member=member2, result="B", confidence=0.8, execution_time_ms=200),
            MemberResult(member=member3, result="C", confidence=0.8, execution_time_ms=150),
        ]

        answer, confidence, details = strategy.aggregate(results)

        # When tied, should pick first encountered
        assert answer == "A"
        assert confidence == 0.8
        assert details["winner_member"] == "m1"
        assert details["score_gap"] == 0.0
        assert details["second_best_score"] == 0.8

    def test_single_result(self) -> None:
        """Test with only one result - should return it."""
        strategy = BestScoreVoting()

        member = EnsembleMember(method_name="only")
        results = [
            MemberResult(member=member, result="Only Answer", confidence=0.9, execution_time_ms=100)
        ]

        answer, confidence, details = strategy.aggregate(results)

        assert answer == "Only Answer"
        assert confidence == 0.9
        assert details["winner_member"] == "only"
        assert details["score_gap"] == 0.0
        assert details["second_best_score"] is None
        assert details["total_members"] == 1

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results list raises ValueError."""
        strategy = BestScoreVoting()

        with pytest.raises(ValueError, match="Cannot aggregate empty results list"):
            strategy.aggregate([])

    def test_two_results_tie(self) -> None:
        """Test with two results having same confidence - picks first."""
        strategy = BestScoreVoting()

        member1 = EnsembleMember(method_name="first")
        member2 = EnsembleMember(method_name="second")

        results = [
            MemberResult(member=member1, result="A", confidence=0.9, execution_time_ms=100),
            MemberResult(member=member2, result="B", confidence=0.9, execution_time_ms=200),
        ]

        answer, confidence, details = strategy.aggregate(results)

        assert answer == "A"  # First encountered wins tie
        assert confidence == 0.9
        assert details["winner_member"] == "first"
        assert details["score_gap"] == 0.0
        assert details["second_best_score"] == 0.9

    def test_extreme_confidence_values(self) -> None:
        """Test with extreme confidence values (0.0 and 1.0)."""
        strategy = BestScoreVoting()

        member1 = EnsembleMember(method_name="low")
        member2 = EnsembleMember(method_name="high")
        member3 = EnsembleMember(method_name="medium")

        results = [
            MemberResult(member=member1, result="Low", confidence=0.0, execution_time_ms=100),
            MemberResult(member=member2, result="High", confidence=1.0, execution_time_ms=200),
            MemberResult(member=member3, result="Mid", confidence=0.5, execution_time_ms=150),
        ]

        answer, confidence, details = strategy.aggregate(results)

        assert answer == "High"
        assert confidence == 1.0
        assert details["winner_member"] == "high"
        assert details["score_gap"] == 0.5
        assert details["second_best_score"] == 0.5

    def test_all_scores_tracked(self) -> None:
        """Test that all_scores dict contains all member scores."""
        strategy = BestScoreVoting()

        members = [
            EnsembleMember(method_name="m1"),
            EnsembleMember(method_name="m2"),
            EnsembleMember(method_name="m3"),
            EnsembleMember(method_name="m4"),
        ]

        results = [
            MemberResult(member=members[0], result="A", confidence=0.7, execution_time_ms=100),
            MemberResult(member=members[1], result="B", confidence=0.8, execution_time_ms=100),
            MemberResult(member=members[2], result="C", confidence=0.9, execution_time_ms=100),
            MemberResult(member=members[3], result="D", confidence=0.6, execution_time_ms=100),
        ]

        answer, confidence, details = strategy.aggregate(results)

        assert len(details["all_scores"]) == 4
        assert details["all_scores"]["m1"] == 0.7
        assert details["all_scores"]["m2"] == 0.8
        assert details["all_scores"]["m3"] == 0.9
        assert details["all_scores"]["m4"] == 0.6

    def test_score_gap_calculation(self) -> None:
        """Test that score gap is calculated correctly."""
        strategy = BestScoreVoting()

        member1 = EnsembleMember(method_name="first")
        member2 = EnsembleMember(method_name="second")
        member3 = EnsembleMember(method_name="third")

        results = [
            MemberResult(member=member1, result="A", confidence=0.95, execution_time_ms=100),
            MemberResult(member=member2, result="B", confidence=0.70, execution_time_ms=200),
            MemberResult(member=member3, result="C", confidence=0.85, execution_time_ms=150),
        ]

        answer, confidence, details = strategy.aggregate(results)

        # Best is 0.95, second-best is 0.85, gap should be 0.10
        assert abs(details["score_gap"] - 0.10) < 0.01
        assert details["second_best_score"] == 0.85

    def test_different_member_weights_ignored(self) -> None:
        """Test that member weights don't affect best score selection."""
        strategy = BestScoreVoting()

        # Members with different weights
        member1 = EnsembleMember(method_name="heavy", weight=10.0)
        member2 = EnsembleMember(method_name="light", weight=0.1)

        results = [
            # Heavy member has lower confidence
            MemberResult(member=member1, result="A", confidence=0.7, execution_time_ms=100),
            # Light member has higher confidence - should win
            MemberResult(member=member2, result="B", confidence=0.9, execution_time_ms=200),
        ]

        answer, confidence, details = strategy.aggregate(results)

        # Weight is ignored - only confidence matters
        assert answer == "B"
        assert confidence == 0.9
        assert details["winner_member"] == "light"

    def test_multiple_same_answers_different_confidence(self) -> None:
        """Test when multiple members give same answer with different confidence."""
        strategy = BestScoreVoting()

        member1 = EnsembleMember(method_name="m1")
        member2 = EnsembleMember(method_name="m2")
        member3 = EnsembleMember(method_name="m3")

        results = [
            MemberResult(member=member1, result="Answer X", confidence=0.85, execution_time_ms=100),
            MemberResult(member=member2, result="Answer X", confidence=0.95, execution_time_ms=200),
            MemberResult(member=member3, result="Answer Y", confidence=0.75, execution_time_ms=150),
        ]

        answer, confidence, details = strategy.aggregate(results)

        # Should pick the higher confidence even though same answer appears twice
        assert answer == "Answer X"
        assert confidence == 0.95
        assert details["winner_member"] == "m2"

    def test_result_with_zero_confidence(self) -> None:
        """Test that a result with 0.0 confidence can still be selected if it's best."""
        strategy = BestScoreVoting()

        member = EnsembleMember(method_name="only")

        # Single result with 0.0 confidence
        results = [
            MemberResult(member=member, result="Uncertain", confidence=0.0, execution_time_ms=100)
        ]

        answer, confidence, details = strategy.aggregate(results)

        assert answer == "Uncertain"
        assert confidence == 0.0
        assert details["winner_member"] == "only"

    def test_preservation_of_result_content(self) -> None:
        """Test that the actual result content is preserved correctly."""
        strategy = BestScoreVoting()

        member = EnsembleMember(method_name="test")

        # Result with complex text content
        complex_result = (
            "This is a long, detailed answer with\nmultiple lines and special chars: @#$%"
        )
        results = [
            MemberResult(
                member=member, result=complex_result, confidence=0.9, execution_time_ms=100
            )
        ]

        answer, confidence, details = strategy.aggregate(results)

        # Ensure result content is not modified
        assert answer == complex_result
        assert answer.count("\n") == 1
        assert "@#$%" in answer
