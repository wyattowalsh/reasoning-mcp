"""
Comprehensive tests for the evaluate() tool function.

This module provides complete test coverage for the evaluate() function:
- Testing with and without criteria parameter
- Testing return type is EvaluationReport
- Testing field values and validation
- Testing async execution
- Testing different evaluation criteria
"""

import pytest

from reasoning_mcp.models.tools import EvaluationReport
from reasoning_mcp.tools.evaluate import evaluate

# ============================================================================
# Test Basic Execution
# ============================================================================


class TestEvaluateBasic:
    """Test basic evaluate() function behavior."""

    @pytest.mark.asyncio
    async def test_evaluate_is_async(self):
        """Test that evaluate() is an async function."""
        result = evaluate(
            session_id="test-session-123",
        )
        # Should return a coroutine
        assert hasattr(result, "__await__")
        # Await and verify it completes
        output = await result
        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_returns_evaluation_report(self):
        """Test that evaluate() returns EvaluationReport type."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_minimal_parameters(self):
        """Test evaluate() with only required parameter (session_id)."""
        output = await evaluate(
            session_id="test-session-456",
        )

        # Verify required fields are present
        assert output.session_id is not None
        assert isinstance(output.overall_score, float)
        assert isinstance(output.coherence_score, float)
        assert isinstance(output.depth_score, float)
        assert isinstance(output.coverage_score, float)

    @pytest.mark.asyncio
    async def test_evaluate_without_criteria(self):
        """Test evaluate() without criteria (comprehensive evaluation)."""
        output = await evaluate(
            session_id="test-session-789",
            criteria=None,
        )

        # Should perform comprehensive evaluation
        assert isinstance(output, EvaluationReport)
        assert output.session_id == "test-session-789"


# ============================================================================
# Test With Criteria
# ============================================================================


class TestEvaluateWithCriteria:
    """Test evaluate() with different criteria."""

    @pytest.mark.asyncio
    async def test_evaluate_with_single_criterion(self):
        """Test evaluate() with a single criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["coherence"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_multiple_criteria(self):
        """Test evaluate() with multiple criteria."""
        output = await evaluate(
            session_id="test-session-456",
            criteria=["coherence", "depth", "coverage"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_coherence_criterion(self):
        """Test evaluate() with coherence criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["coherence"],
        )

        assert output.coherence_score is not None
        assert isinstance(output.coherence_score, float)

    @pytest.mark.asyncio
    async def test_evaluate_with_depth_criterion(self):
        """Test evaluate() with depth criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["depth"],
        )

        assert output.depth_score is not None
        assert isinstance(output.depth_score, float)

    @pytest.mark.asyncio
    async def test_evaluate_with_coverage_criterion(self):
        """Test evaluate() with coverage criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["coverage"],
        )

        assert output.coverage_score is not None
        assert isinstance(output.coverage_score, float)

    @pytest.mark.asyncio
    async def test_evaluate_with_method_diversity_criterion(self):
        """Test evaluate() with method_diversity criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["method_diversity"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_confidence_trend_criterion(self):
        """Test evaluate() with confidence_trend criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["confidence_trend"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_branching_quality_criterion(self):
        """Test evaluate() with branching_quality criterion."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["branching_quality"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_criteria_list(self):
        """Test evaluate() with empty criteria list."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=[],
        )

        # Empty criteria list should still work (maybe default to comprehensive)
        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_all_criteria(self):
        """Test evaluate() with all possible criteria."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=[
                "coherence",
                "depth",
                "coverage",
                "method_diversity",
                "confidence_trend",
                "branching_quality",
            ],
        )

        assert isinstance(output, EvaluationReport)


# ============================================================================
# Test Output Structure
# ============================================================================


class TestEvaluateOutput:
    """Test the structure and content of EvaluationReport."""

    @pytest.mark.asyncio
    async def test_evaluate_output_has_session_id(self):
        """Test that EvaluationReport has session_id field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "session_id")
        assert isinstance(output.session_id, str)
        assert output.session_id == "test-session-123"

    @pytest.mark.asyncio
    async def test_evaluate_output_has_overall_score(self):
        """Test that EvaluationReport has overall_score field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "overall_score")
        assert isinstance(output.overall_score, float)
        assert 0.0 <= output.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_output_has_coherence_score(self):
        """Test that EvaluationReport has coherence_score field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "coherence_score")
        assert isinstance(output.coherence_score, float)
        assert 0.0 <= output.coherence_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_output_has_depth_score(self):
        """Test that EvaluationReport has depth_score field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "depth_score")
        assert isinstance(output.depth_score, float)
        assert 0.0 <= output.depth_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_output_has_coverage_score(self):
        """Test that EvaluationReport has coverage_score field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "coverage_score")
        assert isinstance(output.coverage_score, float)
        assert 0.0 <= output.coverage_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_output_has_insights(self):
        """Test that EvaluationReport has insights field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "insights")
        assert isinstance(output.insights, list)

    @pytest.mark.asyncio
    async def test_evaluate_output_has_recommendations(self):
        """Test that EvaluationReport has recommendations field."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert hasattr(output, "recommendations")
        assert isinstance(output.recommendations, list)


# ============================================================================
# Test Score Validation
# ============================================================================


class TestEvaluateScoreValidation:
    """Test score validation in EvaluationReport."""

    @pytest.mark.asyncio
    async def test_evaluate_overall_score_in_range(self):
        """Test that overall_score is in valid range [0.0, 1.0]."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert 0.0 <= output.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_coherence_score_in_range(self):
        """Test that coherence_score is in valid range [0.0, 1.0]."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert 0.0 <= output.coherence_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_depth_score_in_range(self):
        """Test that depth_score is in valid range [0.0, 1.0]."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert 0.0 <= output.depth_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_coverage_score_in_range(self):
        """Test that coverage_score is in valid range [0.0, 1.0]."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert 0.0 <= output.coverage_score <= 1.0


# ============================================================================
# Test Insights and Recommendations
# ============================================================================


class TestEvaluateInsightsRecommendations:
    """Test insights and recommendations in EvaluationReport."""

    @pytest.mark.asyncio
    async def test_evaluate_insights_is_list(self):
        """Test that insights is a list."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert isinstance(output.insights, list)

    @pytest.mark.asyncio
    async def test_evaluate_insights_contains_strings(self):
        """Test that insights list contains strings if not empty."""
        output = await evaluate(
            session_id="test-session-123",
        )

        for insight in output.insights:
            assert isinstance(insight, str)

    @pytest.mark.asyncio
    async def test_evaluate_recommendations_is_list(self):
        """Test that recommendations is a list."""
        output = await evaluate(
            session_id="test-session-123",
        )

        assert isinstance(output.recommendations, list)

    @pytest.mark.asyncio
    async def test_evaluate_recommendations_contains_strings(self):
        """Test that recommendations list contains strings if not empty."""
        output = await evaluate(
            session_id="test-session-123",
        )

        for recommendation in output.recommendations:
            assert isinstance(recommendation, str)


# ============================================================================
# Test Input Validation
# ============================================================================


class TestEvaluateInputValidation:
    """Test input validation for evaluate() function."""

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_session_id(self):
        """Test evaluate() with empty session_id string."""
        output = await evaluate(
            session_id="",
        )

        # Should still execute (might fail or handle gracefully)
        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_long_session_id(self):
        """Test evaluate() with very long session_id."""
        long_session_id = "test-session-" + "x" * 1000

        output = await evaluate(
            session_id=long_session_id,
        )

        assert isinstance(output, EvaluationReport)
        assert output.session_id == long_session_id

    @pytest.mark.asyncio
    async def test_evaluate_with_special_characters_in_session_id(self):
        """Test evaluate() with special characters in session_id."""
        special_session_id = "test-session-@#$%^&*()"

        output = await evaluate(
            session_id=special_session_id,
        )

        assert isinstance(output, EvaluationReport)


# ============================================================================
# Test Session ID Matching
# ============================================================================


class TestEvaluateSessionIdMatching:
    """Test that output session_id matches input."""

    @pytest.mark.asyncio
    async def test_evaluate_session_id_matches_input(self):
        """Test that output session_id matches input session_id."""
        input_session_id = "my-test-session-123"

        output = await evaluate(
            session_id=input_session_id,
        )

        assert output.session_id == input_session_id

    @pytest.mark.asyncio
    async def test_evaluate_preserves_session_id_format(self):
        """Test that session_id format is preserved."""
        session_ids = [
            "simple",
            "with-dashes",
            "with_underscores",
            "WithMixedCase",
            "with-123-numbers",
        ]

        for session_id in session_ids:
            output = await evaluate(session_id=session_id)
            assert output.session_id == session_id


# ============================================================================
# Test Idempotency
# ============================================================================


class TestEvaluateIdempotency:
    """Test idempotency and consistency of evaluate() function."""

    @pytest.mark.asyncio
    async def test_evaluate_multiple_calls_same_session(self):
        """Test multiple calls with same session_id."""
        session_id = "shared-session-123"

        output1 = await evaluate(session_id=session_id)
        output2 = await evaluate(session_id=session_id)

        # Should evaluate the same session
        assert output1.session_id == output2.session_id
        assert output1.session_id == session_id

    @pytest.mark.asyncio
    async def test_evaluate_different_sessions(self):
        """Test evaluation of different sessions."""
        output1 = await evaluate(session_id="session-1")
        output2 = await evaluate(session_id="session-2")

        # Should have different session IDs
        assert output1.session_id != output2.session_id


# ============================================================================
# Test Criteria Combinations
# ============================================================================


class TestEvaluateCriteriaCombinations:
    """Test various combinations of evaluation criteria."""

    @pytest.mark.asyncio
    async def test_evaluate_core_metrics_only(self):
        """Test evaluation with only core metrics (coherence, depth, coverage)."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["coherence", "depth", "coverage"],
        )

        assert isinstance(output, EvaluationReport)
        assert output.coherence_score >= 0.0
        assert output.depth_score >= 0.0
        assert output.coverage_score >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_advanced_metrics_only(self):
        """Test evaluation with only advanced metrics."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["method_diversity", "confidence_trend", "branching_quality"],
        )

        assert isinstance(output, EvaluationReport)

    @pytest.mark.asyncio
    async def test_evaluate_duplicate_criteria(self):
        """Test evaluation with duplicate criteria in list."""
        output = await evaluate(
            session_id="test-session-123",
            criteria=["coherence", "coherence", "depth"],
        )

        # Should handle duplicates gracefully
        assert isinstance(output, EvaluationReport)


# ============================================================================
# Test Report Completeness
# ============================================================================


class TestEvaluateReportCompleteness:
    """Test completeness of evaluation reports."""

    @pytest.mark.asyncio
    async def test_evaluate_report_has_all_required_fields(self):
        """Test that report has all required fields."""
        output = await evaluate(
            session_id="test-session-123",
        )

        # Required fields
        assert output.session_id is not None
        assert output.overall_score is not None
        assert output.coherence_score is not None
        assert output.depth_score is not None
        assert output.coverage_score is not None
        assert output.insights is not None
        assert output.recommendations is not None

    @pytest.mark.asyncio
    async def test_evaluate_report_is_frozen(self):
        """Test that EvaluationReport is frozen (immutable)."""
        from pydantic import ValidationError

        output = await evaluate(
            session_id="test-session-123",
        )

        # Should not be able to modify frozen fields
        with pytest.raises(ValidationError):
            output.overall_score = 0.99  # type: ignore[misc]
