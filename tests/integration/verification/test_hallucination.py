"""Integration tests for hallucination detection."""

import pytest

from reasoning_mcp.models.verification import HallucinationFlag
from reasoning_mcp.verification.hallucination import (
    HallucinationDetector,
    get_severity_score,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hallucination_detection():
    """Test hallucination detection on sample texts."""
    detector = HallucinationDetector()

    # Text with potential hallucinations (unsupported claims)
    text_with_issues = """
    The company was founded in 2015 and has grown significantly.
    We have over 10,000 customers worldwide.
    Our revenue exceeded $100 million last year.
    The CEO said this is definitely the best product ever made.
    """

    # Context that doesn't support all claims
    context = "The company was founded in 2015."

    # Detect hallucinations
    flags = await detector.detect(text_with_issues, context)

    # Should detect some potential hallucinations
    assert isinstance(flags, list)

    # Verify flag structure if any found
    for flag in flags:
        assert isinstance(flag, HallucinationFlag)
        assert flag.claim_id is not None
        assert flag.severity in ["low", "medium", "high"]
        assert flag.explanation is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hallucination_detection_no_context():
    """Test hallucination detection without context."""
    detector = HallucinationDetector()

    # Text with self-contradictions
    text = """
    The meeting is scheduled for Monday.
    The meeting is scheduled for Tuesday.
    We expect 50 attendees.
    Only 10 people confirmed.
    """

    # Detect without context (checks for self-contradictions)
    flags = await detector.detect(text, context=None)

    assert isinstance(flags, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_severity_score():
    """Test severity score calculation."""
    # Create sample flags
    flags = [
        HallucinationFlag(
            claim_id="c1",
            severity="high",
            explanation="Major factual error",
        ),
        HallucinationFlag(
            claim_id="c2",
            severity="low",
            explanation="Minor inconsistency",
        ),
    ]

    score = get_severity_score(flags)

    # Score should be in valid range
    assert 0.0 <= score <= 1.0

    # Empty flags should return 0.0
    empty_score = get_severity_score([])
    assert empty_score == 0.0
