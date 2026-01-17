"""Tests for verification reporting module."""

from __future__ import annotations

import json

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from reasoning_mcp.verification.reporting import VerificationReporter


class TestVerificationReporter:
    """Test suite for VerificationReporter."""

    def test_reporter(self) -> None:
        """Test basic reporter functionality with to_markdown and to_json."""
        # Create sample claims
        claim1 = Claim(
            claim_id="c1",
            text="The Earth orbits the Sun",
            claim_type=ClaimType.FACTUAL,
            source_span=(0, 25),
            confidence=0.95,
        )
        claim2 = Claim(
            claim_id="c2",
            text="2 + 2 = 4",
            claim_type=ClaimType.NUMERICAL,
            source_span=(26, 35),
            confidence=1.0,
        )

        # Create evidence
        evidence = Evidence(
            evidence_id="e1",
            source=EvidenceSource.EXTERNAL,
            content="NASA confirms Earth's orbital period",
            relevance_score=0.95,
            url="https://nasa.gov/earth",
        )

        # Create verification results
        result1 = VerificationResult(
            claim=claim1,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[evidence],
            reasoning="Multiple authoritative sources confirm this fact",
        )

        # Create verification report
        report = VerificationReport(
            report_id="report_001",
            original_text="The Earth orbits the Sun. 2 + 2 = 4",
            claims=[claim1, claim2],
            results=[result1],
            overall_accuracy=0.85,
            flagged_claims=["c2"],
        )

        # Create reporter
        reporter = VerificationReporter()

        # Test to_markdown
        markdown = reporter.to_markdown(report)
        assert isinstance(markdown, str)
        assert "# Verification Report: report_001" in markdown
        assert "## Overall Accuracy" in markdown
        assert "85.00%" in markdown
        assert "## Extracted Claims" in markdown
        assert "c1" in markdown
        assert "c2" in markdown
        assert "The Earth orbits the Sun" in markdown
        assert "## Verification Results" in markdown
        assert "verified" in markdown
        assert "## Flagged Claims" in markdown

        # Test to_json
        json_str = reporter.to_json(report)
        assert isinstance(json_str, str)

        # Parse JSON and verify it's valid
        parsed = json.loads(json_str)
        assert parsed["report_id"] == "report_001"
        assert parsed["overall_accuracy"] == 0.85
        assert len(parsed["claims"]) == 2
        assert len(parsed["results"]) == 1
        assert len(parsed["flagged_claims"]) == 1

        # Verify enum serialization
        assert parsed["claims"][0]["claim_type"] == "factual"
        assert parsed["results"][0]["status"] == "verified"
        assert parsed["results"][0]["evidence"][0]["source"] == "external"

    def test_generate_summary(self) -> None:
        """Test generate_summary with mixed verification results."""
        # Create claims
        claim1 = Claim(
            claim_id="c1",
            text="Claim 1 verified",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        claim2 = Claim(
            claim_id="c2",
            text="Claim 2 refuted",
            claim_type=ClaimType.LOGICAL,
            confidence=0.7,
        )
        claim3 = Claim(
            claim_id="c3",
            text="Claim 3 uncertain",
            claim_type=ClaimType.CAUSAL,
            confidence=0.6,
        )

        # Create verification results
        result1 = VerificationResult(
            claim=claim1,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Confirmed by multiple sources",
        )
        result2 = VerificationResult(
            claim=claim2,
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=[],
            reasoning="Contradicts known facts",
        )
        result3 = VerificationResult(
            claim=claim3,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            evidence=[],
            reasoning="Insufficient evidence to verify",
        )

        # Create report with flagged claims
        report = VerificationReport(
            report_id="report_002",
            original_text="Test text with multiple claims",
            claims=[claim1, claim2, claim3],
            results=[result1, result2, result3],
            overall_accuracy=0.65,
            flagged_claims=["c2", "c3"],
        )

        # Generate summary
        reporter = VerificationReporter()
        summary = reporter.generate_summary(report)

        # Verify output contains key metrics
        assert isinstance(summary, str)
        assert "report_002" in summary
        assert "Total claims extracted: 3" in summary
        assert "Verified: 1" in summary
        assert "Refuted: 1" in summary
        assert "Uncertain: 1" in summary
        assert "Overall accuracy: 65.00%" in summary

        # Verify flagged claims are mentioned
        assert "Flagged claims requiring attention:" in summary
        assert "c2" in summary
        assert "c3" in summary
        assert "Claim 2 refuted" in summary
        assert "Claim 3 uncertain" in summary
        assert "Contradicts known facts" in summary
        assert "Insufficient evidence to verify" in summary

    def test_empty_report(self) -> None:
        """Test reporting with empty claims and results."""
        # Create empty report
        report = VerificationReport(
            report_id="empty_report",
            original_text="No claims found",
            claims=[],
            results=[],
            overall_accuracy=0.0,
            flagged_claims=[],
        )

        reporter = VerificationReporter()

        # Test to_markdown with empty report
        markdown = reporter.to_markdown(report)
        assert "# Verification Report: empty_report" in markdown
        assert "_No claims extracted_" in markdown
        assert "_No verification results_" in markdown

        # Test to_json with empty report
        json_str = reporter.to_json(report)
        parsed = json.loads(json_str)
        assert parsed["claims"] == []
        assert parsed["results"] == []
        assert parsed["flagged_claims"] == []

        # Test generate_summary with empty report
        summary = reporter.generate_summary(report)
        assert "Total claims extracted: 0" in summary
        assert "No verification results available" in summary
        assert "No claims flagged for review" in summary
