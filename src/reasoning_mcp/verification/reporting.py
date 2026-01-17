"""Verification reporting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.models.verification import VerificationReport


class VerificationReporter:
    """Reporter for converting verification results to various formats."""

    def to_markdown(self, report: VerificationReport) -> str:
        """Convert a verification report to markdown format.

        Args:
            report: The verification report to convert

        Returns:
            Markdown-formatted string representation
        """
        lines = []

        # Header with report ID
        lines.append(f"# Verification Report: {report.report_id}")
        lines.append("")

        # Overall accuracy
        lines.append("## Overall Accuracy")
        lines.append(f"**{report.overall_accuracy:.2%}**")
        lines.append("")

        # Claims section
        lines.append("## Extracted Claims")
        if not report.claims:
            lines.append("_No claims extracted_")
        else:
            for claim in report.claims:
                lines.append(f"- **{claim.claim_id}**: {claim.text}")
                lines.append(f"  - Type: {claim.claim_type.value}")
                lines.append(f"  - Confidence: {claim.confidence:.2%}")
                if claim.source_span:
                    lines.append(f"  - Source span: {claim.source_span}")
        lines.append("")

        # Results section
        lines.append("## Verification Results")
        if not report.results:
            lines.append("_No verification results_")
        else:
            for result in report.results:
                lines.append(f"### {result.claim.claim_id}")
                lines.append(f"- **Status**: {result.status.value}")
                lines.append(f"- **Confidence**: {result.confidence:.2%}")
                lines.append(f"- **Reasoning**: {result.reasoning}")
                if result.evidence:
                    lines.append("- **Evidence**:")
                    for evidence in result.evidence:
                        lines.append(f"  - {evidence.content}")
                        lines.append(f"    - Source: {evidence.source.value}")
                        lines.append(f"    - Relevance: {evidence.relevance_score:.2%}")
                        if evidence.url:
                            lines.append(f"    - URL: {evidence.url}")
                lines.append("")

        # Flagged claims section
        if report.flagged_claims:
            lines.append("## Flagged Claims")
            for claim_id in report.flagged_claims:
                lines.append(f"- {claim_id}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self, report: VerificationReport) -> str:
        """Convert a verification report to JSON format.

        Args:
            report: The verification report to convert

        Returns:
            JSON string representation
        """
        # Use Pydantic's built-in serialization which handles enums properly
        return report.model_dump_json(indent=2)

    def generate_summary(self, report: VerificationReport) -> str:
        """Generate a human-readable summary of verification results.

        Args:
            report: The verification report to summarize

        Returns:
            Human-readable summary string
        """
        lines = []

        # Header
        lines.append(f"Verification Summary for {report.report_id}")
        lines.append("=" * 50)
        lines.append("")

        # Total claims extracted
        total_claims = len(report.claims)
        lines.append(f"Total claims extracted: {total_claims}")
        lines.append("")

        # Verification counts
        if report.results:
            verified_count = sum(1 for r in report.results if r.status.value == "verified")
            refuted_count = sum(1 for r in report.results if r.status.value == "refuted")
            uncertain_count = sum(1 for r in report.results if r.status.value == "uncertain")

            lines.append("Verification Status:")
            lines.append(f"  - Verified: {verified_count}")
            lines.append(f"  - Refuted: {refuted_count}")
            lines.append(f"  - Uncertain: {uncertain_count}")
            lines.append("")
        else:
            lines.append("No verification results available")
            lines.append("")

        # Overall accuracy
        lines.append(f"Overall accuracy: {report.overall_accuracy:.2%}")
        lines.append("")

        # Flagged claims
        if report.flagged_claims:
            lines.append("Flagged claims requiring attention:")
            for claim_id in report.flagged_claims:
                # Find the claim
                claim = next((c for c in report.claims if c.claim_id == claim_id), None)
                if claim:
                    lines.append(f"  - {claim_id}: {claim.text}")

                # Find the verification result for this claim
                result = next((r for r in report.results if r.claim.claim_id == claim_id), None)
                if result:
                    lines.append(f"    Reason: {result.reasoning}")
            lines.append("")
        else:
            lines.append("No claims flagged for review")
            lines.append("")

        return "\n".join(lines)
