"""Verification engine for orchestrating claim extraction and verification.

This module provides the VerificationEngine class that coordinates the full
verification pipeline: extracting claims from text, verifying each claim,
and aggregating results into a comprehensive report.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from reasoning_mcp.verification.checkers import (
    FactChecker,
    LogicalConsistencyChecker,
    NumericalChecker,
    SelfConsistencyChecker,
)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.verification.extractors import ClaimExtractor


class VerificationEngine:
    """Orchestrates the full verification pipeline.

    The VerificationEngine coordinates claim extraction, verification, and
    result aggregation. It uses a ClaimExtractor to identify claims in text,
    then selects appropriate FactCheckers to verify each claim, and finally
    aggregates results into a comprehensive VerificationReport.

    Examples:
        Create a verification engine with default checkers:
        >>> from reasoning_mcp.verification.extractors import RuleBasedExtractor
        >>> from reasoning_mcp.verification.checkers import SelfConsistencyChecker
        >>> extractor = RuleBasedExtractor()
        >>> checker = SelfConsistencyChecker()
        >>> engine = VerificationEngine(
        ...     extractor=extractor,
        ...     checkers=[checker]
        ... )

        Verify text:
        >>> report = engine.verify_text("The Earth is flat and 2+2=4")
        >>> assert len(report.claims) > 0
        >>> assert report.overall_accuracy >= 0.0
    """

    def __init__(
        self,
        extractor: ClaimExtractor,
        checkers: list[FactChecker],
        config: dict[str, Any] | None = None,
        ctx: ExecutionContext | None = None,
    ) -> None:
        """Initialize the verification engine.

        Args:
            extractor: ClaimExtractor instance for extracting claims from text
            checkers: List of FactChecker instances for verifying claims
            config: Optional configuration dictionary for engine behavior
            ctx: Optional execution context for pipeline integration
        """
        self.extractor = extractor
        self.checkers = checkers
        self.config = config or {}
        self.ctx = ctx

    async def verify_text(self, text: str) -> VerificationReport:
        """Run full verification pipeline on text.

        Extracts claims from the text, verifies each claim using appropriate
        checkers, calculates overall accuracy, and identifies flagged claims.

        Args:
            text: The text to extract claims from and verify

        Returns:
            VerificationReport with all extracted claims, verification results,
            overall accuracy, and flagged claims

        Examples:
            >>> from reasoning_mcp.verification.extractors import RuleBasedExtractor
            >>> from reasoning_mcp.verification.checkers import SelfConsistencyChecker
            >>> extractor = RuleBasedExtractor()
            >>> checker = SelfConsistencyChecker()
            >>> engine = VerificationEngine(extractor=extractor, checkers=[checker])
            >>> # report = await engine.verify_text("The sky is blue. 2+2=5.")
            >>> # assert len(report.claims) >= 0
            >>> # assert 0.0 <= report.overall_accuracy <= 1.0
        """
        # Extract claims from text - handle both sync and async extractors
        if hasattr(self.extractor.extract, "__self__"):
            # Check if it's async by trying to get the method
            import inspect

            if inspect.iscoroutinefunction(self.extractor.extract):
                if self.ctx is None:
                    raise ValueError("Async extractor requires ExecutionContext but ctx is None")
                # LLMClaimExtractor and HybridExtractor take ctx parameter
                claims = await self.extractor.extract(text, self.ctx)  # type: ignore[call-arg]
            else:
                claims = self.extractor.extract(text)
        else:
            # Fallback for protocol/duck-typed extractors
            claims = self.extractor.extract(text)

        # Verify each claim
        results: list[VerificationResult] = []
        for claim in claims:
            result = await self.verify_claim(claim)
            results.append(result)

        # Calculate overall accuracy
        overall_accuracy = self.calculate_accuracy(results)

        # Identify flagged claims (REFUTED or low confidence)
        flagged_claims: list[str] = []
        for result in results:
            if result.status == VerificationStatus.REFUTED or result.confidence < 0.5:
                flagged_claims.append(result.claim.claim_id)

        # Generate report
        report = VerificationReport(
            report_id=str(uuid.uuid4()),
            original_text=text,
            claims=claims,
            results=results,
            overall_accuracy=overall_accuracy,
            flagged_claims=flagged_claims,
        )

        return report

    async def verify_claim(self, claim: Claim) -> VerificationResult:
        """Verify a single claim using appropriate checker.

        Selects the best checker for the claim type and runs verification.

        Args:
            claim: The claim to verify

        Returns:
            VerificationResult with verification status, confidence, evidence,
            and reasoning

        Examples:
            >>> from reasoning_mcp.verification.checkers import SelfConsistencyChecker
            >>> from reasoning_mcp.verification.extractors import RuleBasedExtractor
            >>> extractor = RuleBasedExtractor()
            >>> checker = SelfConsistencyChecker()
            >>> engine = VerificationEngine(extractor=extractor, checkers=[checker])
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="The sky is blue",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.9
            ... )
            >>> # result = await engine.verify_claim(claim)
            >>> # assert result.claim == claim
            >>> # assert isinstance(result.status, VerificationStatus)
        """
        # Select appropriate checker
        checker = self._select_checker(claim)

        # Run verification - checkers use async check method
        if self.ctx is None:
            raise ValueError("ExecutionContext required for verification but ctx is None")

        result = await checker.check(claim, self.ctx)

        return result

    def _select_checker(self, claim: Claim) -> FactChecker:
        """Select the best checker for a claim based on claim type.

        Selection strategy:
        - NUMERICAL claims -> NumericalChecker if available
        - LOGICAL claims -> LogicalConsistencyChecker if available
        - Default -> First checker or SelfConsistencyChecker fallback
        - Falls back to first available checker if no match

        Args:
            claim: The claim to select a checker for

        Returns:
            The selected FactChecker instance

        Raises:
            ValueError: If no checkers are available

        Examples:
            >>> from reasoning_mcp.verification.checkers import (
            ...     NumericalChecker,
            ...     LogicalConsistencyChecker,
            ...     SelfConsistencyChecker
            ... )
            >>> from reasoning_mcp.verification.extractors import RuleBasedExtractor
            >>> extractor = RuleBasedExtractor()
            >>> numerical = NumericalChecker()
            >>> logical = LogicalConsistencyChecker()
            >>> self_consistency = SelfConsistencyChecker()
            >>> engine = VerificationEngine(
            ...     extractor=extractor,
            ...     checkers=[numerical, logical, self_consistency]
            ... )
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="2+2=4",
            ...     claim_type=ClaimType.NUMERICAL,
            ...     confidence=0.9
            ... )
            >>> checker = engine._select_checker(claim)
            >>> assert isinstance(checker, NumericalChecker)
        """
        if not self.checkers:
            raise ValueError("No checkers available for verification")

        # Try to find specialized checker for claim type
        if claim.claim_type == ClaimType.NUMERICAL:
            for checker in self.checkers:
                if isinstance(checker, NumericalChecker):
                    return checker

        if claim.claim_type == ClaimType.LOGICAL:
            for checker in self.checkers:
                if isinstance(checker, LogicalConsistencyChecker):
                    return checker

        # Look for SelfConsistencyChecker as fallback
        for checker in self.checkers:
            if isinstance(checker, SelfConsistencyChecker):
                return checker

        # Default to first available checker
        return self.checkers[0]

    def calculate_accuracy(self, results: list[VerificationResult]) -> float:
        """Calculate weighted accuracy from verification results.

        Accuracy calculation:
        - VERIFIED = 1.0
        - UNCERTAIN = 0.5
        - REFUTED/UNVERIFIABLE = 0.0
        - Weighted by confidence

        Args:
            results: List of verification results

        Returns:
            Overall accuracy score (0.0 to 1.0), or 0.0 for empty list

        Examples:
            >>> from reasoning_mcp.verification.extractors import RuleBasedExtractor
            >>> from reasoning_mcp.verification.checkers import SelfConsistencyChecker
            >>> extractor = RuleBasedExtractor()
            >>> checker = SelfConsistencyChecker()
            >>> engine = VerificationEngine(extractor=extractor, checkers=[checker])
            >>> claim1 = Claim(
            ...     claim_id="c1",
            ...     text="Test",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.9
            ... )
            >>> result1 = VerificationResult(
            ...     claim=claim1,
            ...     status=VerificationStatus.VERIFIED,
            ...     confidence=1.0,
            ...     evidence=[],
            ...     reasoning="Test"
            ... )
            >>> accuracy = engine.calculate_accuracy([result1])
            >>> assert accuracy == 1.0
        """
        if not results:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        for result in results:
            # Determine score based on status
            if result.status == VerificationStatus.VERIFIED:
                score = 1.0
            elif result.status == VerificationStatus.UNCERTAIN:
                score = 0.5
            else:  # REFUTED or UNVERIFIABLE
                score = 0.0

            # Weight by confidence
            weight = result.confidence
            total_weighted_score += score * weight
            total_weight += weight

        # Avoid division by zero
        if total_weight == 0.0:
            return 0.0

        return total_weighted_score / total_weight
