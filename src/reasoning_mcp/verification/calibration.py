"""Confidence calibration for verification system.

This module provides tools for calibrating confidence scores in verification
results to improve reliability and accuracy of predictions. It applies
claim-type-specific scaling factors and supports online learning from
verification outcomes.
"""

from __future__ import annotations

from reasoning_mcp.models.verification import ClaimType, VerificationResult, VerificationStatus

# Default calibration factors per claim type as (scale, offset)
CALIBRATION_DEFAULTS: dict[ClaimType, tuple[float, float]] = {
    ClaimType.FACTUAL: (0.85, 0.05),  # Slightly underconfident on facts
    ClaimType.NUMERICAL: (0.90, 0.03),  # Higher confidence on computations
    ClaimType.TEMPORAL: (0.80, 0.08),  # Temporal claims often overconfident
    ClaimType.CAUSAL: (0.75, 0.10),  # Causal claims most overconfident
    ClaimType.COMPARATIVE: (0.82, 0.07),  # Moderate calibration needed
    ClaimType.LOGICAL: (0.88, 0.04),  # Logical deductions fairly reliable
}


class ConfidenceCalibrator:
    """Calibrates confidence scores for verification results.

    Applies claim-type-specific scaling factors to raw confidence scores
    and tracks prediction history for online learning and calibration
    assessment.

    Examples:
        Basic calibration with defaults:
        >>> calibrator = ConfidenceCalibrator()
        >>> raw_confidence = 0.7
        >>> calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)
        >>> assert 0.0 <= calibrated <= 1.0

        Custom calibration factors:
        >>> custom_factors = {
        ...     ClaimType.FACTUAL: (0.9, 0.02),
        ...     ClaimType.NUMERICAL: (0.95, 0.01),
        ... }
        >>> calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)

        Track predictions for reliability assessment:
        >>> calibrator.update_calibration(predicted=0.8, actual=True)
        >>> calibrator.update_calibration(predicted=0.6, actual=False)
    """

    def __init__(
        self,
        calibration_factors: dict[ClaimType, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the confidence calibrator.

        Args:
            calibration_factors: Optional custom calibration factors per claim type.
                Each entry is a tuple of (scale, offset) where:
                - scale: Multiplicative factor applied to raw confidence
                - offset: Additive offset applied after scaling
                If None, uses CALIBRATION_DEFAULTS.

        Examples:
            Initialize with defaults:
            >>> calibrator = ConfidenceCalibrator()
            >>> assert calibrator.factors == CALIBRATION_DEFAULTS

            Initialize with custom factors:
            >>> custom = {ClaimType.FACTUAL: (0.9, 0.02)}
            >>> calibrator = ConfidenceCalibrator(calibration_factors=custom)
            >>> assert calibrator.factors[ClaimType.FACTUAL] == (0.9, 0.02)
        """
        self.factors = calibration_factors or dict(CALIBRATION_DEFAULTS)
        self.history: list[tuple[float, bool]] = []  # (predicted, actual)

    def calibrate(self, raw_confidence: float, claim_type: ClaimType) -> float:
        """Calibrate a raw confidence score based on claim type.

        Applies the formula: calibrated = raw * scale + offset
        Result is clamped to the valid range [0.0, 1.0].

        Args:
            raw_confidence: The uncalibrated confidence score (0.0 to 1.0)
            claim_type: The type of claim being calibrated

        Returns:
            The calibrated confidence score, clamped to [0.0, 1.0]

        Examples:
            Calibrate factual claim:
            >>> calibrator = ConfidenceCalibrator()
            >>> calibrated = calibrator.calibrate(0.8, ClaimType.FACTUAL)
            >>> # With defaults: 0.8 * 0.85 + 0.05 = 0.73
            >>> assert 0.72 <= calibrated <= 0.74

            Calibrate numerical claim:
            >>> calibrated = calibrator.calibrate(0.9, ClaimType.NUMERICAL)
            >>> # With defaults: 0.9 * 0.90 + 0.03 = 0.84
            >>> assert 0.83 <= calibrated <= 0.85

            Boundary cases are clamped:
            >>> calibrated = calibrator.calibrate(0.0, ClaimType.FACTUAL)
            >>> assert calibrated >= 0.0
            >>> calibrated = calibrator.calibrate(1.0, ClaimType.CAUSAL)
            >>> assert calibrated <= 1.0
        """
        # Get calibration factors for this claim type
        # If not in factors dict, use CALIBRATION_DEFAULTS
        scale, offset = self.factors.get(claim_type, CALIBRATION_DEFAULTS[claim_type])

        # Apply calibration formula
        calibrated = raw_confidence * scale + offset

        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, calibrated))

    def update_calibration(self, predicted: float, actual: bool) -> None:
        """Update calibration history with a prediction outcome.

        This enables online learning and recalibration based on empirical
        accuracy. The history can be used to assess and improve calibration
        over time.

        Args:
            predicted: The predicted confidence score (0.0 to 1.0)
            actual: The actual outcome (True = correct, False = incorrect)

        Examples:
            Track successful prediction:
            >>> calibrator = ConfidenceCalibrator()
            >>> calibrator.update_calibration(predicted=0.9, actual=True)
            >>> assert len(calibrator.history) == 1
            >>> assert calibrator.history[0] == (0.9, True)

            Track failed prediction:
            >>> calibrator.update_calibration(predicted=0.8, actual=False)
            >>> assert len(calibrator.history) == 2
            >>> assert calibrator.history[1] == (0.8, False)

            History accumulates:
            >>> calibrator.update_calibration(predicted=0.7, actual=True)
            >>> assert len(calibrator.history) == 3
        """
        self.history.append((predicted, actual))

    def get_reliability_score(self, results: list[VerificationResult]) -> float:
        """Calculate reliability score for verification results.

        Assesses how well-calibrated the verification system is by comparing
        predicted confidence levels to actual outcomes. Uses a metric similar
        to the Brier score to measure calibration quality.

        A score of 1.0 indicates perfect calibration (predictions match outcomes),
        while 0.0 indicates anti-calibration (predictions are inversely related
        to outcomes).

        Args:
            results: List of verification results to assess

        Returns:
            Reliability score in range [0.0, 1.0], where:
            - 1.0 = perfectly calibrated
            - 0.5 = no better than random
            - 0.0 = anti-calibrated

        Examples:
            Perfect calibration (high confidence + verified):
            >>> from reasoning_mcp.models.verification import (
            ...     Claim, VerificationStatus, VerificationResult
            ... )
            >>> calibrator = ConfidenceCalibrator()
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="Test",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.9
            ... )
            >>> result = VerificationResult(
            ...     claim=claim,
            ...     status=VerificationStatus.VERIFIED,
            ...     confidence=0.95,
            ...     evidence=[],
            ...     reasoning="test"
            ... )
            >>> score = calibrator.get_reliability_score([result])
            >>> assert score > 0.9  # High confidence + verified = high reliability

            Poor calibration (high confidence + refuted):
            >>> result_refuted = VerificationResult(
            ...     claim=claim,
            ...     status=VerificationStatus.REFUTED,
            ...     confidence=0.95,
            ...     evidence=[],
            ...     reasoning="test"
            ... )
            >>> score = calibrator.get_reliability_score([result_refuted])
            >>> assert score < 0.2  # High confidence + refuted = low reliability

            Empty results list:
            >>> score = calibrator.get_reliability_score([])
            >>> assert score == 1.0  # No data = assume perfect (neutral)
        """
        if not results:
            return 1.0  # No data, assume perfect calibration

        # Calculate Brier-like score
        total_error = 0.0
        for result in results:
            # Convert status to binary outcome (1.0 for verified, 0.0 for refuted)
            # UNCERTAIN and UNVERIFIABLE treated as 0.5 (neutral)
            if result.status == VerificationStatus.VERIFIED:
                actual = 1.0
            elif result.status == VerificationStatus.REFUTED:
                actual = 0.0
            elif (
                result.status == VerificationStatus.UNCERTAIN
                or result.status == VerificationStatus.UNVERIFIABLE
            ):
                actual = 0.5
            else:  # PENDING
                actual = 0.5

            # Calculate squared error between predicted confidence and actual outcome
            predicted = result.confidence
            error = (predicted - actual) ** 2
            total_error += error

        # Convert average Brier score to reliability score (1 - error)
        # Brier score ranges from 0 (perfect) to 1 (worst)
        avg_brier = total_error / len(results)
        reliability = 1.0 - avg_brier

        # Clamp to valid range
        return max(0.0, min(1.0, reliability))
