"""Unit tests for confidence calibration module."""

from __future__ import annotations

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from reasoning_mcp.verification.calibration import (
    CALIBRATION_DEFAULTS,
    ConfidenceCalibrator,
)


class TestCalibratorInit:
    """Test suite for ConfidenceCalibrator initialization."""

    def test_calibrator_init_with_defaults(self) -> None:
        """Test initialization with default calibration factors."""
        # Act
        calibrator = ConfidenceCalibrator()

        # Assert
        assert calibrator.factors == CALIBRATION_DEFAULTS
        assert calibrator.history == []

    def test_calibrator_init_with_custom_factors(self) -> None:
        """Test initialization with custom calibration factors."""
        # Arrange
        custom_factors = {
            ClaimType.FACTUAL: (0.9, 0.02),
            ClaimType.NUMERICAL: (0.95, 0.01),
        }

        # Act
        calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)

        # Assert
        assert calibrator.factors == custom_factors
        assert calibrator.factors[ClaimType.FACTUAL] == (0.9, 0.02)
        assert calibrator.factors[ClaimType.NUMERICAL] == (0.95, 0.01)
        assert calibrator.history == []

    def test_calibrator_init_with_none_uses_defaults(self) -> None:
        """Test that passing None for factors uses defaults."""
        # Act
        calibrator = ConfidenceCalibrator(calibration_factors=None)

        # Assert
        assert calibrator.factors == CALIBRATION_DEFAULTS


class TestCalibrate:
    """Test suite for calibrate method."""

    def test_calibrate_factual_claim(self) -> None:
        """Test calibration for factual claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.8
        # Expected: 0.8 * 0.85 + 0.05 = 0.73

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert 0.72 <= calibrated <= 0.74
        assert isinstance(calibrated, float)

    def test_calibrate_numerical_claim(self) -> None:
        """Test calibration for numerical claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.9
        # Expected: 0.9 * 0.90 + 0.03 = 0.84

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.NUMERICAL)

        # Assert
        assert 0.83 <= calibrated <= 0.85

    def test_calibrate_temporal_claim(self) -> None:
        """Test calibration for temporal claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.7
        # Expected: 0.7 * 0.80 + 0.08 = 0.64

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.TEMPORAL)

        # Assert
        assert 0.63 <= calibrated <= 0.65

    def test_calibrate_causal_claim(self) -> None:
        """Test calibration for causal claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.8
        # Expected: 0.8 * 0.75 + 0.10 = 0.70

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.CAUSAL)

        # Assert
        assert 0.69 <= calibrated <= 0.71

    def test_calibrate_comparative_claim(self) -> None:
        """Test calibration for comparative claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.75
        # Expected: 0.75 * 0.82 + 0.07 = 0.685

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.COMPARATIVE)

        # Assert
        assert 0.67 <= calibrated <= 0.70

    def test_calibrate_logical_claim(self) -> None:
        """Test calibration for logical claim type."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.85
        # Expected: 0.85 * 0.88 + 0.04 = 0.788

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.LOGICAL)

        # Assert
        assert 0.77 <= calibrated <= 0.80

    def test_calibrate_with_custom_factors(self) -> None:
        """Test calibration with custom factors."""
        # Arrange
        custom_factors = {ClaimType.FACTUAL: (0.9, 0.02)}
        calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)
        raw_confidence = 0.8
        # Expected: 0.8 * 0.9 + 0.02 = 0.74

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert 0.73 <= calibrated <= 0.75

    def test_calibrate_missing_type_uses_defaults(self) -> None:
        """Test that missing claim type in custom factors falls back to defaults."""
        # Arrange
        custom_factors = {ClaimType.FACTUAL: (0.9, 0.02)}
        calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)
        raw_confidence = 0.9
        # Should use CALIBRATION_DEFAULTS for NUMERICAL: 0.9 * 0.90 + 0.03 = 0.84

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.NUMERICAL)

        # Assert
        assert 0.83 <= calibrated <= 0.85


class TestCalibrateBoundaryValues:
    """Test suite for calibrate method edge cases."""

    def test_calibrate_zero_confidence(self) -> None:
        """Test calibration with zero raw confidence."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 0.0

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert calibrated >= 0.0
        assert calibrated <= 1.0
        # With FACTUAL defaults: 0.0 * 0.85 + 0.05 = 0.05
        assert 0.04 <= calibrated <= 0.06

    def test_calibrate_max_confidence(self) -> None:
        """Test calibration with maximum raw confidence."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        raw_confidence = 1.0

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert calibrated >= 0.0
        assert calibrated <= 1.0
        # With FACTUAL defaults: 1.0 * 0.85 + 0.05 = 0.90
        assert 0.89 <= calibrated <= 0.91

    def test_calibrate_clamps_upper_bound(self) -> None:
        """Test that calibrated confidence is clamped to 1.0."""
        # Arrange
        # Use factors that would exceed 1.0
        custom_factors = {ClaimType.FACTUAL: (1.0, 0.2)}
        calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)
        raw_confidence = 0.9
        # Without clamping: 0.9 * 1.0 + 0.2 = 1.1

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert calibrated == 1.0

    def test_calibrate_clamps_lower_bound(self) -> None:
        """Test that calibrated confidence is clamped to 0.0."""
        # Arrange
        # Use factors that would go below 0.0
        custom_factors = {ClaimType.FACTUAL: (0.5, -0.3)}
        calibrator = ConfidenceCalibrator(calibration_factors=custom_factors)
        raw_confidence = 0.1
        # Without clamping: 0.1 * 0.5 - 0.3 = -0.25

        # Act
        calibrated = calibrator.calibrate(raw_confidence, ClaimType.FACTUAL)

        # Assert
        assert calibrated == 0.0

    def test_calibrate_mid_range_values(self) -> None:
        """Test calibration with various mid-range values."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        test_values = [0.3, 0.5, 0.7]

        # Act & Assert
        for raw_conf in test_values:
            calibrated = calibrator.calibrate(raw_conf, ClaimType.FACTUAL)
            assert 0.0 <= calibrated <= 1.0


class TestUpdateCalibration:
    """Test suite for update_calibration method."""

    def test_update_calibration_adds_to_history(self) -> None:
        """Test that update_calibration adds entries to history."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        calibrator.update_calibration(predicted=0.9, actual=True)

        # Assert
        assert len(calibrator.history) == 1
        assert calibrator.history[0] == (0.9, True)

    def test_update_calibration_tracks_correct_prediction(self) -> None:
        """Test tracking a correct prediction."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        calibrator.update_calibration(predicted=0.85, actual=True)

        # Assert
        assert calibrator.history[-1] == (0.85, True)

    def test_update_calibration_tracks_incorrect_prediction(self) -> None:
        """Test tracking an incorrect prediction."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        calibrator.update_calibration(predicted=0.75, actual=False)

        # Assert
        assert calibrator.history[-1] == (0.75, False)

    def test_update_calibration_accumulates(self) -> None:
        """Test that history accumulates over multiple updates."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        calibrator.update_calibration(predicted=0.9, actual=True)
        calibrator.update_calibration(predicted=0.8, actual=False)
        calibrator.update_calibration(predicted=0.7, actual=True)

        # Assert
        assert len(calibrator.history) == 3
        assert calibrator.history[0] == (0.9, True)
        assert calibrator.history[1] == (0.8, False)
        assert calibrator.history[2] == (0.7, True)

    def test_update_calibration_with_extreme_values(self) -> None:
        """Test update with extreme confidence values."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        calibrator.update_calibration(predicted=0.0, actual=False)
        calibrator.update_calibration(predicted=1.0, actual=True)

        # Assert
        assert len(calibrator.history) == 2
        assert calibrator.history[0] == (0.0, False)
        assert calibrator.history[1] == (1.0, True)


class TestCalibrationDefaults:
    """Test suite for CALIBRATION_DEFAULTS constant."""

    def test_calibration_defaults_has_all_claim_types(self) -> None:
        """Test that all claim types have default calibration factors."""
        # Arrange
        all_claim_types = [
            ClaimType.FACTUAL,
            ClaimType.NUMERICAL,
            ClaimType.TEMPORAL,
            ClaimType.CAUSAL,
            ClaimType.COMPARATIVE,
            ClaimType.LOGICAL,
        ]

        # Act & Assert
        for claim_type in all_claim_types:
            assert claim_type in CALIBRATION_DEFAULTS
            scale, offset = CALIBRATION_DEFAULTS[claim_type]
            assert isinstance(scale, float)
            assert isinstance(offset, float)

    def test_calibration_defaults_values_in_range(self) -> None:
        """Test that default calibration values are reasonable."""
        # Act & Assert
        for claim_type, (scale, offset) in CALIBRATION_DEFAULTS.items():
            # Scale should be between 0.5 and 1.0 (conservative)
            assert 0.5 <= scale <= 1.0, f"{claim_type} scale out of range"
            # Offset should be small positive value
            assert 0.0 <= offset <= 0.2, f"{claim_type} offset out of range"

    def test_calibration_defaults_immutable(self) -> None:
        """Test that CALIBRATION_DEFAULTS is properly defined."""
        # Arrange & Act
        defaults_copy = dict(CALIBRATION_DEFAULTS)

        # Assert
        assert len(defaults_copy) == 6
        assert defaults_copy == CALIBRATION_DEFAULTS


class TestReliabilityScore:
    """Test suite for get_reliability_score method."""

    def test_reliability_score_perfect_calibration(self) -> None:
        """Test reliability score with perfect calibration."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        # High confidence + verified = good calibration
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Verified",
        )

        # Act
        score = calibrator.get_reliability_score([result])

        # Assert
        # With confidence 0.95 and actual=1.0: error = (0.95-1.0)^2 = 0.0025
        # Reliability = 1.0 - 0.0025 = 0.9975
        assert score > 0.99

    def test_reliability_score_poor_calibration(self) -> None:
        """Test reliability score with poor calibration."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        # High confidence + refuted = poor calibration
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.REFUTED,
            confidence=0.95,
            evidence=[],
            reasoning="Refuted",
        )

        # Act
        score = calibrator.get_reliability_score([result])

        # Assert
        # With confidence 0.95 and actual=0.0: error = (0.95-0.0)^2 = 0.9025
        # Reliability = 1.0 - 0.9025 = 0.0975
        assert score < 0.15

    def test_reliability_score_empty_results(self) -> None:
        """Test reliability score with no results."""
        # Arrange
        calibrator = ConfidenceCalibrator()

        # Act
        score = calibrator.get_reliability_score([])

        # Assert
        assert score == 1.0  # Assume perfect with no data

    def test_reliability_score_multiple_results(self) -> None:
        """Test reliability score with multiple verification results."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim1 = Claim(
            claim_id="c1",
            text="Claim 1",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        claim2 = Claim(
            claim_id="c2",
            text="Claim 2",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.8,
        )

        # Mix of good and poor calibration
        result1 = VerificationResult(
            claim=claim1,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Verified",
        )
        result2 = VerificationResult(
            claim=claim2,
            status=VerificationStatus.REFUTED,
            confidence=0.2,
            evidence=[],
            reasoning="Refuted",
        )

        # Act
        score = calibrator.get_reliability_score([result1, result2])

        # Assert
        # result1: error = (0.95-1.0)^2 = 0.0025
        # result2: error = (0.2-0.0)^2 = 0.04
        # avg_error = (0.0025 + 0.04) / 2 = 0.02125
        # reliability = 1.0 - 0.02125 = 0.97875
        assert 0.9 <= score <= 1.0

    def test_reliability_score_uncertain_status(self) -> None:
        """Test reliability score with uncertain verification status."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            evidence=[],
            reasoning="Uncertain",
        )

        # Act
        score = calibrator.get_reliability_score([result])

        # Assert
        # With confidence 0.5 and actual=0.5 (uncertain): error = 0
        # Reliability = 1.0 - 0.0 = 1.0
        assert score == 1.0

    def test_reliability_score_unverifiable_status(self) -> None:
        """Test reliability score with unverifiable status."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.5,
            evidence=[],
            reasoning="Cannot verify",
        )

        # Act
        score = calibrator.get_reliability_score([result])

        # Assert
        assert score == 1.0  # Perfect match with neutral outcome

    def test_reliability_score_pending_status(self) -> None:
        """Test reliability score with pending status."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.PENDING,
            confidence=0.5,
            evidence=[],
            reasoning="Pending verification",
        )

        # Act
        score = calibrator.get_reliability_score([result])

        # Assert
        assert score == 1.0  # Perfect match with neutral outcome

    def test_reliability_score_range_bounds(self) -> None:
        """Test that reliability score is always in valid range."""
        # Arrange
        calibrator = ConfidenceCalibrator()
        claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )

        # Test various combinations
        statuses = [
            VerificationStatus.VERIFIED,
            VerificationStatus.REFUTED,
            VerificationStatus.UNCERTAIN,
        ]
        confidences = [0.0, 0.5, 1.0]

        # Act & Assert
        for status in statuses:
            for conf in confidences:
                result = VerificationResult(
                    claim=claim,
                    status=status,
                    confidence=conf,
                    evidence=[],
                    reasoning="Test",
                )
                score = calibrator.get_reliability_score([result])
                assert 0.0 <= score <= 1.0
