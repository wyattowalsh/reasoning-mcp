"""Confidence calibration for ensemble members."""

from __future__ import annotations

from collections import defaultdict


class ConfidenceCalibrator:
    """Learns and applies confidence calibration for ensemble members.

    Tracks the relationship between predicted confidence and actual
    correctness to improve confidence estimates over time.

    This class uses historical performance data to calibrate confidence scores,
    adjusting for members that are consistently over-confident or under-confident.

    Attributes:
        _history: Stores (predicted, actual) pairs per ensemble member.
            The predicted value is the raw confidence (0.0-1.0).
            The actual value is 1.0 for correct predictions, 0.0 for incorrect.
        _calibration_factors: Cached calibration multipliers per member.
            These are recalculated when new calibration data is added.

    Example:
        Basic calibration workflow:

        >>> calibrator = ConfidenceCalibrator()

        # Learn from outcomes
        >>> calibrator.calibrate("cot", predicted=0.9, actual=1.0)  # Correct
        >>> calibrator.calibrate("cot", predicted=0.8, actual=0.0)  # Wrong
        >>> calibrator.calibrate("cot", predicted=0.85, actual=1.0)  # Correct

        # Get calibrated confidence
        >>> calibrated = calibrator.get_calibrated_confidence("cot", 0.85)
        >>> print(f"Calibrated confidence: {calibrated:.2f}")

        Advanced usage with multiple members:

        >>> calibrator = ConfidenceCalibrator()

        # CoT tends to be overconfident
        >>> for _ in range(10):
        ...     calibrator.calibrate("cot", predicted=0.9, actual=0.5)

        # React tends to be underconfident
        >>> for _ in range(10):
        ...     calibrator.calibrate("react", predicted=0.6, actual=1.0)

        # Apply calibration
        >>> cot_calibrated = calibrator.get_calibrated_confidence("cot", 0.9)
        >>> react_calibrated = calibrator.get_calibrated_confidence("react", 0.6)
        >>> print(f"CoT: {cot_calibrated:.2f}, React: {react_calibrated:.2f}")

        # Reset specific member
        >>> calibrator.reset("cot")

        # Or reset all
        >>> calibrator.reset()
    """

    def __init__(self) -> None:
        """Initialize the calibrator.

        Creates empty storage for calibration history and cached factors.
        No calibration data is present initially, so all confidence scores
        will pass through unchanged until calibrate() is called.
        """
        # Store (predicted, actual) pairs per member
        self._history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # Cached calibration factors per member
        self._calibration_factors: dict[str, float] = {}

    def calibrate(self, member: str, predicted: float, actual: float) -> None:
        """Record a calibration data point.

        Stores a prediction-outcome pair to improve future calibration.
        The calibration factor is recalculated lazily on next access.

        Args:
            member: Name of the ensemble member (e.g., "cot", "react", "tot").
                This identifies which reasoning method made the prediction.
            predicted: The predicted confidence (0.0-1.0). This is the raw
                confidence score produced by the member.
            actual: The actual outcome (0.0 wrong, 1.0 correct). Use 1.0
                if the prediction was correct, 0.0 if incorrect. Partial
                credit values between 0.0 and 1.0 are also supported.

        Example:
            Track multiple predictions for calibration:

            >>> calibrator = ConfidenceCalibrator()

            # Record correct predictions
            >>> calibrator.calibrate("cot", predicted=0.95, actual=1.0)
            >>> calibrator.calibrate("cot", predicted=0.88, actual=1.0)

            # Record incorrect predictions
            >>> calibrator.calibrate("cot", predicted=0.75, actual=0.0)

            # Now calibration will adjust based on this history
            >>> adjusted = calibrator.get_calibrated_confidence("cot", 0.9)
        """
        self._history[member].append((predicted, actual))
        # Invalidate cached factor to force recalculation
        self._calibration_factors.pop(member, None)

    def get_calibrated_confidence(self, member: str, raw_confidence: float) -> float:
        """Get a calibrated confidence score.

        Adjusts raw confidence based on historical accuracy. If the member
        has historically been overconfident, the calibration will reduce
        confidence scores. If underconfident, scores will be increased.

        The calibration uses a simple linear scaling factor computed as:
            factor = (average actual outcome) / (average predicted confidence)

        With no calibration data, returns the raw confidence unchanged.

        Args:
            member: Name of the ensemble member to calibrate for. This must
                match the member names used in calibrate() calls.
            raw_confidence: The raw confidence to calibrate (0.0-1.0). This
                is typically the confidence score produced by the member.

        Returns:
            Calibrated confidence between 0.0 and 1.0. The value is clamped
            to ensure it stays within valid probability bounds.

        Example:
            Calibrate confidence from an overconfident member:

            >>> calibrator = ConfidenceCalibrator()

            # Train on overconfident predictions (high confidence, low accuracy)
            >>> for _ in range(5):
            ...     calibrator.calibrate("overconfident", predicted=0.9, actual=0.5)

            # Raw confidence gets adjusted down
            >>> calibrated = calibrator.get_calibrated_confidence(
            ...     "overconfident", raw_confidence=0.9
            ... )
            >>> assert calibrated < 0.9  # Reduced due to historical overconfidence

            # Calibrate from an underconfident member:
            >>> for _ in range(5):
            ...     calibrator.calibrate("underconfident", predicted=0.5, actual=0.9)

            # Raw confidence gets adjusted up
            >>> calibrated = calibrator.get_calibrated_confidence(
            ...     "underconfident", raw_confidence=0.5
            ... )
            >>> assert calibrated > 0.5  # Increased due to historical underconfidence
        """
        # Calculate or use cached calibration factor
        factor = self._get_calibration_factor(member)

        # Apply calibration (simple linear adjustment)
        calibrated = raw_confidence * factor
        return max(0.0, min(1.0, calibrated))

    def _get_calibration_factor(self, member: str) -> float:
        """Calculate calibration factor for a member.

        Computes the ratio of actual outcomes to predicted confidence.
        This factor is cached until new calibration data is added.

        Args:
            member: Name of the ensemble member.

        Returns:
            Calibration factor (typically between 0.5 and 2.0).
            Returns 1.0 if no calibration data exists.

        Note:
            The calibration factor represents how much to scale confidence:
            - factor < 1.0: Member is overconfident, reduce scores
            - factor = 1.0: Member is well-calibrated, no adjustment
            - factor > 1.0: Member is underconfident, increase scores
        """
        if member in self._calibration_factors:
            return self._calibration_factors[member]

        history = self._history.get(member, [])
        if not history:
            return 1.0  # No calibration data, use raw

        # Simple calibration: ratio of actual accuracy to average predicted
        avg_predicted = sum(p for p, _ in history) / len(history)
        avg_actual = sum(a for _, a in history) / len(history)

        factor = 1.0 if avg_predicted == 0 else avg_actual / avg_predicted

        self._calibration_factors[member] = factor
        return factor

    def reset(self, member: str | None = None) -> None:
        """Reset calibration data.

        Clears calibration history and cached factors. Use this to start
        fresh calibration for a member or all members.

        Args:
            member: Name of the member to reset. If None, resets all members.

        Example:
            Reset specific member:

            >>> calibrator = ConfidenceCalibrator()
            >>> calibrator.calibrate("cot", predicted=0.9, actual=0.5)
            >>> calibrator.calibrate("react", predicted=0.8, actual=0.9)

            # Reset just CoT
            >>> calibrator.reset("cot")
            >>> cot_conf = calibrator.get_calibrated_confidence("cot", 0.9)
            >>> assert cot_conf == 0.9  # No calibration, returns raw

            # React still has calibration
            >>> react_conf = calibrator.get_calibrated_confidence("react", 0.8)
            >>> assert react_conf != 0.8  # Still calibrated

            Reset all members:

            >>> calibrator.reset()  # Clear everything
            >>> assert len(calibrator._history) == 0
        """
        if member:
            self._history.pop(member, None)
            self._calibration_factors.pop(member, None)
        else:
            self._history.clear()
            self._calibration_factors.clear()
