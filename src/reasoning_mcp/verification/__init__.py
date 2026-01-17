"""Verification and fact-checking module for reasoning-mcp.

This module provides comprehensive verification capabilities including:
- Claim extraction from reasoning text
- Fact checking with multiple strategies
- Hallucination detection
- Confidence calibration
- Verification reporting

Example:
    >>> from reasoning_mcp.verification import VerificationEngine
    >>> from reasoning_mcp.verification import RuleBasedExtractor
    >>> engine = VerificationEngine(extractor=RuleBasedExtractor())
    >>> report = await engine.verify_text("Some reasoning text...")
"""

from __future__ import annotations

# Calibration
from reasoning_mcp.verification.calibration import (
    CALIBRATION_DEFAULTS,
    ConfidenceCalibrator,
)

# Checkers
from reasoning_mcp.verification.checkers import (
    ExternalSourceChecker,
    FactChecker,
    LogicalConsistencyChecker,
    NumericalChecker,
    SelfConsistencyChecker,
)

# Engine
from reasoning_mcp.verification.engine import VerificationEngine

# Extractors
from reasoning_mcp.verification.extractors import (
    ClaimExtractor,
    HybridExtractor,
    LLMClaimExtractor,
    RuleBasedExtractor,
    get_extractor,
)

# Hallucination detection
from reasoning_mcp.verification.hallucination import (
    HallucinationDetector,
    get_severity_score,
)

# Reporting
from reasoning_mcp.verification.reporting import VerificationReporter

__all__ = [
    # Extractors
    "ClaimExtractor",
    "LLMClaimExtractor",
    "RuleBasedExtractor",
    "HybridExtractor",
    "get_extractor",
    # Checkers
    "FactChecker",
    "SelfConsistencyChecker",
    "LogicalConsistencyChecker",
    "NumericalChecker",
    "ExternalSourceChecker",
    # Engine
    "VerificationEngine",
    # Hallucination
    "HallucinationDetector",
    "get_severity_score",
    # Calibration
    "ConfidenceCalibrator",
    "CALIBRATION_DEFAULTS",
    # Reporting
    "VerificationReporter",
]
