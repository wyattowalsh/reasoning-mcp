"""Verification tools for text and claim validation.

This module provides functions for verifying text accuracy, validating claims,
and detecting hallucinations in reasoning outputs.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from pydantic import BaseModel, Field

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
)
from reasoning_mcp.verification import (
    HallucinationDetector,
    SelfConsistencyChecker,
    VerificationEngine,
    get_extractor,
)


class VerifyTextInput(BaseModel):
    """Input for verify_text tool.

    Examples:
        Create basic input:
        >>> input = VerifyTextInput(text="The Earth is flat and 2+2=4")
        >>> assert input.text == "The Earth is flat and 2+2=4"
        >>> assert input.context is None
        >>> assert input.check_hallucinations is True

        Create with context:
        >>> input = VerifyTextInput(
        ...     text="Paris is the capital of France",
        ...     context="France is a European country",
        ...     check_hallucinations=False
        ... )
        >>> assert input.context == "France is a European country"
    """

    text: str = Field(description="The text to verify")
    context: str | None = Field(default=None, description="Optional context for grounding")
    check_hallucinations: bool = Field(
        default=True, description="Whether to check for hallucinations"
    )


class VerifyClaimInput(BaseModel):
    """Input for verify_claim tool.

    Examples:
        Create basic claim input:
        >>> input = VerifyClaimInput(claim_text="2+2=4")
        >>> assert input.claim_text == "2+2=4"
        >>> assert input.claim_type is None

        Create with claim type:
        >>> input = VerifyClaimInput(
        ...     claim_text="The capital of France is Paris",
        ...     claim_type="factual"
        ... )
        >>> assert input.claim_type == "factual"
    """

    claim_text: str = Field(description="The claim text to verify")
    claim_type: str | None = Field(
        default=None,
        description="Type of claim: factual, numerical, temporal, causal, comparative, logical",
    )


async def verify_text(
    text: str,
    context: str | None = None,
    check_hallucinations: bool = True,
) -> dict[str, Any]:
    """Verify text for accuracy and hallucinations.

    This function analyzes text to extract and verify claims, optionally checking
    for hallucinations if a context is provided. It uses the VerificationEngine
    to orchestrate claim extraction and verification.

    Args:
        text: The text to verify
        context: Optional context for grounding verification
        check_hallucinations: Whether to check for hallucinations (default: True)

    Returns:
        Dictionary representation of VerificationReport containing:
            - report_id: Unique identifier for this report
            - original_text: The text that was analyzed
            - claims: List of extracted claims
            - results: Verification results for each claim
            - overall_accuracy: Overall accuracy score (0.0 to 1.0)
            - flagged_claims: List of claim IDs flagged for review
            - hallucination_flags: List of detected hallucinations (if check enabled)

    Examples:
        Basic verification:
        >>> report = await verify_text("The sky is blue and 2+2=5")
        >>> assert "report_id" in report
        >>> assert "claims" in report
        >>> assert "overall_accuracy" in report

        Verification with context:
        >>> report = await verify_text(
        ...     text="Paris is the largest city in France",
        ...     context="France is a European country with Paris as its capital"
        ... )

        Disable hallucination checking:
        >>> report = await verify_text(
        ...     text="Some factual statement",
        ...     check_hallucinations=False
        ... )
    """
    # Create VerificationEngine with default extractor and checkers
    extractor = get_extractor({"extractor_type": "hybrid"})
    checker = SelfConsistencyChecker()
    engine = VerificationEngine(
        extractor=extractor,
        checkers=[checker],
    )

    # Run verify_text on engine
    verification_report = await engine.verify_text(text)

    # Convert to dict - Pydantic doesn't auto-serialize enums to strings in model_dump()
    # We need to serialize properly for JSON compatibility
    report_dict: dict[str, Any] = json.loads(verification_report.model_dump_json())

    # Optionally run hallucination detection
    if check_hallucinations and context is not None:
        detector = HallucinationDetector()
        hallucination_flags = await detector.detect(text, context)
        report_dict["hallucination_flags"] = [
            json.loads(flag.model_dump_json()) for flag in hallucination_flags
        ]
    elif check_hallucinations:
        # Run without context
        detector = HallucinationDetector()
        hallucination_flags = await detector.detect(text, None)
        report_dict["hallucination_flags"] = [
            json.loads(flag.model_dump_json()) for flag in hallucination_flags
        ]
    else:
        report_dict["hallucination_flags"] = []

    return report_dict


async def verify_claim(
    claim_text: str,
    claim_type: str | None = None,
) -> dict[str, Any]:
    """Verify a single claim.

    This function verifies a single claim by creating a Claim object and using
    the VerificationEngine to check its validity. It returns a verification
    result with status, confidence, evidence, and reasoning.

    Args:
        claim_text: The claim text to verify
        claim_type: Type of claim (factual, numerical, temporal, causal,
            comparative, logical). If None, defaults to "factual"

    Returns:
        Dictionary representation of VerificationResult containing:
            - claim: The claim that was verified
            - status: Verification status (verified, refuted, uncertain, etc.)
            - confidence: Confidence in the verification (0.0 to 1.0)
            - evidence: List of supporting/refuting evidence
            - reasoning: Explanation of the verification decision

    Examples:
        Verify a factual claim:
        >>> result = await verify_claim("The Earth orbits the Sun")
        >>> assert "claim" in result
        >>> assert "status" in result
        >>> assert "confidence" in result

        Verify with claim type:
        >>> result = await verify_claim(
        ...     claim_text="2+2=4",
        ...     claim_type="numerical"
        ... )
        >>> assert result["claim"]["claim_type"] == "numerical"
    """
    # Create Claim object
    if claim_type is None:
        parsed_claim_type = ClaimType.FACTUAL
    else:
        # Map string to ClaimType enum
        type_map = {
            "factual": ClaimType.FACTUAL,
            "numerical": ClaimType.NUMERICAL,
            "temporal": ClaimType.TEMPORAL,
            "causal": ClaimType.CAUSAL,
            "comparative": ClaimType.COMPARATIVE,
            "logical": ClaimType.LOGICAL,
        }
        parsed_claim_type = type_map.get(claim_type.lower(), ClaimType.FACTUAL)

    claim = Claim(
        claim_id=str(uuid.uuid4()),
        text=claim_text,
        claim_type=parsed_claim_type,
        confidence=0.9,  # Default confidence for user-provided claims
    )

    # Use VerificationEngine to verify
    extractor = get_extractor({"extractor_type": "hybrid"})
    checker = SelfConsistencyChecker()
    engine = VerificationEngine(
        extractor=extractor,
        checkers=[checker],
    )

    result = await engine.verify_claim(claim)

    # Return VerificationResult as dict - use JSON serialization to properly handle enums
    return cast("dict[str, Any]", json.loads(result.model_dump_json()))


async def detect_hallucinations(
    text: str,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Detect potential hallucinations in text.

    This function analyzes text for potential hallucinations by checking for:
    - Factual claims not grounded in provided context
    - Self-contradictions within the text
    - Unsupported assertions made without evidence

    Args:
        text: The text to analyze for hallucinations
        context: Optional context to check factual grounding against

    Returns:
        List of dictionaries representing HallucinationFlag objects, each containing:
            - claim_id: ID of the claim flagged as potential hallucination
            - severity: Severity level (low, medium, high)
            - explanation: Why this claim is flagged
            - suggested_correction: Optional suggested correction

    Examples:
        Detect hallucinations without context:
        >>> flags = await detect_hallucinations("The capital of France is London")
        >>> assert isinstance(flags, list)

        Detect with context:
        >>> flags = await detect_hallucinations(
        ...     text="Paris has 10 million people",
        ...     context="Paris is the capital of France with 2.2 million residents"
        ... )
        >>> # Should flag the population claim

        Empty text:
        >>> flags = await detect_hallucinations("")
        >>> assert flags == []
    """
    # Use HallucinationDetector
    detector = HallucinationDetector()
    hallucination_flags = await detector.detect(text, context)

    # Return list of HallucinationFlag as dicts - use JSON serialization for proper enum handling
    return [json.loads(flag.model_dump_json()) for flag in hallucination_flags]
