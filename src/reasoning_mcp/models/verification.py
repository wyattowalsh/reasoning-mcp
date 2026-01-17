"""Verification models for reasoning-mcp.

This module defines models related to verification of reasoning steps,
including verification status and related data structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(Enum):
    """Status of a verification attempt for a reasoning step or conclusion.

    This enum represents the possible outcomes when verifying a piece of
    reasoning, answer, or intermediate step.
    """

    VERIFIED = "verified"
    """The reasoning or conclusion has been successfully verified as correct."""

    REFUTED = "refuted"
    """The reasoning or conclusion has been refuted or proven incorrect."""

    UNCERTAIN = "uncertain"
    """The verification result is uncertain or inconclusive."""

    UNVERIFIABLE = "unverifiable"
    """The reasoning or conclusion cannot be verified with available information."""

    PENDING = "pending"
    """Verification has not yet been attempted or is in progress."""


class ClaimType(Enum):
    """Types of claims that can be verified during reasoning.

    This enum categorizes different types of claims that reasoning methods
    may make, allowing for specialized verification strategies for each type.

    Attributes:
        FACTUAL: Claims about objective facts that can be verified against
            external knowledge sources or databases.
        NUMERICAL: Claims involving numerical calculations, quantities,
            or mathematical results that can be computed or checked.
        TEMPORAL: Claims about time, sequence, duration, or chronological
            relationships between events.
        CAUSAL: Claims about cause-and-effect relationships, where one
            event or condition leads to another outcome.
        COMPARATIVE: Claims that compare two or more entities, values,
            or concepts (e.g., greater than, similar to, better than).
        LOGICAL: Claims based on logical deduction, inference, or
            reasoning from premises to conclusions.
    """

    FACTUAL = "factual"
    """Claims about objective facts that can be verified against external knowledge."""

    NUMERICAL = "numerical"
    """Claims involving numerical calculations, quantities, or mathematical results."""

    TEMPORAL = "temporal"
    """Claims about time, sequence, duration, or chronological relationships."""

    CAUSAL = "causal"
    """Claims about cause-and-effect relationships between events or conditions."""

    COMPARATIVE = "comparative"
    """Claims that compare two or more entities, values, or concepts."""

    LOGICAL = "logical"
    """Claims based on logical deduction, inference, or reasoning from premises."""


class Claim(BaseModel):
    """A claim extracted from reasoning output for verification.

    Represents a single verifiable claim that has been extracted from
    reasoning text, with metadata about its location, type, and confidence.

    Examples:
        Create a factual claim:
        >>> claim = Claim(
        ...     claim_id="claim_001",
        ...     text="The Earth orbits the Sun",
        ...     claim_type=ClaimType.FACTUAL,
        ...     source_span=(0, 24),
        ...     confidence=0.95
        ... )
        >>> assert claim.confidence >= 0.0 and claim.confidence <= 1.0

        Create a claim without source span:
        >>> claim = Claim(
        ...     claim_id="claim_002",
        ...     text="If X then Y",
        ...     claim_type=ClaimType.LOGICAL,
        ...     source_span=None,
        ...     confidence=0.8
        ... )
    """

    claim_id: str = Field(
        description="Unique identifier for this claim",
    )
    text: str = Field(
        description="The text content of the claim",
    )
    claim_type: ClaimType = Field(
        description="The type/category of this claim",
    )
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Character span (start, end) in the source text where this claim appears",
    )
    confidence: float = Field(
        description="Confidence score for claim extraction (0.0 to 1.0)",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Validate that confidence is in the valid range [0.0, 1.0].

        Args:
            v: The confidence value to validate

        Returns:
            The validated confidence value

        Raises:
            ValueError: If confidence is not in the range [0.0, 1.0]

        Examples:
            >>> # Valid confidence values
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="test",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.5
            ... )
            >>> claim.confidence
            0.5

            >>> # Invalid confidence value raises error
            >>> try:
            ...     Claim(
            ...         claim_id="c2",
            ...         text="test",
            ...         claim_type=ClaimType.FACTUAL,
            ...         confidence=1.5
            ...     )
            ... except ValueError as e:
            ...     print("Error:", str(e))
            Error: Confidence must be between 0.0 and 1.0, got 1.5
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class EvidenceSource(Enum):
    """Source type for evidence supporting or refuting a claim.

    This enum categorizes the different sources from which evidence
    can be obtained during the verification process.

    Attributes:
        INTERNAL: Evidence derived from the model's internal reasoning
            or knowledge, without external references.
        EXTERNAL: Evidence obtained from external sources such as
            databases, APIs, or knowledge bases.
        COMPUTED: Evidence generated through computation, calculation,
            or algorithmic verification.
        USER_PROVIDED: Evidence explicitly provided by the user as
            input or feedback.
    """

    INTERNAL = "internal"
    """Evidence from internal model reasoning or knowledge."""

    EXTERNAL = "external"
    """Evidence from external sources like databases or APIs."""

    COMPUTED = "computed"
    """Evidence from computation, calculation, or algorithmic verification."""

    USER_PROVIDED = "user_provided"
    """Evidence explicitly provided by the user."""


class Evidence(BaseModel):
    """Evidence supporting or refuting a claim during verification.

    Represents a piece of evidence that can be used to verify or
    refute a claim, with metadata about its source, relevance, and
    optional reference URL.

    Examples:
        Create evidence from an external source:
        >>> evidence = Evidence(
        ...     evidence_id="ev_001",
        ...     source=EvidenceSource.EXTERNAL,
        ...     content="According to NASA, Earth completes one orbit in 365.25 days",
        ...     relevance_score=0.95,
        ...     url="https://nasa.gov/earth"
        ... )
        >>> assert evidence.relevance_score >= 0.0 and evidence.relevance_score <= 1.0

        Create computed evidence without URL:
        >>> evidence = Evidence(
        ...     evidence_id="ev_002",
        ...     source=EvidenceSource.COMPUTED,
        ...     content="Calculated result: 2 + 2 = 4",
        ...     relevance_score=1.0,
        ...     url=None
        ... )
    """

    evidence_id: str = Field(
        description="Unique identifier for this evidence",
    )
    source: EvidenceSource = Field(
        description="The source type of this evidence",
    )
    content: str = Field(
        description="The text content or description of the evidence",
    )
    relevance_score: float = Field(
        description="How relevant this evidence is to the claim (0.0 to 1.0)",
    )
    url: str | None = Field(
        default=None,
        description="Optional URL reference for the evidence source",
    )

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance_score_range(cls, v: float) -> float:
        """Validate that relevance_score is in the valid range [0.0, 1.0].

        Args:
            v: The relevance score value to validate

        Returns:
            The validated relevance score value

        Raises:
            ValueError: If relevance_score is not in the range [0.0, 1.0]

        Examples:
            >>> # Valid relevance score
            >>> evidence = Evidence(
            ...     evidence_id="e1",
            ...     source=EvidenceSource.INTERNAL,
            ...     content="test",
            ...     relevance_score=0.75
            ... )
            >>> evidence.relevance_score
            0.75

            >>> # Invalid relevance score raises error
            >>> try:
            ...     Evidence(
            ...         evidence_id="e2",
            ...         source=EvidenceSource.INTERNAL,
            ...         content="test",
            ...         relevance_score=1.2
            ...     )
            ... except ValueError as e:
            ...     print("Error:", str(e))
            Error: Relevance score must be between 0.0 and 1.0, got 1.2
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Relevance score must be between 0.0 and 1.0, got {v}")
        return v


class VerificationResult(BaseModel):
    """Result of verifying a single claim with supporting evidence.

    Represents the outcome of verifying a claim, including the
    verification status, confidence level, supporting evidence,
    and reasoning behind the verification decision.

    Examples:
        Create a verification result for a verified claim:
        >>> claim = Claim(
        ...     claim_id="c1",
        ...     text="Earth orbits the Sun",
        ...     claim_type=ClaimType.FACTUAL,
        ...     confidence=0.9
        ... )
        >>> evidence = Evidence(
        ...     evidence_id="e1",
        ...     source=EvidenceSource.EXTERNAL,
        ...     content="NASA confirms orbital period",
        ...     relevance_score=0.95
        ... )
        >>> result = VerificationResult(
        ...     claim=claim,
        ...     status=VerificationStatus.VERIFIED,
        ...     confidence=0.95,
        ...     evidence=[evidence],
        ...     reasoning="Multiple authoritative sources confirm this fact"
        ... )
        >>> assert result.status == VerificationStatus.VERIFIED
    """

    claim: Claim = Field(
        description="The claim being verified",
    )
    status: VerificationStatus = Field(
        description="The verification status of the claim",
    )
    confidence: float = Field(
        description="Confidence in the verification result (0.0 to 1.0)",
    )
    evidence: list[Evidence] = Field(
        description="List of evidence supporting or refuting the claim",
    )
    reasoning: str = Field(
        description="Explanation of how the verification decision was reached",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Validate that confidence is in the valid range [0.0, 1.0].

        Args:
            v: The confidence value to validate

        Returns:
            The validated confidence value

        Raises:
            ValueError: If confidence is not in the range [0.0, 1.0]

        Examples:
            >>> # Valid confidence value (with minimal setup)
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="test",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.5
            ... )
            >>> result = VerificationResult(
            ...     claim=claim,
            ...     status=VerificationStatus.VERIFIED,
            ...     confidence=0.8,
            ...     evidence=[],
            ...     reasoning="test"
            ... )
            >>> result.confidence
            0.8

            >>> # Invalid confidence raises error
            >>> try:
            ...     VerificationResult(
            ...         claim=claim,
            ...         status=VerificationStatus.VERIFIED,
            ...         confidence=1.5,
            ...         evidence=[],
            ...         reasoning="test"
            ...     )
            ... except ValueError as e:
            ...     print("Error:", str(e))
            Error: Confidence must be between 0.0 and 1.0, got 1.5
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class VerificationReport(BaseModel):
    """Comprehensive report of verification results for multiple claims.

    Represents a complete verification analysis including all extracted
    claims, their verification results, overall accuracy metrics, and
    flagged claims requiring attention.

    Examples:
        Create a verification report:
        >>> claim1 = Claim(
        ...     claim_id="c1",
        ...     text="Test claim 1",
        ...     claim_type=ClaimType.FACTUAL,
        ...     confidence=0.9
        ... )
        >>> claim2 = Claim(
        ...     claim_id="c2",
        ...     text="Test claim 2",
        ...     claim_type=ClaimType.LOGICAL,
        ...     confidence=0.7
        ... )
        >>> result1 = VerificationResult(
        ...     claim=claim1,
        ...     status=VerificationStatus.VERIFIED,
        ...     confidence=0.95,
        ...     evidence=[],
        ...     reasoning="Verified through external sources"
        ... )
        >>> report = VerificationReport(
        ...     report_id="report_001",
        ...     original_text="Original reasoning text",
        ...     claims=[claim1, claim2],
        ...     results=[result1],
        ...     overall_accuracy=0.85,
        ...     flagged_claims=["c2"]
        ... )
        >>> assert len(report.claims) == 2
        >>> assert len(report.flagged_claims) == 1
    """

    report_id: str = Field(
        description="Unique identifier for this verification report",
    )
    original_text: str = Field(
        description="The original text that was analyzed for claims",
    )
    claims: list[Claim] = Field(
        description="All claims extracted from the original text",
    )
    results: list[VerificationResult] = Field(
        description="Verification results for each claim",
    )
    overall_accuracy: float = Field(
        description="Overall accuracy score of the verified claims (0.0 to 1.0)",
    )
    flagged_claims: list[str] = Field(
        description="List of claim IDs that have been flagged for review",
    )

    @field_validator("overall_accuracy")
    @classmethod
    def validate_overall_accuracy_range(cls, v: float) -> float:
        """Validate that overall_accuracy is in the valid range [0.0, 1.0].

        Args:
            v: The overall accuracy value to validate

        Returns:
            The validated overall accuracy value

        Raises:
            ValueError: If overall_accuracy is not in the range [0.0, 1.0]

        Examples:
            >>> # Valid overall accuracy
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="test",
            ...     claim_type=ClaimType.FACTUAL,
            ...     confidence=0.5
            ... )
            >>> report = VerificationReport(
            ...     report_id="r1",
            ...     original_text="test",
            ...     claims=[claim],
            ...     results=[],
            ...     overall_accuracy=0.9,
            ...     flagged_claims=[]
            ... )
            >>> report.overall_accuracy
            0.9

            >>> # Invalid overall accuracy raises error
            >>> try:
            ...     VerificationReport(
            ...         report_id="r2",
            ...         original_text="test",
            ...         claims=[claim],
            ...         results=[],
            ...         overall_accuracy=1.1,
            ...         flagged_claims=[]
            ...     )
            ... except ValueError as e:
            ...     print("Error:", str(e))
            Error: Overall accuracy must be between 0.0 and 1.0, got 1.1
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Overall accuracy must be between 0.0 and 1.0, got {v}")
        return v


class HallucinationFlag(BaseModel):
    """Flag indicating a potential hallucination in a claim.

    Represents a flagged claim that may contain hallucinated or
    incorrect information, with severity level, explanation, and
    optional suggested correction.

    Examples:
        Create a high-severity hallucination flag:
        >>> flag = HallucinationFlag(
        ...     claim_id="c1",
        ...     severity="high",
        ...     explanation="Claim contradicts established facts",
        ...     suggested_correction="The correct fact is X, not Y"
        ... )
        >>> assert flag.severity == "high"

        Create a flag without suggested correction:
        >>> flag = HallucinationFlag(
        ...     claim_id="c2",
        ...     severity="low",
        ...     explanation="Minor inconsistency detected",
        ...     suggested_correction=None
        ... )
        >>> assert flag.suggested_correction is None
    """

    claim_id: str = Field(
        description="ID of the claim being flagged as potential hallucination",
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity level of the potential hallucination",
    )
    explanation: str = Field(
        description="Explanation of why this claim is flagged as potentially hallucinated",
    )
    suggested_correction: str | None = Field(
        default=None,
        description="Optional suggested correction for the hallucinated claim",
    )
