"""
Tests for verification models in reasoning_mcp.models.verification.

This module provides test coverage for verification-related models:
- VerificationStatus enum (5 values)

Each enum is tested for:
1. Expected value existence
2. Enum type
3. Value count
4. Value uniqueness
5. String representation
6. Membership checks
"""

from enum import Enum

import pytest
from pydantic import ValidationError

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
    HallucinationFlag,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)

# ============================================================================
# VerificationStatus Tests
# ============================================================================


class TestVerificationStatus:
    """Test suite for VerificationStatus enum (5 values)."""

    EXPECTED_STATUSES = {
        "VERIFIED",
        "REFUTED",
        "UNCERTAIN",
        "UNVERIFIABLE",
        "PENDING",
    }

    EXPECTED_COUNT = 5

    def test_is_enum(self) -> None:
        """Test that VerificationStatus is an Enum."""
        assert issubclass(VerificationStatus, Enum)

    def test_all_expected_values_exist(self) -> None:
        """Test that all 5 expected statuses exist."""
        actual_names = {member.name for member in VerificationStatus}
        assert actual_names == self.EXPECTED_STATUSES

    def test_value_count(self) -> None:
        """Test that exactly 5 statuses are defined."""
        assert len(VerificationStatus) == self.EXPECTED_COUNT

    def test_values_are_strings(self) -> None:
        """Test that all enum values are strings."""
        for member in VerificationStatus:
            assert isinstance(member.value, str)

    def test_values_are_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [member.value for member in VerificationStatus]
        assert len(values) == len(set(values))

    def test_string_representation(self) -> None:
        """Test string representation of enum members."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.REFUTED.value == "refuted"
        assert VerificationStatus.UNCERTAIN.value == "uncertain"
        assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"
        assert VerificationStatus.PENDING.value == "pending"

    def test_membership_checks(self) -> None:
        """Test membership checks work correctly."""
        # Valid values
        assert VerificationStatus.VERIFIED in VerificationStatus
        assert VerificationStatus.REFUTED in VerificationStatus
        assert VerificationStatus.UNCERTAIN in VerificationStatus
        assert VerificationStatus.UNVERIFIABLE in VerificationStatus
        assert VerificationStatus.PENDING in VerificationStatus

    def test_access_by_name(self) -> None:
        """Test that enum members can be accessed by name."""
        assert VerificationStatus["VERIFIED"] == VerificationStatus.VERIFIED
        assert VerificationStatus["REFUTED"] == VerificationStatus.REFUTED
        assert VerificationStatus["UNCERTAIN"] == VerificationStatus.UNCERTAIN
        assert VerificationStatus["UNVERIFIABLE"] == VerificationStatus.UNVERIFIABLE
        assert VerificationStatus["PENDING"] == VerificationStatus.PENDING

    def test_access_by_value(self) -> None:
        """Test that enum members can be accessed by value."""
        assert VerificationStatus("verified") == VerificationStatus.VERIFIED
        assert VerificationStatus("refuted") == VerificationStatus.REFUTED
        assert VerificationStatus("uncertain") == VerificationStatus.UNCERTAIN
        assert VerificationStatus("unverifiable") == VerificationStatus.UNVERIFIABLE
        assert VerificationStatus("pending") == VerificationStatus.PENDING

    def test_comparison(self) -> None:
        """Test that enum members can be compared."""
        verified = VerificationStatus.VERIFIED
        refuted = VerificationStatus.REFUTED

        # Same member comparison
        assert verified == VerificationStatus.VERIFIED
        assert refuted == VerificationStatus.REFUTED

        # Different member comparison
        assert verified != refuted
        assert refuted != VerificationStatus.PENDING

    def test_iteration(self) -> None:
        """Test that we can iterate over all enum members."""
        members = list(VerificationStatus)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, VerificationStatus) for m in members)

    def test_status_values_lowercase(self) -> None:
        """Test that all status values are lowercase strings."""
        for member in VerificationStatus:
            assert member.value.islower()
            assert " " not in member.value  # No spaces
            assert member.value.isalpha()  # Only letters

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            VerificationStatus("invalid")

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid names raise KeyError."""
        with pytest.raises(KeyError):
            VerificationStatus["INVALID"]


# ============================================================================
# ClaimType Tests
# ============================================================================


class TestClaimType:
    """Test suite for ClaimType enum (6 values)."""

    EXPECTED_CLAIM_TYPES = {
        "FACTUAL",
        "NUMERICAL",
        "TEMPORAL",
        "CAUSAL",
        "COMPARATIVE",
        "LOGICAL",
    }

    EXPECTED_COUNT = 6

    def test_claim_type_enum(self) -> None:
        """Test that ClaimType is an Enum with expected values."""
        # Test it's an Enum
        assert issubclass(ClaimType, Enum)

        # Test all expected values exist
        actual_names = {member.name for member in ClaimType}
        assert actual_names == self.EXPECTED_CLAIM_TYPES

        # Test exact count
        assert len(ClaimType) == self.EXPECTED_COUNT

    def test_is_enum(self) -> None:
        """Test that ClaimType is an Enum."""
        assert issubclass(ClaimType, Enum)

    def test_all_expected_values_exist(self) -> None:
        """Test that all 6 expected claim types exist."""
        actual_names = {member.name for member in ClaimType}
        assert actual_names == self.EXPECTED_CLAIM_TYPES

    def test_value_count(self) -> None:
        """Test that exactly 6 claim types are defined."""
        assert len(ClaimType) == self.EXPECTED_COUNT

    def test_values_are_strings(self) -> None:
        """Test that all enum values are strings."""
        for member in ClaimType:
            assert isinstance(member.value, str)

    def test_values_are_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [member.value for member in ClaimType]
        assert len(values) == len(set(values))

    def test_string_representation(self) -> None:
        """Test string representation of enum members."""
        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.NUMERICAL.value == "numerical"
        assert ClaimType.TEMPORAL.value == "temporal"
        assert ClaimType.CAUSAL.value == "causal"
        assert ClaimType.COMPARATIVE.value == "comparative"
        assert ClaimType.LOGICAL.value == "logical"

    def test_membership_checks(self) -> None:
        """Test membership checks work correctly."""
        # Valid values
        assert ClaimType.FACTUAL in ClaimType
        assert ClaimType.NUMERICAL in ClaimType
        assert ClaimType.TEMPORAL in ClaimType
        assert ClaimType.CAUSAL in ClaimType
        assert ClaimType.COMPARATIVE in ClaimType
        assert ClaimType.LOGICAL in ClaimType

    def test_access_by_name(self) -> None:
        """Test that enum members can be accessed by name."""
        assert ClaimType["FACTUAL"] == ClaimType.FACTUAL
        assert ClaimType["NUMERICAL"] == ClaimType.NUMERICAL
        assert ClaimType["TEMPORAL"] == ClaimType.TEMPORAL
        assert ClaimType["CAUSAL"] == ClaimType.CAUSAL
        assert ClaimType["COMPARATIVE"] == ClaimType.COMPARATIVE
        assert ClaimType["LOGICAL"] == ClaimType.LOGICAL

    def test_access_by_value(self) -> None:
        """Test that enum members can be accessed by value."""
        assert ClaimType("factual") == ClaimType.FACTUAL
        assert ClaimType("numerical") == ClaimType.NUMERICAL
        assert ClaimType("temporal") == ClaimType.TEMPORAL
        assert ClaimType("causal") == ClaimType.CAUSAL
        assert ClaimType("comparative") == ClaimType.COMPARATIVE
        assert ClaimType("logical") == ClaimType.LOGICAL

    def test_comparison(self) -> None:
        """Test that enum members can be compared."""
        factual = ClaimType.FACTUAL
        numerical = ClaimType.NUMERICAL

        # Same member comparison
        assert factual == ClaimType.FACTUAL
        assert numerical == ClaimType.NUMERICAL

        # Different member comparison
        assert factual != numerical
        assert numerical != ClaimType.LOGICAL

    def test_iteration(self) -> None:
        """Test that we can iterate over all enum members."""
        members = list(ClaimType)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, ClaimType) for m in members)

    def test_claim_type_values_lowercase(self) -> None:
        """Test that all claim type values are lowercase strings."""
        for member in ClaimType:
            assert member.value.islower()
            assert " " not in member.value  # No spaces
            assert member.value.isalpha()  # Only letters

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            ClaimType("invalid")

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid names raise KeyError."""
        with pytest.raises(KeyError):
            ClaimType["INVALID"]


# ============================================================================
# Claim Model Tests
# ============================================================================


def test_claim_model() -> None:
    """Test the Claim model with all fields and validations."""
    # Test creating a Claim with all required fields
    claim = Claim(
        claim_id="claim_001",
        text="The Earth orbits the Sun",
        claim_type=ClaimType.FACTUAL,
        source_span=(0, 24),
        confidence=0.95,
    )

    assert claim.claim_id == "claim_001"
    assert claim.text == "The Earth orbits the Sun"
    assert claim.claim_type == ClaimType.FACTUAL
    assert claim.source_span == (0, 24)
    assert claim.confidence == 0.95

    # Test creating a Claim without source_span (optional field)
    claim_no_span = Claim(
        claim_id="claim_002",
        text="If X then Y",
        claim_type=ClaimType.LOGICAL,
        confidence=0.8,
    )

    assert claim_no_span.claim_id == "claim_002"
    assert claim_no_span.text == "If X then Y"
    assert claim_no_span.claim_type == ClaimType.LOGICAL
    assert claim_no_span.source_span is None
    assert claim_no_span.confidence == 0.8

    # Test confidence validation - valid boundary values
    claim_min = Claim(
        claim_id="c1",
        text="test",
        claim_type=ClaimType.FACTUAL,
        confidence=0.0,
    )
    assert claim_min.confidence == 0.0

    claim_max = Claim(
        claim_id="c2",
        text="test",
        claim_type=ClaimType.FACTUAL,
        confidence=1.0,
    )
    assert claim_max.confidence == 1.0

    # Test confidence validation - below range should fail
    with pytest.raises(ValidationError) as exc_info:
        Claim(
            claim_id="c3",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=-0.1,
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "confidence" in str(errors[0]["loc"])
    assert "Confidence must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test confidence validation - above range should fail
    with pytest.raises(ValidationError) as exc_info:
        Claim(
            claim_id="c4",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=1.5,
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "confidence" in str(errors[0]["loc"])
    assert "Confidence must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test serialization (mode='python' returns enum objects)
    data = claim.model_dump()
    assert data["claim_id"] == "claim_001"
    assert data["text"] == "The Earth orbits the Sun"
    assert data["claim_type"] == ClaimType.FACTUAL
    assert data["source_span"] == (0, 24)
    assert data["confidence"] == 0.95

    # Test JSON-mode serialization (enums are converted to values)
    import json

    json_data = json.loads(claim.model_dump_json())
    assert json_data["claim_type"] == "factual"

    # Test deserialization
    claim_from_dict = Claim.model_validate(
        {
            "claim_id": "claim_003",
            "text": "Test claim",
            "claim_type": "causal",
            "source_span": (5, 15),
            "confidence": 0.75,
        }
    )
    assert claim_from_dict.claim_id == "claim_003"
    assert claim_from_dict.text == "Test claim"
    assert claim_from_dict.claim_type == ClaimType.CAUSAL
    assert claim_from_dict.source_span == (5, 15)
    assert claim_from_dict.confidence == 0.75


# ============================================================================
# EvidenceSource Tests
# ============================================================================


class TestEvidenceSource:
    """Test suite for EvidenceSource enum (4 values)."""

    EXPECTED_SOURCES = {
        "INTERNAL",
        "EXTERNAL",
        "COMPUTED",
        "USER_PROVIDED",
    }

    EXPECTED_COUNT = 4

    def test_is_enum(self) -> None:
        """Test that EvidenceSource is an Enum."""
        assert issubclass(EvidenceSource, Enum)

    def test_all_expected_values_exist(self) -> None:
        """Test that all 4 expected evidence sources exist."""
        actual_names = {member.name for member in EvidenceSource}
        assert actual_names == self.EXPECTED_SOURCES

    def test_value_count(self) -> None:
        """Test that exactly 4 evidence sources are defined."""
        assert len(EvidenceSource) == self.EXPECTED_COUNT

    def test_values_are_strings(self) -> None:
        """Test that all enum values are strings."""
        for member in EvidenceSource:
            assert isinstance(member.value, str)

    def test_values_are_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [member.value for member in EvidenceSource]
        assert len(values) == len(set(values))

    def test_string_representation(self) -> None:
        """Test string representation of enum members."""
        assert EvidenceSource.INTERNAL.value == "internal"
        assert EvidenceSource.EXTERNAL.value == "external"
        assert EvidenceSource.COMPUTED.value == "computed"
        assert EvidenceSource.USER_PROVIDED.value == "user_provided"

    def test_membership_checks(self) -> None:
        """Test membership checks work correctly."""
        assert EvidenceSource.INTERNAL in EvidenceSource
        assert EvidenceSource.EXTERNAL in EvidenceSource
        assert EvidenceSource.COMPUTED in EvidenceSource
        assert EvidenceSource.USER_PROVIDED in EvidenceSource

    def test_access_by_name(self) -> None:
        """Test that enum members can be accessed by name."""
        assert EvidenceSource["INTERNAL"] == EvidenceSource.INTERNAL
        assert EvidenceSource["EXTERNAL"] == EvidenceSource.EXTERNAL
        assert EvidenceSource["COMPUTED"] == EvidenceSource.COMPUTED
        assert EvidenceSource["USER_PROVIDED"] == EvidenceSource.USER_PROVIDED

    def test_access_by_value(self) -> None:
        """Test that enum members can be accessed by value."""
        assert EvidenceSource("internal") == EvidenceSource.INTERNAL
        assert EvidenceSource("external") == EvidenceSource.EXTERNAL
        assert EvidenceSource("computed") == EvidenceSource.COMPUTED
        assert EvidenceSource("user_provided") == EvidenceSource.USER_PROVIDED

    def test_comparison(self) -> None:
        """Test that enum members can be compared."""
        internal = EvidenceSource.INTERNAL
        external = EvidenceSource.EXTERNAL

        # Same member comparison
        assert internal == EvidenceSource.INTERNAL
        assert external == EvidenceSource.EXTERNAL

        # Different member comparison
        assert internal != external
        assert external != EvidenceSource.COMPUTED

    def test_iteration(self) -> None:
        """Test that we can iterate over all enum members."""
        members = list(EvidenceSource)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, EvidenceSource) for m in members)

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            EvidenceSource("invalid")

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid names raise KeyError."""
        with pytest.raises(KeyError):
            EvidenceSource["INVALID"]


# ============================================================================
# Evidence Model Tests
# ============================================================================


def test_evidence_model() -> None:
    """Test the Evidence model with all fields and validations."""
    # Test creating Evidence with all fields
    evidence = Evidence(
        evidence_id="ev_001",
        source=EvidenceSource.EXTERNAL,
        content="According to NASA, Earth completes one orbit in 365.25 days",
        relevance_score=0.95,
        url="https://nasa.gov/earth",
    )

    assert evidence.evidence_id == "ev_001"
    assert evidence.source == EvidenceSource.EXTERNAL
    assert evidence.content == "According to NASA, Earth completes one orbit in 365.25 days"
    assert evidence.relevance_score == 0.95
    assert evidence.url == "https://nasa.gov/earth"

    # Test creating Evidence without URL (optional field)
    evidence_no_url = Evidence(
        evidence_id="ev_002",
        source=EvidenceSource.COMPUTED,
        content="Calculated result: 2 + 2 = 4",
        relevance_score=1.0,
    )

    assert evidence_no_url.evidence_id == "ev_002"
    assert evidence_no_url.source == EvidenceSource.COMPUTED
    assert evidence_no_url.content == "Calculated result: 2 + 2 = 4"
    assert evidence_no_url.relevance_score == 1.0
    assert evidence_no_url.url is None

    # Test relevance_score validation - valid boundary values
    evidence_min = Evidence(
        evidence_id="e1",
        source=EvidenceSource.INTERNAL,
        content="test",
        relevance_score=0.0,
    )
    assert evidence_min.relevance_score == 0.0

    evidence_max = Evidence(
        evidence_id="e2",
        source=EvidenceSource.INTERNAL,
        content="test",
        relevance_score=1.0,
    )
    assert evidence_max.relevance_score == 1.0

    # Test relevance_score validation - below range should fail
    with pytest.raises(ValidationError) as exc_info:
        Evidence(
            evidence_id="e3",
            source=EvidenceSource.INTERNAL,
            content="test",
            relevance_score=-0.1,
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "relevance_score" in str(errors[0]["loc"])
    assert "Relevance score must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test relevance_score validation - above range should fail
    with pytest.raises(ValidationError) as exc_info:
        Evidence(
            evidence_id="e4",
            source=EvidenceSource.INTERNAL,
            content="test",
            relevance_score=1.2,
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "relevance_score" in str(errors[0]["loc"])
    assert "Relevance score must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test serialization (mode='python' returns enum objects)
    data = evidence.model_dump()
    assert data["evidence_id"] == "ev_001"
    assert data["source"] == EvidenceSource.EXTERNAL
    assert data["content"] == "According to NASA, Earth completes one orbit in 365.25 days"
    assert data["relevance_score"] == 0.95
    assert data["url"] == "https://nasa.gov/earth"

    # Test JSON-mode serialization (enums are converted to values)
    import json

    json_data = json.loads(evidence.model_dump_json())
    assert json_data["source"] == "external"

    # Test deserialization
    evidence_from_dict = Evidence.model_validate(
        {
            "evidence_id": "ev_003",
            "source": "computed",
            "content": "Mathematical proof",
            "relevance_score": 0.88,
            "url": None,
        }
    )
    assert evidence_from_dict.evidence_id == "ev_003"
    assert evidence_from_dict.source == EvidenceSource.COMPUTED
    assert evidence_from_dict.content == "Mathematical proof"
    assert evidence_from_dict.relevance_score == 0.88
    assert evidence_from_dict.url is None


# ============================================================================
# VerificationResult Model Tests
# ============================================================================


def test_verification_result_model() -> None:
    """Test the VerificationResult model with nested Claim and Evidence."""
    # Create test claim
    claim = Claim(
        claim_id="c1",
        text="Earth orbits the Sun",
        claim_type=ClaimType.FACTUAL,
        confidence=0.9,
    )

    # Create test evidence
    evidence1 = Evidence(
        evidence_id="e1",
        source=EvidenceSource.EXTERNAL,
        content="NASA confirms orbital period",
        relevance_score=0.95,
    )
    evidence2 = Evidence(
        evidence_id="e2",
        source=EvidenceSource.COMPUTED,
        content="Calculation verified",
        relevance_score=0.85,
        url="https://example.com",
    )

    # Test creating VerificationResult with all fields
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.VERIFIED,
        confidence=0.95,
        evidence=[evidence1, evidence2],
        reasoning="Multiple authoritative sources confirm this fact",
    )

    assert result.claim == claim
    assert result.status == VerificationStatus.VERIFIED
    assert result.confidence == 0.95
    assert len(result.evidence) == 2
    assert result.evidence[0] == evidence1
    assert result.evidence[1] == evidence2
    assert result.reasoning == "Multiple authoritative sources confirm this fact"

    # Test creating VerificationResult with empty evidence list
    result_no_evidence = VerificationResult(
        claim=claim,
        status=VerificationStatus.UNCERTAIN,
        confidence=0.5,
        evidence=[],
        reasoning="Insufficient evidence to verify",
    )

    assert result_no_evidence.claim == claim
    assert result_no_evidence.status == VerificationStatus.UNCERTAIN
    assert result_no_evidence.confidence == 0.5
    assert len(result_no_evidence.evidence) == 0
    assert result_no_evidence.reasoning == "Insufficient evidence to verify"

    # Test confidence validation - valid boundary values
    result_min = VerificationResult(
        claim=claim,
        status=VerificationStatus.VERIFIED,
        confidence=0.0,
        evidence=[],
        reasoning="test",
    )
    assert result_min.confidence == 0.0

    result_max = VerificationResult(
        claim=claim,
        status=VerificationStatus.VERIFIED,
        confidence=1.0,
        evidence=[],
        reasoning="test",
    )
    assert result_max.confidence == 1.0

    # Test confidence validation - below range should fail
    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=-0.1,
            evidence=[],
            reasoning="test",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "confidence" in str(errors[0]["loc"])
    assert "Confidence must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test confidence validation - above range should fail
    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=1.5,
            evidence=[],
            reasoning="test",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "confidence" in str(errors[0]["loc"])
    assert "Confidence must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test serialization
    data = result.model_dump()
    assert data["claim"]["claim_id"] == "c1"
    assert data["status"] == VerificationStatus.VERIFIED
    assert data["confidence"] == 0.95
    assert len(data["evidence"]) == 2
    assert data["evidence"][0]["evidence_id"] == "e1"

    # Test JSON serialization
    import json

    json_data = json.loads(result.model_dump_json())
    assert json_data["status"] == "verified"
    assert json_data["claim"]["claim_type"] == "factual"
    assert json_data["evidence"][0]["source"] == "external"

    # Test deserialization
    result_from_dict = VerificationResult.model_validate(
        {
            "claim": {
                "claim_id": "c2",
                "text": "Test claim",
                "claim_type": "numerical",
                "confidence": 0.8,
            },
            "status": "refuted",
            "confidence": 0.75,
            "evidence": [
                {
                    "evidence_id": "e3",
                    "source": "internal",
                    "content": "Internal check",
                    "relevance_score": 0.7,
                }
            ],
            "reasoning": "Claim contradicts known facts",
        }
    )
    assert result_from_dict.claim.claim_id == "c2"
    assert result_from_dict.status == VerificationStatus.REFUTED
    assert result_from_dict.confidence == 0.75
    assert len(result_from_dict.evidence) == 1
    assert result_from_dict.evidence[0].source == EvidenceSource.INTERNAL


# ============================================================================
# VerificationReport Model Tests
# ============================================================================


def test_verification_report_model() -> None:
    """Test the VerificationReport model with lists of claims and results."""
    # Create test claims
    claim1 = Claim(
        claim_id="c1",
        text="Test claim 1",
        claim_type=ClaimType.FACTUAL,
        confidence=0.9,
    )
    claim2 = Claim(
        claim_id="c2",
        text="Test claim 2",
        claim_type=ClaimType.LOGICAL,
        confidence=0.7,
    )
    claim3 = Claim(
        claim_id="c3",
        text="Test claim 3",
        claim_type=ClaimType.NUMERICAL,
        confidence=0.85,
    )

    # Create test results
    result1 = VerificationResult(
        claim=claim1,
        status=VerificationStatus.VERIFIED,
        confidence=0.95,
        evidence=[],
        reasoning="Verified through external sources",
    )
    result2 = VerificationResult(
        claim=claim2,
        status=VerificationStatus.UNCERTAIN,
        confidence=0.6,
        evidence=[],
        reasoning="Unable to fully verify logical claim",
    )

    # Test creating VerificationReport with all fields
    report = VerificationReport(
        report_id="report_001",
        original_text="Original reasoning text with multiple claims",
        claims=[claim1, claim2, claim3],
        results=[result1, result2],
        overall_accuracy=0.85,
        flagged_claims=["c2", "c3"],
    )

    assert report.report_id == "report_001"
    assert report.original_text == "Original reasoning text with multiple claims"
    assert len(report.claims) == 3
    assert report.claims[0] == claim1
    assert report.claims[1] == claim2
    assert report.claims[2] == claim3
    assert len(report.results) == 2
    assert report.results[0] == result1
    assert report.results[1] == result2
    assert report.overall_accuracy == 0.85
    assert len(report.flagged_claims) == 2
    assert "c2" in report.flagged_claims
    assert "c3" in report.flagged_claims

    # Test creating VerificationReport with empty lists
    report_empty = VerificationReport(
        report_id="report_002",
        original_text="Text with no claims",
        claims=[],
        results=[],
        overall_accuracy=0.0,
        flagged_claims=[],
    )

    assert report_empty.report_id == "report_002"
    assert len(report_empty.claims) == 0
    assert len(report_empty.results) == 0
    assert report_empty.overall_accuracy == 0.0
    assert len(report_empty.flagged_claims) == 0

    # Test overall_accuracy validation - valid boundary values
    report_min = VerificationReport(
        report_id="r1",
        original_text="test",
        claims=[],
        results=[],
        overall_accuracy=0.0,
        flagged_claims=[],
    )
    assert report_min.overall_accuracy == 0.0

    report_max = VerificationReport(
        report_id="r2",
        original_text="test",
        claims=[],
        results=[],
        overall_accuracy=1.0,
        flagged_claims=[],
    )
    assert report_max.overall_accuracy == 1.0

    # Test overall_accuracy validation - below range should fail
    with pytest.raises(ValidationError) as exc_info:
        VerificationReport(
            report_id="r3",
            original_text="test",
            claims=[],
            results=[],
            overall_accuracy=-0.1,
            flagged_claims=[],
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "overall_accuracy" in str(errors[0]["loc"])
    assert "Overall accuracy must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test overall_accuracy validation - above range should fail
    with pytest.raises(ValidationError) as exc_info:
        VerificationReport(
            report_id="r4",
            original_text="test",
            claims=[],
            results=[],
            overall_accuracy=1.1,
            flagged_claims=[],
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "overall_accuracy" in str(errors[0]["loc"])
    assert "Overall accuracy must be between 0.0 and 1.0" in str(errors[0]["msg"])

    # Test serialization
    data = report.model_dump()
    assert data["report_id"] == "report_001"
    assert len(data["claims"]) == 3
    assert len(data["results"]) == 2
    assert data["overall_accuracy"] == 0.85
    assert len(data["flagged_claims"]) == 2

    # Test JSON serialization
    import json

    json_data = json.loads(report.model_dump_json())
    assert json_data["report_id"] == "report_001"
    assert json_data["claims"][0]["claim_type"] == "factual"
    assert json_data["results"][0]["status"] == "verified"

    # Test deserialization
    report_from_dict = VerificationReport.model_validate(
        {
            "report_id": "report_003",
            "original_text": "Test text",
            "claims": [
                {
                    "claim_id": "c4",
                    "text": "Claim 4",
                    "claim_type": "causal",
                    "confidence": 0.65,
                }
            ],
            "results": [],
            "overall_accuracy": 0.75,
            "flagged_claims": ["c4"],
        }
    )
    assert report_from_dict.report_id == "report_003"
    assert len(report_from_dict.claims) == 1
    assert report_from_dict.claims[0].claim_id == "c4"
    assert report_from_dict.overall_accuracy == 0.75


# ============================================================================
# HallucinationFlag Model Tests
# ============================================================================


def test_hallucination_flag_model() -> None:
    """Test the HallucinationFlag model with Literal severity field."""
    # Test creating HallucinationFlag with all fields
    flag_high = HallucinationFlag(
        claim_id="c1",
        severity="high",
        explanation="Claim contradicts established facts",
        suggested_correction="The correct fact is X, not Y",
    )

    assert flag_high.claim_id == "c1"
    assert flag_high.severity == "high"
    assert flag_high.explanation == "Claim contradicts established facts"
    assert flag_high.suggested_correction == "The correct fact is X, not Y"

    # Test creating HallucinationFlag without suggested_correction (optional field)
    flag_no_correction = HallucinationFlag(
        claim_id="c2",
        severity="low",
        explanation="Minor inconsistency detected",
    )

    assert flag_no_correction.claim_id == "c2"
    assert flag_no_correction.severity == "low"
    assert flag_no_correction.explanation == "Minor inconsistency detected"
    assert flag_no_correction.suggested_correction is None

    # Test all valid severity levels
    flag_low = HallucinationFlag(
        claim_id="c3",
        severity="low",
        explanation="Low severity issue",
    )
    assert flag_low.severity == "low"

    flag_medium = HallucinationFlag(
        claim_id="c4",
        severity="medium",
        explanation="Medium severity issue",
    )
    assert flag_medium.severity == "medium"

    flag_high_2 = HallucinationFlag(
        claim_id="c5",
        severity="high",
        explanation="High severity issue",
    )
    assert flag_high_2.severity == "high"

    # Test invalid severity level should fail
    with pytest.raises(ValidationError) as exc_info:
        HallucinationFlag(
            claim_id="c6",
            severity="critical",  # type: ignore[arg-type]
            explanation="Invalid severity",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "severity" in str(errors[0]["loc"])

    # Test another invalid severity level
    with pytest.raises(ValidationError) as exc_info:
        HallucinationFlag(
            claim_id="c7",
            severity="extreme",  # type: ignore[arg-type]
            explanation="Invalid severity",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "severity" in str(errors[0]["loc"])

    # Test serialization
    data = flag_high.model_dump()
    assert data["claim_id"] == "c1"
    assert data["severity"] == "high"
    assert data["explanation"] == "Claim contradicts established facts"
    assert data["suggested_correction"] == "The correct fact is X, not Y"

    # Test JSON serialization
    import json

    json_data = json.loads(flag_high.model_dump_json())
    assert json_data["claim_id"] == "c1"
    assert json_data["severity"] == "high"

    # Test deserialization
    flag_from_dict = HallucinationFlag.model_validate(
        {
            "claim_id": "c8",
            "severity": "medium",
            "explanation": "Moderately concerning issue",
            "suggested_correction": "Consider rephrasing as...",
        }
    )
    assert flag_from_dict.claim_id == "c8"
    assert flag_from_dict.severity == "medium"
    assert flag_from_dict.explanation == "Moderately concerning issue"
    assert flag_from_dict.suggested_correction == "Consider rephrasing as..."
