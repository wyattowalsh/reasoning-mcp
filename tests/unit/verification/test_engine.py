"""Tests for verification engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
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
from reasoning_mcp.verification.engine import VerificationEngine
from reasoning_mcp.verification.extractors import ClaimExtractor


class TestVerificationEngineInit:
    """Tests for VerificationEngine initialization."""

    def test_engine_init_with_required_params(self) -> None:
        """Test initialization with required parameters."""
        # Arrange
        extractor = Mock(spec=ClaimExtractor)
        checkers = [Mock(spec=FactChecker)]

        # Act
        engine = VerificationEngine(extractor=extractor, checkers=checkers)

        # Assert
        assert engine.extractor == extractor
        assert engine.checkers == checkers
        assert engine.config == {}
        assert engine.ctx is None

    def test_engine_init_with_all_params(self) -> None:
        """Test initialization with all parameters."""
        # Arrange
        extractor = Mock(spec=ClaimExtractor)
        checkers = [Mock(spec=FactChecker)]
        config = {"threshold": 0.8}
        ctx = Mock()

        # Act
        engine = VerificationEngine(extractor=extractor, checkers=checkers, config=config, ctx=ctx)

        # Assert
        assert engine.extractor == extractor
        assert engine.checkers == checkers
        assert engine.config == config
        assert engine.ctx == ctx


class TestVerifyText:
    """Tests for verify_text method."""

    async def test_verify_text_full_pipeline(self) -> None:
        """Test full pipeline with mocked components."""
        # Arrange
        claim1 = Claim(
            claim_id="c1",
            text="The Earth is round",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        claim2 = Claim(claim_id="c2", text="2+2=4", claim_type=ClaimType.NUMERICAL, confidence=0.95)

        evidence1 = Evidence(
            evidence_id="e1",
            source=EvidenceSource.EXTERNAL,
            content="Scientific consensus",
            relevance_score=0.95,
        )

        result1 = VerificationResult(
            claim=claim1,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[evidence1],
            reasoning="Well-established fact",
        )
        result2 = VerificationResult(
            claim=claim2,
            status=VerificationStatus.VERIFIED,
            confidence=1.0,
            evidence=[],
            reasoning="Mathematical fact",
        )

        extractor = Mock()
        extractor.extract = Mock(return_value=[claim1, claim2])

        checker = Mock()
        checker.check = AsyncMock(side_effect=[result1, result2])

        ctx = Mock()
        engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=ctx)

        # Act
        report = await engine.verify_text("The Earth is round and 2+2=4")

        # Assert
        assert isinstance(report, VerificationReport)
        assert report.original_text == "The Earth is round and 2+2=4"
        assert len(report.claims) == 2
        assert report.claims == [claim1, claim2]
        assert len(report.results) == 2
        assert report.results == [result1, result2]
        assert report.overall_accuracy > 0.9
        assert len(report.flagged_claims) == 0
        extractor.extract.assert_called_once_with("The Earth is round and 2+2=4")

    async def test_verify_text_with_flagged_claims(self) -> None:
        """Test verify_text identifies flagged claims."""
        # Arrange
        claim1 = Claim(
            claim_id="c1",
            text="The Earth is flat",
            claim_type=ClaimType.FACTUAL,
            confidence=0.8,
        )
        claim2 = Claim(
            claim_id="c2",
            text="Uncertain claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.5,
        )

        result1 = VerificationResult(
            claim=claim1,
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=[],
            reasoning="Contradicts established science",
        )
        result2 = VerificationResult(
            claim=claim2,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.4,
            evidence=[],
            reasoning="Insufficient evidence",
        )

        extractor = Mock()
        extractor.extract = Mock(return_value=[claim1, claim2])

        checker = Mock()
        checker.check = AsyncMock(side_effect=[result1, result2])

        ctx = Mock()
        engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=ctx)

        # Act
        report = await engine.verify_text("Test text")

        # Assert
        assert len(report.flagged_claims) == 2
        assert "c1" in report.flagged_claims  # REFUTED
        assert "c2" in report.flagged_claims  # Low confidence (0.4)

    async def test_verify_text_empty_claims(self) -> None:
        """Test verify_text with no claims extracted."""
        # Arrange
        extractor = Mock()
        extractor.extract = Mock(return_value=[])

        checker = Mock()

        ctx = Mock()
        engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=ctx)

        # Act
        report = await engine.verify_text("No claims here")

        # Assert
        assert len(report.claims) == 0
        assert len(report.results) == 0
        assert report.overall_accuracy == 0.0
        assert len(report.flagged_claims) == 0


class TestVerifyClaim:
    """Tests for verify_claim method."""

    async def test_verify_claim_success(self) -> None:
        """Test single claim verification."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="The sky is blue",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )

        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Common knowledge",
        )

        extractor = Mock()
        checker = Mock()
        checker.check = AsyncMock(return_value=result)

        ctx = Mock()
        engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=ctx)

        # Act
        actual_result = await engine.verify_claim(claim)

        # Assert
        assert actual_result == result
        checker.check.assert_called_once_with(claim, ctx)

    async def test_verify_claim_uses_selected_checker(self) -> None:
        """Test that verify_claim uses the checker selected by _select_checker."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="2+2=4",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )

        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=1.0,
            evidence=[],
            reasoning="Mathematical fact",
        )

        extractor = Mock()
        numerical_checker = Mock(spec=NumericalChecker)
        numerical_checker.check = AsyncMock(return_value=result)

        ctx = Mock()
        engine = VerificationEngine(extractor=extractor, checkers=[numerical_checker], ctx=ctx)

        # Act
        actual_result = await engine.verify_claim(claim)

        # Assert
        assert actual_result == result
        numerical_checker.check.assert_called_once_with(claim, ctx)


class TestSelectChecker:
    """Tests for _select_checker method."""

    def test_select_checker_numerical(self) -> None:
        """Test selection of NumericalChecker for NUMERICAL claims."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="2+2=4",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )

        extractor = Mock(spec=ClaimExtractor)
        numerical_checker = Mock(spec=NumericalChecker)
        self_checker = Mock(spec=SelfConsistencyChecker)

        engine = VerificationEngine(extractor=extractor, checkers=[self_checker, numerical_checker])

        # Act
        checker = engine._select_checker(claim)

        # Assert
        assert checker == numerical_checker

    def test_select_checker_logical(self) -> None:
        """Test selection of LogicalConsistencyChecker for LOGICAL claims."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="If A then B",
            claim_type=ClaimType.LOGICAL,
            confidence=0.8,
        )

        extractor = Mock(spec=ClaimExtractor)
        logical_checker = Mock(spec=LogicalConsistencyChecker)
        self_checker = Mock(spec=SelfConsistencyChecker)

        engine = VerificationEngine(extractor=extractor, checkers=[self_checker, logical_checker])

        # Act
        checker = engine._select_checker(claim)

        # Assert
        assert checker == logical_checker

    def test_select_checker_fallback_to_self_consistency(self) -> None:
        """Test fallback to SelfConsistencyChecker when no specialized checker available."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="Random factual claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )

        extractor = Mock(spec=ClaimExtractor)
        self_checker = Mock(spec=SelfConsistencyChecker)
        other_checker = Mock(spec=FactChecker)

        engine = VerificationEngine(extractor=extractor, checkers=[other_checker, self_checker])

        # Act
        checker = engine._select_checker(claim)

        # Assert
        assert checker == self_checker

    def test_select_checker_default_to_first(self) -> None:
        """Test default to first checker when no specialized or SelfConsistency checker."""
        # Arrange
        claim = Claim(
            claim_id="c1",
            text="Random claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.7,
        )

        extractor = Mock(spec=ClaimExtractor)
        first_checker = Mock(spec=FactChecker)
        second_checker = Mock(spec=FactChecker)

        engine = VerificationEngine(extractor=extractor, checkers=[first_checker, second_checker])

        # Act
        checker = engine._select_checker(claim)

        # Assert
        assert checker == first_checker

    def test_select_checker_no_checkers_raises(self) -> None:
        """Test that _select_checker raises ValueError when no checkers available."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.5)

        extractor = Mock(spec=ClaimExtractor)
        engine = VerificationEngine(extractor=extractor, checkers=[])

        # Act & Assert
        with pytest.raises(ValueError, match="No checkers available"):
            engine._select_checker(claim)


class TestCalculateAccuracy:
    """Tests for calculate_accuracy method."""

    def test_calculate_accuracy_all_verified(self) -> None:
        """Test accuracy calculation with all verified results."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.9)

        result1 = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=1.0,
            evidence=[],
            reasoning="Test",
        )
        result2 = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.8,
            evidence=[],
            reasoning="Test",
        )

        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([result1, result2])

        # Assert
        # (1.0 * 1.0 + 1.0 * 0.8) / (1.0 + 0.8) = 1.8 / 1.8 = 1.0
        assert accuracy == 1.0

    def test_calculate_accuracy_mixed_statuses(self) -> None:
        """Test accuracy calculation with mixed verification statuses."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.9)

        result1 = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=1.0,
            evidence=[],
            reasoning="Test",
        )
        result2 = VerificationResult(
            claim=claim,
            status=VerificationStatus.UNCERTAIN,
            confidence=1.0,
            evidence=[],
            reasoning="Test",
        )
        result3 = VerificationResult(
            claim=claim,
            status=VerificationStatus.REFUTED,
            confidence=1.0,
            evidence=[],
            reasoning="Test",
        )

        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([result1, result2, result3])

        # Assert
        # (1.0 * 1.0 + 0.5 * 1.0 + 0.0 * 1.0) / (1.0 + 1.0 + 1.0) = 1.5 / 3.0 = 0.5
        assert accuracy == 0.5

    def test_calculate_accuracy_weighted_by_confidence(self) -> None:
        """Test accuracy calculation is weighted by confidence."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.9)

        result1 = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.8,
            evidence=[],
            reasoning="Test",
        )
        result2 = VerificationResult(
            claim=claim,
            status=VerificationStatus.REFUTED,
            confidence=0.2,
            evidence=[],
            reasoning="Test",
        )

        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([result1, result2])

        # Assert
        # (1.0 * 0.8 + 0.0 * 0.2) / (0.8 + 0.2) = 0.8 / 1.0 = 0.8
        assert accuracy == 0.8

    def test_calculate_accuracy_empty_results(self) -> None:
        """Test accuracy calculation with empty results list."""
        # Arrange
        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([])

        # Assert
        assert accuracy == 0.0

    def test_calculate_accuracy_zero_weight(self) -> None:
        """Test accuracy calculation with zero total weight."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.9)

        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.0,  # Zero confidence = zero weight
            evidence=[],
            reasoning="Test",
        )

        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([result])

        # Assert
        assert accuracy == 0.0

    def test_calculate_accuracy_unverifiable_status(self) -> None:
        """Test accuracy calculation treats UNVERIFIABLE as 0.0."""
        # Arrange
        claim = Claim(claim_id="c1", text="Test", claim_type=ClaimType.FACTUAL, confidence=0.9)

        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=1.0,
            evidence=[],
            reasoning="Test",
        )

        extractor = Mock(spec=ClaimExtractor)
        checker = Mock(spec=FactChecker)
        engine = VerificationEngine(extractor=extractor, checkers=[checker])

        # Act
        accuracy = engine.calculate_accuracy([result])

        # Assert
        # (0.0 * 1.0) / 1.0 = 0.0
        assert accuracy == 0.0
