"""
Comprehensive tests for verification tool functions.

This module provides complete test coverage for:
- VerifyTextInput and VerifyClaimInput models
- verify_text() function
- verify_claim() function
- detect_hallucinations() function
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    HallucinationFlag,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from reasoning_mcp.tools.verify import (
    VerifyClaimInput,
    VerifyTextInput,
    detect_hallucinations,
    verify_claim,
    verify_text,
)

# ============================================================================
# Test VerifyTextInput Model
# ============================================================================


class TestVerifyTextInput:
    """Test VerifyTextInput Pydantic model."""

    def test_verify_text_input_minimal(self):
        """Test VerifyTextInput with minimal required fields."""
        input_model = VerifyTextInput(text="Test text")

        assert input_model.text == "Test text"
        assert input_model.context is None
        assert input_model.check_hallucinations is True

    def test_verify_text_input_with_context(self):
        """Test VerifyTextInput with context."""
        input_model = VerifyTextInput(
            text="Test text",
            context="Test context",
        )

        assert input_model.text == "Test text"
        assert input_model.context == "Test context"
        assert input_model.check_hallucinations is True

    def test_verify_text_input_disable_hallucination_check(self):
        """Test VerifyTextInput with hallucination check disabled."""
        input_model = VerifyTextInput(
            text="Test text",
            check_hallucinations=False,
        )

        assert input_model.text == "Test text"
        assert input_model.check_hallucinations is False

    def test_verify_text_input_all_fields(self):
        """Test VerifyTextInput with all fields."""
        input_model = VerifyTextInput(
            text="Test text",
            context="Test context",
            check_hallucinations=True,
        )

        assert input_model.text == "Test text"
        assert input_model.context == "Test context"
        assert input_model.check_hallucinations is True

    def test_verify_text_input_empty_text(self):
        """Test VerifyTextInput with empty text string."""
        input_model = VerifyTextInput(text="")

        assert input_model.text == ""

    def test_verify_text_input_long_text(self):
        """Test VerifyTextInput with very long text."""
        long_text = "Test " * 1000
        input_model = VerifyTextInput(text=long_text)

        assert input_model.text == long_text


# ============================================================================
# Test VerifyClaimInput Model
# ============================================================================


class TestVerifyClaimInput:
    """Test VerifyClaimInput Pydantic model."""

    def test_verify_claim_input_minimal(self):
        """Test VerifyClaimInput with minimal required fields."""
        input_model = VerifyClaimInput(claim_text="Test claim")

        assert input_model.claim_text == "Test claim"
        assert input_model.claim_type is None

    def test_verify_claim_input_with_claim_type(self):
        """Test VerifyClaimInput with claim_type."""
        input_model = VerifyClaimInput(
            claim_text="Test claim",
            claim_type="factual",
        )

        assert input_model.claim_text == "Test claim"
        assert input_model.claim_type == "factual"

    def test_verify_claim_input_all_claim_types(self):
        """Test VerifyClaimInput with all valid claim types."""
        claim_types = ["factual", "numerical", "temporal", "causal", "comparative", "logical"]

        for claim_type in claim_types:
            input_model = VerifyClaimInput(
                claim_text="Test claim",
                claim_type=claim_type,
            )
            assert input_model.claim_type == claim_type

    def test_verify_claim_input_empty_claim_text(self):
        """Test VerifyClaimInput with empty claim text."""
        input_model = VerifyClaimInput(claim_text="")

        assert input_model.claim_text == ""


# ============================================================================
# Test verify_text Function
# ============================================================================


class TestVerifyText:
    """Test verify_text() function."""

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_verify_text_minimal(
        self, mock_detector_class, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_text() with minimal parameters."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_report = VerificationReport(
            report_id="r1",
            original_text="Test text",
            claims=[mock_claim],
            results=[],
            overall_accuracy=0.9,
            flagged_claims=[],
        )

        mock_engine = MagicMock()
        mock_engine.verify_text = AsyncMock(return_value=mock_report)
        mock_engine_class.return_value = mock_engine

        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await verify_text("Test text")

        # Verify
        assert isinstance(result, dict)
        assert "report_id" in result
        assert "original_text" in result
        assert "claims" in result
        assert "results" in result
        assert "overall_accuracy" in result
        assert "flagged_claims" in result
        assert "hallucination_flags" in result

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_verify_text_with_context(
        self, mock_detector_class, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_text() with context provided."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_report = VerificationReport(
            report_id="r1",
            original_text="Test text",
            claims=[mock_claim],
            results=[],
            overall_accuracy=0.9,
            flagged_claims=[],
        )

        mock_engine = MagicMock()
        mock_engine.verify_text = AsyncMock(return_value=mock_report)
        mock_engine_class.return_value = mock_engine

        mock_flag = HallucinationFlag(
            claim_id="c1",
            severity="low",
            explanation="Test",
            suggested_correction=None,
        )
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[mock_flag])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await verify_text("Test text", context="Test context")

        # Verify
        assert isinstance(result, dict)
        assert "hallucination_flags" in result
        assert len(result["hallucination_flags"]) == 1

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_text_hallucination_check_disabled(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_text() with hallucination check disabled."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_report = VerificationReport(
            report_id="r1",
            original_text="Test text",
            claims=[mock_claim],
            results=[],
            overall_accuracy=0.9,
            flagged_claims=[],
        )

        mock_engine = MagicMock()
        mock_engine.verify_text = AsyncMock(return_value=mock_report)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await verify_text("Test text", check_hallucinations=False)

        # Verify
        assert isinstance(result, dict)
        assert "hallucination_flags" in result
        assert result["hallucination_flags"] == []

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_verify_text_is_async(
        self, mock_detector_class, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test that verify_text() is an async function."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_report = VerificationReport(
            report_id="r1",
            original_text="Test text",
            claims=[mock_claim],
            results=[],
            overall_accuracy=0.9,
            flagged_claims=[],
        )

        mock_engine = MagicMock()
        mock_engine.verify_text = AsyncMock(return_value=mock_report)
        mock_engine_class.return_value = mock_engine

        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        coro = verify_text("Test text")

        # Verify it's a coroutine
        assert hasattr(coro, "__await__")

        # Clean up
        await coro


# ============================================================================
# Test verify_claim Function
# ============================================================================


class TestVerifyClaim:
    """Test verify_claim() function."""

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_claim_minimal(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_claim() with minimal parameters."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_result = VerificationResult(
            claim=mock_claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Test reasoning",
        )

        mock_engine = MagicMock()
        mock_engine.verify_claim = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await verify_claim("Test claim")

        # Verify
        assert isinstance(result, dict)
        assert "claim" in result
        assert "status" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "reasoning" in result

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_claim_with_claim_type(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_claim() with claim_type specified."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="2+2=4",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )
        mock_result = VerificationResult(
            claim=mock_claim,
            status=VerificationStatus.VERIFIED,
            confidence=1.0,
            evidence=[],
            reasoning="Correct calculation",
        )

        mock_engine = MagicMock()
        mock_engine.verify_claim = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await verify_claim("2+2=4", claim_type="numerical")

        # Verify
        assert isinstance(result, dict)
        # claim_type is serialized as string value after JSON conversion
        assert result["claim"]["claim_type"] == "numerical"

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_claim_all_claim_types(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test verify_claim() with all claim types."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        claim_types = ["factual", "numerical", "temporal", "causal", "comparative", "logical"]

        for claim_type in claim_types:
            mock_claim = Claim(
                claim_id="c1",
                text="Test claim",
                claim_type=ClaimType.FACTUAL,  # Will be overridden by claim_type param
                confidence=0.9,
            )
            mock_result = VerificationResult(
                claim=mock_claim,
                status=VerificationStatus.VERIFIED,
                confidence=0.95,
                evidence=[],
                reasoning="Test",
            )

            mock_engine = MagicMock()
            mock_engine.verify_claim = AsyncMock(return_value=mock_result)
            mock_engine_class.return_value = mock_engine

            # Execute
            result = await verify_claim("Test claim", claim_type=claim_type)

            # Verify
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_claim_is_async(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test that verify_claim() is an async function."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_result = VerificationResult(
            claim=mock_claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Test",
        )

        mock_engine = MagicMock()
        mock_engine.verify_claim = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        coro = verify_claim("Test claim")

        # Verify it's a coroutine
        assert hasattr(coro, "__await__")

        # Clean up
        await coro


# ============================================================================
# Test detect_hallucinations Function
# ============================================================================


class TestDetectHallucinations:
    """Test detect_hallucinations() function."""

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_minimal(self, mock_detector_class):
        """Test detect_hallucinations() with minimal parameters."""
        # Setup mock
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await detect_hallucinations("Test text")

        # Verify
        assert isinstance(result, list)

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_with_context(self, mock_detector_class):
        """Test detect_hallucinations() with context."""
        # Setup mock
        mock_flag = HallucinationFlag(
            claim_id="c1",
            severity="medium",
            explanation="Claim not supported by context",
            suggested_correction="Correct version",
        )
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[mock_flag])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await detect_hallucinations("Test text", context="Test context")

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["claim_id"] == "c1"
        assert result[0]["severity"] == "medium"

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_multiple_flags(self, mock_detector_class):
        """Test detect_hallucinations() with multiple flags."""
        # Setup mocks
        mock_flag1 = HallucinationFlag(
            claim_id="c1",
            severity="low",
            explanation="Minor issue",
            suggested_correction=None,
        )
        mock_flag2 = HallucinationFlag(
            claim_id="c2",
            severity="high",
            explanation="Major issue",
            suggested_correction="Fix this",
        )
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[mock_flag1, mock_flag2])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await detect_hallucinations("Test text")

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["severity"] == "low"
        assert result[1]["severity"] == "high"

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_empty_text(self, mock_detector_class):
        """Test detect_hallucinations() with empty text."""
        # Setup mock
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await detect_hallucinations("")

        # Verify
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_is_async(self, mock_detector_class):
        """Test that detect_hallucinations() is an async function."""
        # Setup mock
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        coro = detect_hallucinations("Test text")

        # Verify it's a coroutine
        assert hasattr(coro, "__await__")

        # Clean up
        await coro

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_detect_hallucinations_returns_dicts(self, mock_detector_class):
        """Test that detect_hallucinations() returns list of dicts."""
        # Setup mock
        mock_flag = HallucinationFlag(
            claim_id="c1",
            severity="low",
            explanation="Test",
            suggested_correction=None,
        )
        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[mock_flag])
        mock_detector_class.return_value = mock_detector

        # Execute
        result = await detect_hallucinations("Test text")

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "claim_id" in result[0]
        assert "severity" in result[0]
        assert "explanation" in result[0]
        assert "suggested_correction" in result[0]


# ============================================================================
# Test Integration
# ============================================================================


class TestVerifyIntegration:
    """Test integration between verify functions."""

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    @patch("reasoning_mcp.tools.verify.HallucinationDetector")
    async def test_verify_text_calls_engine_correctly(
        self, mock_detector_class, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test that verify_text() calls VerificationEngine correctly."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="test",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_report = VerificationReport(
            report_id="r1",
            original_text="Test text",
            claims=[mock_claim],
            results=[],
            overall_accuracy=0.9,
            flagged_claims=[],
        )

        mock_engine = MagicMock()
        mock_engine.verify_text = AsyncMock(return_value=mock_report)
        mock_engine_class.return_value = mock_engine

        mock_detector = MagicMock()
        mock_detector.detect = AsyncMock(return_value=[])
        mock_detector_class.return_value = mock_detector

        # Execute
        await verify_text("Test text")

        # Verify engine was called
        mock_engine.verify_text.assert_called_once_with("Test text")

    @pytest.mark.asyncio
    @patch("reasoning_mcp.tools.verify.get_extractor")
    @patch("reasoning_mcp.tools.verify.SelfConsistencyChecker")
    @patch("reasoning_mcp.tools.verify.VerificationEngine")
    async def test_verify_claim_calls_engine_correctly(
        self, mock_engine_class, mock_checker_class, mock_get_extractor
    ):
        """Test that verify_claim() calls VerificationEngine correctly."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_get_extractor.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_claim = Claim(
            claim_id="c1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9,
        )
        mock_result = VerificationResult(
            claim=mock_claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[],
            reasoning="Test",
        )

        mock_engine = MagicMock()
        mock_engine.verify_claim = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        await verify_claim("Test claim")

        # Verify engine was called
        mock_engine.verify_claim.assert_called_once()
