"""Tests for fact checker implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.engine.executor import ExecutionContext
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.verification import Claim, ClaimType, VerificationStatus
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.verification.checkers import (
    ExternalSourceChecker,
    LogicalConsistencyChecker,
    NumericalChecker,
    SelfConsistencyChecker,
)


@pytest.fixture
def mock_context() -> ExecutionContext:
    """Create a mock execution context for testing."""
    session = Session().start()
    registry = MethodRegistry()
    return ExecutionContext(
        session=session,
        registry=registry,
        input_data={},
        variables={},
    )


@pytest.fixture
def mock_context_with_sampling(mock_context: ExecutionContext) -> ExecutionContext:
    """Create a mock execution context with sampling capability."""
    # Mock the FastMCP context
    mock_ctx = MagicMock()
    mock_context.ctx = mock_ctx
    return mock_context


@pytest.fixture
def sample_claim() -> Claim:
    """Create a sample claim for testing."""
    return Claim(
        claim_id="test_claim_1",
        text="The Earth orbits the Sun",
        claim_type=ClaimType.FACTUAL,
        source_span=(0, 26),
        confidence=0.9,
    )


@pytest.fixture
def numerical_claim() -> Claim:
    """Create a numerical claim for testing."""
    return Claim(
        claim_id="test_claim_2",
        text="2 + 2 = 4",
        claim_type=ClaimType.NUMERICAL,
        source_span=(0, 9),
        confidence=1.0,
    )


class TestCheckerProtocol:
    """Test that checkers conform to the FactChecker protocol."""

    def test_self_consistency_checker_conforms(self) -> None:
        """Test that SelfConsistencyChecker implements FactChecker protocol."""
        checker = SelfConsistencyChecker()
        assert hasattr(checker, "check")
        assert callable(checker.check)

    def test_logical_consistency_checker_conforms(self) -> None:
        """Test that LogicalConsistencyChecker implements FactChecker protocol."""
        checker = LogicalConsistencyChecker()
        assert hasattr(checker, "check")
        assert callable(checker.check)

    def test_numerical_checker_conforms(self) -> None:
        """Test that NumericalChecker implements FactChecker protocol."""
        checker = NumericalChecker()
        assert hasattr(checker, "check")
        assert callable(checker.check)

    def test_external_source_checker_conforms(self) -> None:
        """Test that ExternalSourceChecker implements FactChecker protocol."""
        checker = ExternalSourceChecker()
        assert hasattr(checker, "check")
        assert callable(checker.check)


class TestSelfConsistencyChecker:
    """Test suite for SelfConsistencyChecker."""

    def test_initialization(self) -> None:
        """Test checker initialization with parameters."""
        checker = SelfConsistencyChecker(num_samples=7, temperature=0.5)
        assert checker.num_samples == 7
        assert checker.temperature == 0.5

    def test_initialization_clamping(self) -> None:
        """Test that parameters are clamped to valid ranges."""
        # Test num_samples clamping
        checker_low = SelfConsistencyChecker(num_samples=1)
        assert checker_low.num_samples == 3  # Clamped to minimum

        checker_high = SelfConsistencyChecker(num_samples=20)
        assert checker_high.num_samples == 10  # Clamped to maximum

        # Test temperature clamping
        checker_temp_low = SelfConsistencyChecker(temperature=-1.0)
        assert checker_temp_low.temperature == 0.0  # Clamped to minimum

        checker_temp_high = SelfConsistencyChecker(temperature=5.0)
        assert checker_temp_high.temperature == 2.0  # Clamped to maximum

    async def test_check_without_sampling(
        self, sample_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test that check returns UNVERIFIABLE when sampling is not available."""
        checker = SelfConsistencyChecker()
        result = await checker.check(sample_claim, mock_context)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence == 0.0
        assert len(result.evidence) == 0
        assert "requires LLM sampling" in result.reasoning

    async def test_check_with_high_agreement(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test verification with >80% agreement across samples."""
        checker = SelfConsistencyChecker(num_samples=5)

        # Mock sample responses with high agreement (4 out of 5 agree)
        responses = ["AGREE", "AGREE", "AGREE", "AGREE", "DISAGREE"]
        mock_context_with_sampling.sample = AsyncMock(side_effect=responses)

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence >= 0.8
        assert len(result.evidence) == 5
        assert "4/5" in result.reasoning and (
            "80" in result.reasoning or ">=80" in result.reasoning
        )

    async def test_check_with_medium_agreement(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test verification with 50-80% agreement."""
        checker = SelfConsistencyChecker(num_samples=5)

        # Mock sample responses with medium agreement (3 out of 5)
        responses = ["AGREE", "AGREE", "AGREE", "DISAGREE", "DISAGREE"]
        mock_context_with_sampling.sample = AsyncMock(side_effect=responses)

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.UNCERTAIN
        assert result.confidence == 0.5
        assert len(result.evidence) == 5
        assert "3/5" in result.reasoning

    async def test_check_with_low_agreement(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test verification with <50% agreement (refuted)."""
        checker = SelfConsistencyChecker(num_samples=5)

        # Mock sample responses with low agreement (1 out of 5)
        responses = ["DISAGREE", "DISAGREE", "DISAGREE", "DISAGREE", "AGREE"]
        mock_context_with_sampling.sample = AsyncMock(side_effect=responses)

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence > 0.5  # High confidence in refutation
        assert len(result.evidence) == 5
        assert "1/5" in result.reasoning

    async def test_check_with_failed_samples(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test handling of failed LLM samples."""
        checker = SelfConsistencyChecker(num_samples=3)

        # Mock sample to raise exception
        mock_context_with_sampling.sample = AsyncMock(side_effect=Exception("Sampling failed"))

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence == 0.0
        assert len(result.evidence) == 3  # Evidence of failures
        assert "All LLM samples failed" in result.reasoning


class TestLogicalConsistencyChecker:
    """Test suite for LogicalConsistencyChecker."""

    async def test_check_without_sampling(
        self, sample_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test that check returns UNVERIFIABLE when sampling is not available."""
        checker = LogicalConsistencyChecker()
        result = await checker.check(sample_claim, mock_context)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence == 0.0
        assert "requires LLM sampling" in result.reasoning

    async def test_check_valid_logic(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test verification of logically valid claim."""
        checker = LogicalConsistencyChecker()
        mock_context_with_sampling.sample = AsyncMock(
            return_value="VALID: The claim follows logically from the premises."
        )

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 0.9
        assert len(result.evidence) == 1
        assert "VALID" in result.reasoning

    async def test_check_invalid_logic(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test detection of logically invalid claim."""
        checker = LogicalConsistencyChecker()
        mock_context_with_sampling.sample = AsyncMock(
            return_value="INVALID: The conclusion does not follow from the premises."
        )

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == 0.9
        assert len(result.evidence) == 1
        assert "INVALID" in result.reasoning

    async def test_check_uncertain_logic(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test handling of uncertain logical analysis."""
        checker = LogicalConsistencyChecker()
        mock_context_with_sampling.sample = AsyncMock(
            return_value="UNCERTAIN: Cannot determine logical validity from context."
        )

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.UNCERTAIN
        assert result.confidence == 0.5
        assert len(result.evidence) == 1
        assert "UNCERTAIN" in result.reasoning

    async def test_check_with_context(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test that checker uses reasoning context when available."""
        mock_context_with_sampling.input_data["reasoning_text"] = (
            "All planets orbit the Sun. Earth is a planet."
        )
        checker = LogicalConsistencyChecker()
        mock_context_with_sampling.sample = AsyncMock(return_value="VALID: Logic holds.")

        result = await checker.check(sample_claim, mock_context_with_sampling)

        # Verify the sample was called (context was used)
        assert mock_context_with_sampling.sample.called
        assert result.status == VerificationStatus.VERIFIED

    async def test_check_sampling_error(
        self, sample_claim: Claim, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test handling of sampling errors."""
        checker = LogicalConsistencyChecker()
        mock_context_with_sampling.sample = AsyncMock(side_effect=Exception("Sampling error"))

        result = await checker.check(sample_claim, mock_context_with_sampling)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence == 0.0
        assert "failed" in result.reasoning.lower()


class TestNumericalChecker:
    """Test suite for NumericalChecker."""

    async def test_check_non_numerical_claim(
        self, sample_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test that checker rejects non-numerical claims."""
        checker = NumericalChecker()
        result = await checker.check(sample_claim, mock_context)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert "only handles NUMERICAL claims" in result.reasoning

    async def test_check_simple_addition(
        self, numerical_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test verification of simple arithmetic (2 + 2 = 4)."""
        checker = NumericalChecker()
        result = await checker.check(numerical_claim, mock_context)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 1.0
        assert len(result.evidence) == 1
        assert "Computation verified" in result.reasoning

    async def test_check_incorrect_calculation(self, mock_context: ExecutionContext) -> None:
        """Test detection of incorrect calculation."""
        incorrect_claim = Claim(
            claim_id="test_claim_3",
            text="2 + 2 = 5",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.8,
        )
        checker = NumericalChecker()
        result = await checker.check(incorrect_claim, mock_context)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == 1.0
        assert "Computation error" in result.reasoning

    async def test_check_multiplication(self, mock_context: ExecutionContext) -> None:
        """Test verification of multiplication."""
        mult_claim = Claim(
            claim_id="test_claim_4",
            text="3 * 7 = 21",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )
        checker = NumericalChecker()
        result = await checker.check(mult_claim, mock_context)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 1.0

    async def test_check_division(self, mock_context: ExecutionContext) -> None:
        """Test verification of division."""
        div_claim = Claim(
            claim_id="test_claim_5",
            text="10 / 2 = 5",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )
        checker = NumericalChecker()
        result = await checker.check(div_claim, mock_context)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 1.0

    async def test_check_floating_point(self, mock_context: ExecutionContext) -> None:
        """Test verification with floating point numbers."""
        float_claim = Claim(
            claim_id="test_claim_6",
            text="1.5 + 2.5 = 4.0",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )
        checker = NumericalChecker()
        result = await checker.check(float_claim, mock_context)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 1.0

    async def test_safe_eval_rejects_unsafe_code(self) -> None:
        """Test that _safe_eval rejects unsafe expressions."""
        checker = NumericalChecker()

        # Test various unsafe patterns
        unsafe_expressions = [
            "__import__('os')",
            "exec('print(1)')",
            "eval('2+2')",
            "open('/etc/passwd')",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(ValueError, match="Unsafe pattern detected"):
                checker._safe_eval(expr)

    async def test_check_with_llm_fallback(
        self, mock_context_with_sampling: ExecutionContext
    ) -> None:
        """Test LLM fallback for complex expressions."""
        complex_claim = Claim(
            claim_id="test_claim_7",
            text="sqrt(16) = 4",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.9,
        )

        checker = NumericalChecker()
        mock_context_with_sampling.sample = AsyncMock(return_value="CORRECT: sqrt(16) equals 4")

        result = await checker.check(complex_claim, mock_context_with_sampling)

        # Should use LLM fallback since sqrt is not in safe_dict
        assert result.status == VerificationStatus.VERIFIED
        assert mock_context_with_sampling.sample.called

    async def test_check_malformed_equation(self, mock_context: ExecutionContext) -> None:
        """Test handling of malformed equations without LLM."""
        malformed_claim = Claim(
            claim_id="test_claim_8",
            text="This is not an equation",
            claim_type=ClaimType.NUMERICAL,
            confidence=0.5,
        )

        checker = NumericalChecker()
        result = await checker.check(malformed_claim, mock_context)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert "Could not parse" in result.reasoning


class TestExternalSourceChecker:
    """Test suite for ExternalSourceChecker."""

    async def test_check_returns_unverifiable(
        self, sample_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test that checker returns UNVERIFIABLE (placeholder)."""
        checker = ExternalSourceChecker()
        result = await checker.check(sample_claim, mock_context)

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence == 0.0
        assert len(result.evidence) == 0
        assert "not yet implemented" in result.reasoning

    async def test_check_explains_future_integration(
        self, numerical_claim: Claim, mock_context: ExecutionContext
    ) -> None:
        """Test that checker explains future integration plans."""
        checker = ExternalSourceChecker()
        result = await checker.check(numerical_claim, mock_context)

        assert "Future integration" in result.reasoning
        assert "search engines" in result.reasoning or "knowledge bases" in result.reasoning
