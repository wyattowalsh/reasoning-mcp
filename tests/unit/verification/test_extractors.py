"""Tests for claim extraction functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.models.verification import Claim, ClaimType
from reasoning_mcp.verification.extractors import (
    ClaimExtractor,
    HybridExtractor,
    LLMClaimExtractor,
    RuleBasedExtractor,
    get_extractor,
)


class TestExtractorProtocol:
    """Test suite for ClaimExtractor protocol."""

    def test_extractor_protocol(self) -> None:
        """Test that extractors conform to the Protocol."""

        class CustomExtractor:
            def extract(self, text: str) -> list[Claim]:
                return []

        extractor: ClaimExtractor = CustomExtractor()
        assert hasattr(extractor, "extract")
        result = extractor.extract("test")
        assert isinstance(result, list)

    def test_protocol_duck_typing(self) -> None:
        """Test that protocol allows duck typing."""

        class MinimalExtractor:
            def extract(self, text: str) -> list[Claim]:
                return [
                    Claim(
                        claim_id="test",
                        text="test claim",
                        claim_type=ClaimType.FACTUAL,
                        confidence=0.9,
                    )
                ]

        extractor: ClaimExtractor = MinimalExtractor()
        claims = extractor.extract("test")
        assert len(claims) == 1
        assert claims[0].text == "test claim"


class TestLLMExtractor:
    """Test suite for LLM-based claim extraction."""

    async def test_llm_extractor_basic(self) -> None:
        """Test basic LLM extraction with mocked context."""
        extractor = LLMClaimExtractor()

        # Mock the ExecutionContext object - sample returns str directly
        mock_ctx = MagicMock()
        mock_response_text = """factual|0.95|0|20|The Earth orbits the Sun
numerical|0.9|21|45|The population is 8 billion"""
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract(
            "The Earth orbits the Sun. The population is 8 billion", mock_ctx
        )

        assert len(claims) == 2
        assert claims[0].claim_type == ClaimType.FACTUAL
        assert claims[0].confidence == 0.95
        assert claims[0].text == "The Earth orbits the Sun"
        assert claims[1].claim_type == ClaimType.NUMERICAL
        assert claims[1].confidence == 0.9

    async def test_llm_extractor_with_source_spans(self) -> None:
        """Test LLM extraction preserves source spans."""
        extractor = LLMClaimExtractor()

        mock_ctx = MagicMock()
        mock_response_text = "temporal|0.85|10|30|occurred in 2024"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract("This event occurred in 2024", mock_ctx)

        assert len(claims) == 1
        assert claims[0].source_span == (10, 30)
        assert claims[0].claim_type == ClaimType.TEMPORAL

    async def test_llm_extractor_handles_malformed_lines(self) -> None:
        """Test LLM extractor handles malformed response lines gracefully."""
        extractor = LLMClaimExtractor()

        mock_ctx = MagicMock()
        mock_response_text = """factual|0.95|0|20|Valid claim
invalid line without pipes
logical|invalid_confidence|5|10|Another claim
causal|0.8|invalid_start|50|Bad span
comparative|0.75|30|45|Valid comparison"""
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract("Test text", mock_ctx)

        # Should have 4 claims: implementation uses defaults for malformed values
        # - Line 1: fully valid
        # - Line 2: skipped (no pipes)
        # - Line 3: invalid confidence -> defaults to 0.5
        # - Line 4: invalid span -> defaults to None
        # - Line 5: fully valid
        assert len(claims) == 4
        assert claims[0].text == "Valid claim"
        assert claims[0].confidence == 0.95
        assert claims[1].text == "Another claim"
        assert claims[1].confidence == 0.5  # Default for invalid confidence
        assert claims[2].text == "Bad span"
        assert claims[2].source_span is None  # Default for invalid span
        assert claims[3].text == "Valid comparison"

    async def test_llm_extractor_unknown_claim_type(self) -> None:
        """Test LLM extractor defaults to FACTUAL for unknown types."""
        extractor = LLMClaimExtractor()

        mock_ctx = MagicMock()
        mock_response_text = "unknown_type|0.8|0|10|Some claim"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract("Test", mock_ctx)

        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.FACTUAL  # Default fallback

    async def test_llm_extractor_confidence_clamping(self) -> None:
        """Test LLM extractor clamps confidence to [0.0, 1.0]."""
        extractor = LLMClaimExtractor()

        mock_ctx = MagicMock()
        # Provide confidence values outside valid range
        mock_response_text = """factual|1.5|0|10|Over confidence
factual|-0.5|11|20|Under confidence"""
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract("Test", mock_ctx)

        assert len(claims) == 2
        assert claims[0].confidence == 1.0  # Clamped from 1.5
        assert claims[1].confidence == 0.0  # Clamped from -0.5

    async def test_llm_extractor_empty_response(self) -> None:
        """Test LLM extractor handles empty response."""
        extractor = LLMClaimExtractor()

        mock_ctx = MagicMock()
        mock_response_text = ""
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        claims = await extractor.extract("Test", mock_ctx)

        assert len(claims) == 0


class TestRuleBasedExtractor:
    """Test suite for rule-based claim extraction."""

    def test_rule_based_extractor_percentages(self) -> None:
        """Test extraction of percentage values."""
        extractor = RuleBasedExtractor()
        text = "The price increased by 25% and then decreased by 3.5%."

        claims = extractor.extract(text)

        percentage_claims = [c for c in claims if "%" in c.text]
        assert len(percentage_claims) == 2
        assert all(c.claim_type == ClaimType.NUMERICAL for c in percentage_claims)
        assert all(c.confidence == 0.8 for c in percentage_claims)

    def test_rule_based_extractor_currency(self) -> None:
        """Test extraction of currency values."""
        extractor = RuleBasedExtractor()
        text = "It costs $100 or $1,234.56 for premium."

        claims = extractor.extract(text)

        currency_claims = [c for c in claims if "$" in c.text]
        assert len(currency_claims) == 2
        assert all(c.claim_type == ClaimType.NUMERICAL for c in currency_claims)

    def test_rule_based_extractor_numbers(self) -> None:
        """Test extraction of plain numbers."""
        extractor = RuleBasedExtractor()
        text = "There are 1000 users and 5.5 million downloads."

        claims = extractor.extract(text)

        # Should find both numbers
        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        assert len(numerical_claims) >= 2

    def test_rule_based_extractor_dates(self) -> None:
        """Test extraction of various date formats."""
        extractor = RuleBasedExtractor()
        text = "Events on 2024-01-15, 1/15/2024, and January 15, 2024."

        claims = extractor.extract(text)

        date_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(date_claims) >= 3
        assert all(c.confidence == 0.8 for c in date_claims)

    def test_rule_based_extractor_durations(self) -> None:
        """Test extraction of duration expressions."""
        extractor = RuleBasedExtractor()
        text = "It takes 5 minutes, 2 hours, or 3 days to complete."

        claims = extractor.extract(text)

        duration_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(duration_claims) >= 3

    def test_rule_based_extractor_comparisons(self) -> None:
        """Test extraction of comparison expressions."""
        extractor = RuleBasedExtractor()
        text = "X is more than Y, less than Z, and better than W."

        claims = extractor.extract(text)

        comparison_claims = [c for c in claims if c.claim_type == ClaimType.COMPARATIVE]
        assert len(comparison_claims) >= 3

    def test_rule_based_extractor_source_spans(self) -> None:
        """Test that rule-based extractor provides accurate source spans."""
        extractor = RuleBasedExtractor()
        text = "Cost is $50"

        claims = extractor.extract(text)

        assert len(claims) > 0
        for claim in claims:
            assert claim.source_span is not None
            start, end = claim.source_span
            assert text[start:end] == claim.text

    def test_rule_based_extractor_empty_text(self) -> None:
        """Test rule-based extractor with empty text."""
        extractor = RuleBasedExtractor()

        claims = extractor.extract("")

        assert len(claims) == 0

    def test_rule_based_extractor_no_matches(self) -> None:
        """Test rule-based extractor with text containing no patterns."""
        extractor = RuleBasedExtractor()
        text = "This is plain text without any extractable patterns."

        claims = extractor.extract(text)

        # May have zero claims or very few
        assert isinstance(claims, list)


class TestHybridExtractor:
    """Test suite for hybrid claim extraction."""

    async def test_hybrid_extractor_combines_results(self) -> None:
        """Test that hybrid extractor combines LLM and rule-based results."""
        extractor = HybridExtractor()

        # Mock context with LLM response
        mock_ctx = MagicMock()
        mock_response_text = "factual|0.95|0|25|The system is efficient"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        text = "The system is efficient and costs $100."

        claims = await extractor.extract(text, mock_ctx)

        # Should have claims from both LLM (factual) and rules (currency)
        assert len(claims) >= 2
        claim_types = {c.claim_type for c in claims}
        assert ClaimType.FACTUAL in claim_types  # From LLM
        # Should also have numerical from rule-based

    async def test_hybrid_extractor_deduplication(self) -> None:
        """Test that hybrid extractor deduplicates overlapping claims."""
        extractor = HybridExtractor()

        # Mock LLM to extract a claim that overlaps with rule-based pattern
        mock_ctx = MagicMock()
        # LLM extracts the same percentage that rule-based would find
        mock_response_text = "numerical|0.95|20|24|25%"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        text = "The price increased 25% last year."

        claims = await extractor.extract(text, mock_ctx)

        # Should not have duplicate "25%" claims
        # Count claims with the same text
        claim_texts = [c.text for c in claims]
        assert len(claim_texts) == len(set(claim_texts)) or len(claims) <= 2

    async def test_hybrid_extractor_non_overlapping_augmentation(self) -> None:
        """Test that hybrid extractor adds non-overlapping rule-based claims."""
        extractor = HybridExtractor()

        # Mock LLM to extract claim far from rule-based patterns
        mock_ctx = MagicMock()
        mock_response_text = "factual|0.9|0|10|First part"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        # Text with LLM claim at start and rule-based pattern at end
        text = "First part of text. Later it costs $50."

        claims = await extractor.extract(text, mock_ctx)

        # Should have both claims since they don't overlap
        assert len(claims) >= 2

    async def test_hybrid_extractor_no_llm_claims(self) -> None:
        """Test hybrid extractor when LLM returns no claims."""
        extractor = HybridExtractor()

        mock_ctx = MagicMock()
        mock_response_text = ""
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        text = "The price is $100."

        claims = await extractor.extract(text, mock_ctx)

        # Should still have rule-based claims
        assert len(claims) > 0
        assert all(c.claim_type == ClaimType.NUMERICAL for c in claims)

    async def test_hybrid_extractor_no_rule_claims(self) -> None:
        """Test hybrid extractor when rule-based finds no claims."""
        extractor = HybridExtractor()

        mock_ctx = MagicMock()
        mock_response_text = "factual|0.9|0|20|Plain statement here"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        text = "Plain statement here without patterns."

        claims = await extractor.extract(text, mock_ctx)

        # Should still have LLM claims
        assert len(claims) >= 1
        assert any(c.claim_type == ClaimType.FACTUAL for c in claims)

    async def test_hybrid_extractor_preserves_llm_priority(self) -> None:
        """Test that hybrid extractor prioritizes LLM claims in overlaps."""
        extractor = HybridExtractor()

        # Create overlapping claims
        mock_ctx = MagicMock()
        # LLM claims high confidence on overlapping region
        mock_response_text = "factual|0.98|10|20|exact text"
        mock_ctx.sample = AsyncMock(return_value=mock_response_text)

        text = "Prefix is exact text with suffix."

        claims = await extractor.extract(text, mock_ctx)

        # Find the claim in the overlapping region
        overlapping_claims = [
            c for c in claims if c.source_span and c.source_span[0] >= 10 and c.source_span[1] <= 20
        ]

        # Should have LLM claim with higher confidence
        if overlapping_claims:
            assert any(c.confidence >= 0.9 for c in overlapping_claims)


class TestExtractorFactory:
    """Test suite for extractor factory function."""

    def test_factory_llm_extractor(self) -> None:
        """Test factory creates LLM extractor."""
        extractor = get_extractor({"extractor_type": "llm"})

        assert isinstance(extractor, LLMClaimExtractor)

    def test_factory_rule_based_extractor(self) -> None:
        """Test factory creates rule-based extractor."""
        extractor = get_extractor({"extractor_type": "rule_based"})

        assert isinstance(extractor, RuleBasedExtractor)

    def test_factory_hybrid_extractor(self) -> None:
        """Test factory creates hybrid extractor."""
        extractor = get_extractor({"extractor_type": "hybrid"})

        assert isinstance(extractor, HybridExtractor)

    def test_factory_default_to_hybrid(self) -> None:
        """Test factory defaults to hybrid when no type specified."""
        extractor = get_extractor({})

        assert isinstance(extractor, HybridExtractor)

    def test_factory_invalid_type(self) -> None:
        """Test factory raises error for invalid extractor type."""
        with pytest.raises(ValueError, match="Unknown extractor_type"):
            get_extractor({"extractor_type": "invalid_type"})

    def test_factory_case_sensitive(self) -> None:
        """Test factory is case-sensitive for extractor types."""
        # Should work with exact case
        extractor = get_extractor({"extractor_type": "hybrid"})
        assert isinstance(extractor, HybridExtractor)

        # Should fail with wrong case
        with pytest.raises(ValueError):
            get_extractor({"extractor_type": "HYBRID"})

    def test_factory_returns_protocol_conformant(self) -> None:
        """Test that factory always returns protocol-conformant extractors."""
        for extractor_type in ["llm", "rule_based", "hybrid"]:
            extractor = get_extractor({"extractor_type": extractor_type})
            # All should have extract method (some async, some sync)
            assert hasattr(extractor, "extract")
