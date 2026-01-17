"""Claim extraction module for verification.

This module provides different strategies for extracting verifiable claims
from reasoning text, including LLM-based, rule-based, and hybrid approaches.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext

from reasoning_mcp.models.verification import Claim, ClaimType


class ClaimExtractor(Protocol):
    """Protocol for claim extraction from text.

    This protocol defines the interface that all claim extractors must implement,
    allowing for different extraction strategies (LLM-based, rule-based, hybrid).

    Examples:
        Implementing a custom extractor:
        >>> class MyExtractor:
        ...     def extract(self, text: str) -> list[Claim]:
        ...         return []
        >>> extractor: ClaimExtractor = MyExtractor()
    """

    def extract(self, text: str) -> list[Claim]:
        """Extract claims from the given text.

        Args:
            text: The text to extract claims from

        Returns:
            List of extracted Claim objects
        """
        ...


class LLMClaimExtractor:
    """LLM-based claim extractor.

    Uses an LLM (via ExecutionContext.sample) to identify and extract claims from text.
    This extractor is good at understanding context and extracting nuanced claims
    that rule-based approaches might miss.

    Examples:
        >>> extractor = LLMClaimExtractor()
        >>> # Would be used with: await extractor.extract(text, execution_ctx)
    """

    async def extract(self, text: str, ctx: ExecutionContext) -> list[Claim]:
        """Extract claims from text using LLM analysis.

        Args:
            text: The text to extract claims from
            ctx: ExecutionContext for LLM sampling

        Returns:
            List of extracted Claim objects with unique claim_ids

        Examples:
            >>> # This would be used in an async context with a real Context object
            >>> # claims = await extractor.extract("The Earth is round.", ctx)
            >>> # assert len(claims) > 0
            >>> pass
        """
        prompt = f"""Analyze the following text and identify all verifiable claims.
For each claim, specify:
1. The claim text (exact quote or paraphrase)
2. The claim type (factual, numerical, temporal, causal, comparative, logical)
3. Confidence in the extraction (0.0 to 1.0)
4. Character span in the original text (start, end positions)

Text to analyze:
{text}

Return each claim on a separate line in this format:
TYPE|CONFIDENCE|START|END|CLAIM_TEXT

Example:
factual|0.95|0|20|The Earth orbits the Sun
numerical|0.9|21|45|The population is 8 billion

Focus on identifying:
- Factual claims about the world
- Numerical quantities and calculations
- Temporal relationships and sequences
- Cause-and-effect relationships
- Comparisons between entities
- Logical deductions and inferences
"""

        # Sample the LLM for claim extraction
        response = await ctx.sample(prompt)
        # When result_type is not provided, sample returns str
        response_text = response if isinstance(response, str) else str(response)

        # Parse the LLM response to create Claim objects
        claims: list[Claim] = []
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue

            try:
                parts = line.split("|", 4)
                if len(parts) != 5:
                    continue

                claim_type_str, confidence_str, start_str, end_str, claim_text = parts

                # Parse claim type
                try:
                    claim_type = ClaimType(claim_type_str.strip().lower())
                except ValueError:
                    # Default to factual if unknown type
                    claim_type = ClaimType.FACTUAL

                # Parse confidence
                try:
                    confidence = float(confidence_str.strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
                except ValueError:
                    confidence = 0.5  # Default confidence

                # Parse source span
                try:
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    source_span = (start, end) if start >= 0 and end > start else None
                except ValueError:
                    source_span = None

                # Create claim with unique ID
                claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    text=claim_text.strip(),
                    claim_type=claim_type,
                    source_span=source_span,
                    confidence=confidence,
                )
                claims.append(claim)

            except Exception:
                # Skip malformed lines
                continue

        return claims


class RuleBasedExtractor:
    """Rule-based claim extractor using regex patterns.

    Extracts claims based on predefined patterns for numbers, dates, comparisons,
    and other structured information. Fast and deterministic but may miss nuanced claims.

    Examples:
        >>> extractor = RuleBasedExtractor()
        >>> claims = extractor.extract("The price increased by 25% to $100.")
        >>> assert len(claims) > 0
        >>> assert any(c.claim_type == ClaimType.NUMERICAL for c in claims)
    """

    # Regex patterns for different claim types
    PATTERNS = {
        ClaimType.NUMERICAL: [
            r"\d+(?:\.\d+)?%",  # Percentages: 25%, 3.14%
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Currency: $100, $1,000.50
            r"\d+(?:,\d{3})*(?:\.\d+)?",  # Numbers: 1000, 1,234.56
        ],
        ClaimType.TEMPORAL: [
            r"\d{4}-\d{2}-\d{2}",  # ISO dates: 2024-01-15
            r"\d{1,2}/\d{1,2}/\d{4}",  # US dates: 1/15/2024
            # Month Day, Year format
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"\d+\s+(?:second|minute|hour|day|week|month|year)s?",  # Durations
        ],
        ClaimType.COMPARATIVE: [
            r"(?:more|less|greater|smaller|higher|lower|better|worse)\s+than",
            r"(?:superior|inferior)\s+to",
            r"(?:increase|decrease)[ds]?\s+(?:by|to)",
            r"compared\s+(?:to|with)",
        ],
    }

    def extract(self, text: str) -> list[Claim]:
        """Extract claims from text using regex patterns.

        Args:
            text: The text to extract claims from

        Returns:
            List of extracted Claim objects

        Examples:
            >>> extractor = RuleBasedExtractor()
            >>> claims = extractor.extract("It costs $50 and takes 2 hours.")
            >>> assert len(claims) >= 2  # Should find currency and duration
            >>> numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
            >>> assert len(numerical_claims) > 0
        """
        claims: list[Claim] = []

        for claim_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    claim = Claim(
                        claim_id=str(uuid.uuid4()),
                        text=match.group(),
                        claim_type=claim_type,
                        source_span=(match.start(), match.end()),
                        confidence=0.8,  # Rule-based has good confidence but not perfect
                    )
                    claims.append(claim)

        return claims


class HybridExtractor:
    """Hybrid claim extractor combining LLM and rule-based approaches.

    Uses both LLM-based extraction for nuanced claims and rule-based extraction
    for structured patterns, then deduplicates overlapping claims to provide
    comprehensive coverage.

    Examples:
        >>> extractor = HybridExtractor()
        >>> # Would be used with: await extractor.extract(text, execution_ctx)
    """

    def __init__(self) -> None:
        """Initialize the hybrid extractor with both sub-extractors."""
        self.rule_based = RuleBasedExtractor()

    async def extract(self, text: str, ctx: ExecutionContext) -> list[Claim]:
        """Extract claims using both LLM and rule-based approaches.

        Combines results from both extractors and deduplicates overlapping claims,
        prioritizing LLM claims but augmenting with rule-based for numerical/temporal
        patterns that might be missed.

        Args:
            text: The text to extract claims from
            ctx: ExecutionContext for LLM sampling

        Returns:
            Deduplicated list of extracted Claim objects

        Examples:
            >>> # This would be used in an async context with a real Context object
            >>> # extractor = HybridExtractor()
            >>> # claims = await extractor.extract("Text with claims.", ctx)
            >>> # assert len(claims) > 0
            >>> pass
        """
        # Get claims from both extractors
        llm_extractor = LLMClaimExtractor()
        llm_claims = await llm_extractor.extract(text, ctx)
        rule_claims = self.rule_based.extract(text)

        # Deduplicate: prioritize LLM claims but add non-overlapping rule-based claims
        all_claims = list(llm_claims)  # Start with all LLM claims

        for rule_claim in rule_claims:
            # Check if this rule-based claim overlaps with any LLM claim
            overlaps = False
            if rule_claim.source_span:
                rule_start, rule_end = rule_claim.source_span
                for llm_claim in llm_claims:
                    if llm_claim.source_span:
                        llm_start, llm_end = llm_claim.source_span
                        # Check for overlap
                        if not (rule_end <= llm_start or rule_start >= llm_end):
                            overlaps = True
                            break

            # Add rule-based claim if it doesn't overlap
            if not overlaps:
                all_claims.append(rule_claim)

        return all_claims


def get_extractor(config: dict[str, Any]) -> ClaimExtractor:
    """Factory function to get appropriate claim extractor based on config.

    Args:
        config: Configuration dictionary with "extractor_type" key

    Returns:
        Appropriate ClaimExtractor instance

    Raises:
        ValueError: If extractor_type is not recognized

    Examples:
        >>> extractor = get_extractor({"extractor_type": "rule_based"})
        >>> isinstance(extractor, RuleBasedExtractor)
        True

        >>> extractor = get_extractor({"extractor_type": "hybrid"})
        >>> isinstance(extractor, HybridExtractor)
        True

        >>> extractor = get_extractor({})  # Default to hybrid
        >>> isinstance(extractor, HybridExtractor)
        True
    """
    extractor_type = config.get("extractor_type", "hybrid")

    if extractor_type == "llm":
        return LLMClaimExtractor()  # type: ignore[return-value]
    elif extractor_type == "rule_based":
        return RuleBasedExtractor()
    elif extractor_type == "hybrid":
        return HybridExtractor()  # type: ignore[return-value]
    else:
        raise ValueError(
            f"Unknown extractor_type: {extractor_type}. "
            f"Must be one of: 'llm', 'rule_based', 'hybrid'"
        )
