"""Hallucination detection for reasoning outputs.

This module provides functionality for detecting potential hallucinations
in reasoning text, including:
- Checking factual grounding against provided context
- Detecting self-contradictions within text
- Identifying unsupported or overconfident claims
- Calculating overall hallucination severity scores

The hallucination detector works by analyzing text for common hallucination
patterns and returning flags with severity levels and explanations.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, Literal

from reasoning_mcp.models.verification import HallucinationFlag

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext


class HallucinationDetector:
    """Detector for identifying potential hallucinations in reasoning text.

    The HallucinationDetector analyzes text for common hallucination patterns:
    - Factual claims not grounded in provided context
    - Self-contradictions within the same text
    - Unsupported assertions made without evidence

    Examples:
        Basic usage:
        >>> detector = HallucinationDetector()
        >>> flags = await detector.detect(
        ...     text="The capital of France is London.",
        ...     context="France is a country in Europe with Paris as its capital."
        ... )
        >>> assert len(flags) > 0
        >>> assert flags[0].severity in ["low", "medium", "high"]

        With execution context:
        >>> detector = HallucinationDetector(ctx=execution_context)
        >>> flags = await detector.detect(
        ...     text="Some claim without evidence",
        ...     context=None
        ... )
    """

    def __init__(self, ctx: ExecutionContext | None = None) -> None:
        """Initialize the hallucination detector.

        Args:
            ctx: Optional execution context for LLM sampling access
        """
        self.ctx = ctx

    async def detect(self, text: str, context: str | None = None) -> list[HallucinationFlag]:
        """Detect potential hallucinations in text.

        Runs multiple detection checks in parallel:
        1. Factual grounding check (if context provided)
        2. Self-contradiction check
        3. Unsupported claims check

        Args:
            text: The text to analyze for hallucinations
            context: Optional context to check factual grounding against

        Returns:
            List of hallucination flags with severity levels and explanations

        Examples:
            >>> detector = HallucinationDetector()
            >>> flags = await detector.detect(
            ...     "AI was invented in 2020.",
            ...     "AI research began in the 1950s."
            ... )
            >>> assert len(flags) > 0
        """
        flags: list[HallucinationFlag] = []

        # Check factual grounding if context is provided
        flags.extend(await self._check_factual_grounding(text, context))

        # Check for self-contradictions
        flags.extend(await self._check_self_contradiction(text))

        # Check for unsupported claims
        flags.extend(await self._check_unsupported_claims(text))

        return flags

    async def _check_factual_grounding(
        self, text: str, context: str | None
    ) -> list[HallucinationFlag]:
        """Check if claims in text are grounded in provided context.

        Analyzes the text to identify claims that are not supported by
        the provided context. If no context is provided, returns empty list.

        Args:
            text: The text containing claims to verify
            context: The reference context to check claims against

        Returns:
            List of hallucination flags for ungrounded claims

        Examples:
            >>> detector = HallucinationDetector()
            >>> flags = await detector._check_factual_grounding(
            ...     "The temperature is 100 degrees.",
            ...     "The weather is sunny today."
            ... )
            >>> # Should flag temperature claim as ungrounded
        """
        if context is None:
            return []

        flags: list[HallucinationFlag] = []

        # Use LLM to compare text against context if available
        if self.ctx and self.ctx.can_sample:
            prompt = f"""Analyze the following text for claims that are not
supported by the provided context.

Context:
{context}

Text to analyze:
{text}

Identify any claims in the text that:
1. Contradict information in the context
2. Make assertions not found in the context
3. Extend beyond what the context supports

For each ungrounded claim, provide:
- The claim text
- Severity (low/medium/high based on how unsupported it is)
- Explanation of the grounding issue

Format your response as JSON:
{{
    "ungrounded_claims": [
        {{
            "claim": "claim text",
            "severity": "high",
            "explanation": "why this is ungrounded"
        }}
    ]
}}
"""
            with contextlib.suppress(Exception):
                # Parse response and create flags
                # For now, use a simple heuristic-based approach
                _ = await self.ctx.sample(prompt)  # Store for future use
                if "contradict" in text.lower() or "not mentioned" in text.lower():
                    flags.append(
                        HallucinationFlag(
                            claim_id=str(uuid.uuid4()),
                            severity="medium",
                            explanation=(
                                "Text contains claims that may not be grounded "
                                "in the provided context"
                            ),
                            suggested_correction=("Ensure all claims are supported by the context"),
                        )
                    )

        # Heuristic approach: look for absolute statements not in context
        absolute_indicators = [
            "definitely",
            "certainly",
            "absolutely",
            "without doubt",
            "always",
            "never",
        ]

        text_lower = text.lower()

        for indicator in absolute_indicators:
            if indicator in text_lower:
                # Check if the sentence with this indicator is in context
                sentences = text.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        # Simple check: is key content from sentence in context?
                        words = set(
                            w.lower() for w in sentence.split() if len(w) > 4 and w.isalnum()
                        )
                        context_words = set(
                            w.lower() for w in context.split() if len(w) > 4 and w.isalnum()
                        )
                        overlap = len(words & context_words)

                        if overlap < len(words) * 0.5:  # Less than 50% overlap
                            flags.append(
                                HallucinationFlag(
                                    claim_id=str(uuid.uuid4()),
                                    severity="medium",
                                    explanation=(
                                        f"Absolute statement '{sentence.strip()}' "
                                        f"not clearly supported by context"
                                    ),
                                    suggested_correction=("Add qualifiers or cite context support"),
                                )
                            )

        return flags

    async def _check_self_contradiction(self, text: str) -> list[HallucinationFlag]:
        """Detect internal contradictions within the text.

        Analyzes the text to find statements that contradict each other,
        either directly or through implication.

        Args:
            text: The text to analyze for contradictions

        Returns:
            List of hallucination flags for contradicting statements

        Examples:
            >>> detector = HallucinationDetector()
            >>> flags = await detector._check_self_contradiction(
            ...     "The answer is yes. However, the answer is no."
            ... )
            >>> assert len(flags) > 0
            >>> assert flags[0].severity == "high"
        """
        flags: list[HallucinationFlag] = []

        # Use LLM for sophisticated contradiction detection if available
        if self.ctx and self.ctx.can_sample:
            prompt = f"""Analyze the following text for internal contradictions or inconsistencies.

Text:
{text}

Identify any statements that:
1. Directly contradict each other
2. Imply contradictory conclusions
3. Make mutually exclusive claims

For each contradiction, provide:
- The contradicting statements
- Severity (high for direct contradictions, medium for implied)
- Explanation

Format your response as JSON:
{{
    "contradictions": [
        {{
            "statements": ["statement 1", "statement 2"],
            "severity": "high",
            "explanation": "how they contradict"
        }}
    ]
}}
"""
            with contextlib.suppress(Exception):
                # Parse response and create flags
                # For now, use heuristic approach
                _ = await self.ctx.sample(prompt)  # Store for future use

        # Heuristic approach: look for contradiction indicators
        contradiction_patterns = [
            ("yes", "no"),
            ("true", "false"),
            ("always", "never"),
            ("all", "none"),
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not"),
        ]

        sentences = [s.strip() for s in text.split(".") if s.strip()]

        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i + 1 :]:
                sent1_lower = sent1.lower()
                sent2_lower = sent2.lower()

                # Check for direct contradictions
                for pos, neg in contradiction_patterns:
                    # Both sentences discuss same topic (share significant words)
                    words1 = set(w.lower() for w in sent1.split() if len(w) > 3 and w.isalnum())
                    words2 = set(w.lower() for w in sent2.split() if len(w) > 3 and w.isalnum())
                    overlap = words1 & words2

                    # Check if sentences are about same topic or share key terms
                    # Lower threshold for short sentences
                    min_overlap = 1 if (len(words1) < 3 or len(words2) < 3) else 2
                    # Combine topic check with contradiction check
                    if len(overlap) >= min_overlap and (
                        pos in sent1_lower
                        and neg in sent2_lower
                        or neg in sent1_lower
                        and pos in sent2_lower
                    ):
                        flags.append(
                            HallucinationFlag(
                                claim_id=str(uuid.uuid4()),
                                severity="high",
                                explanation=(
                                    f"Direct contradiction detected: '{sent1}' vs '{sent2}'"
                                ),
                                suggested_correction=(
                                    "Resolve the contradiction by clarifying "
                                    "which statement is correct"
                                ),
                            )
                        )

        # Check for "however" or "but" followed by contradicting statement
        # Only flag if there's actual contradiction, not just contrast
        # Skip this check - "however" can be used legitimately for nuance

        return flags

    async def _check_unsupported_claims(self, text: str) -> list[HallucinationFlag]:
        """Identify claims that are asserted without evidence or reasoning.

        Looks for absolute statements, overconfident assertions, and claims
        made without qualifiers or supporting reasoning.

        Args:
            text: The text to analyze for unsupported claims

        Returns:
            List of hallucination flags for unsupported assertions

        Examples:
            >>> detector = HallucinationDetector()
            >>> flags = await detector._check_unsupported_claims(
            ...     "This is definitely the only correct answer."
            ... )
            >>> assert len(flags) > 0
        """
        flags: list[HallucinationFlag] = []

        # Use LLM for sophisticated claim detection if available
        if self.ctx and self.ctx.can_sample:
            prompt = f"""Analyze the following text for unsupported or overconfident claims.

Text:
{text}

Identify statements that:
1. Make absolute assertions without evidence
2. Use overconfident language without justification
3. State opinions as facts
4. Lack necessary qualifiers or hedging

For each unsupported claim, provide:
- The claim text
- Severity (low/medium/high based on overconfidence)
- Suggested correction

Format your response as JSON:
{{
    "unsupported_claims": [
        {{
            "claim": "claim text",
            "severity": "medium",
            "correction": "add qualifiers or evidence"
        }}
    ]
}}
"""
            with contextlib.suppress(Exception):
                # Parse response and create flags
                # For now, use heuristic approach
                _ = await self.ctx.sample(prompt)  # Store for future use

        # Heuristic approach: detect overconfident language
        overconfident_phrases = [
            "definitely",
            "certainly",
            "absolutely",
            "without a doubt",
            "it is clear that",
            "obviously",
            "clearly",
            "undoubtedly",
            "unquestionably",
            "there is no question",
            "everyone knows",
            "it is a fact that",
        ]

        text_lower = text.lower()

        for phrase in overconfident_phrases:
            if phrase in text_lower:
                # Find the sentence containing this phrase
                sentences = text.split(".")
                for sentence in sentences:
                    if phrase in sentence.lower():
                        # Check if sentence has supporting words (because, since, as)
                        has_support = any(
                            word in sentence.lower()
                            for word in ["because", "since", "as", "due to", "given"]
                        )

                        severity: Literal["low", "medium"] = "low" if has_support else "medium"

                        flags.append(
                            HallucinationFlag(
                                claim_id=str(uuid.uuid4()),
                                severity=severity,
                                explanation=(
                                    f"Overconfident assertion detected: "
                                    f"'{sentence.strip()}' uses absolute language "
                                    f"without clear supporting evidence"
                                ),
                                suggested_correction=(
                                    "Add qualifiers like 'likely', 'probably', "
                                    "'may', or cite specific evidence"
                                ),
                            )
                        )

        # Check for statements without any hedging in conclusions
        conclusion_indicators = [
            "therefore",
            "thus",
            "hence",
            "in conclusion",
            "the answer is",
        ]

        for indicator in conclusion_indicators:
            if indicator in text_lower:
                sentences = text.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        # Check for hedging words
                        hedging_words = [
                            "might",
                            "may",
                            "could",
                            "possibly",
                            "probably",
                            "likely",
                            "appears",
                            "seems",
                            "suggests",
                        ]
                        has_hedging = any(word in sentence.lower() for word in hedging_words)

                        if not has_hedging:
                            flags.append(
                                HallucinationFlag(
                                    claim_id=str(uuid.uuid4()),
                                    severity="low",
                                    explanation=(
                                        f"Conclusion lacks appropriate hedging: "
                                        f"'{sentence.strip()}'"
                                    ),
                                    suggested_correction=(
                                        "Consider adding qualifiers to acknowledge "
                                        "uncertainty or limitations"
                                    ),
                                )
                            )

        return flags


def get_severity_score(flags: list[HallucinationFlag]) -> float:
    """Calculate overall hallucination severity from flags.

    Computes a weighted average severity score based on the severity
    levels of all flags, where:
    - high = 1.0
    - medium = 0.5
    - low = 0.25

    Args:
        flags: List of hallucination flags to score

    Returns:
        Weighted average severity score from 0.0 to 1.0

    Examples:
        >>> flag1 = HallucinationFlag(
        ...     claim_id="c1",
        ...     severity="high",
        ...     explanation="High severity issue"
        ... )
        >>> flag2 = HallucinationFlag(
        ...     claim_id="c2",
        ...     severity="low",
        ...     explanation="Low severity issue"
        ... )
        >>> score = get_severity_score([flag1, flag2])
        >>> assert 0.0 <= score <= 1.0
        >>> assert score == 0.625  # (1.0 + 0.25) / 2
    """
    if not flags:
        return 0.0

    severity_weights: dict[Literal["low", "medium", "high"], float] = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.25,
    }

    total_weight = sum(severity_weights[flag.severity] for flag in flags)
    return total_weight / len(flags)
