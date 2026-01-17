"""Fact checker implementations for claim verification.

This module provides various strategies for checking the validity of claims
extracted from reasoning outputs:

- SelfConsistencyChecker: Verifies claims using multiple LLM samples
- LogicalConsistencyChecker: Checks logical validity and coherence
- NumericalChecker: Validates numerical claims through computation
- ExternalSourceChecker: Placeholder for external source verification

Each checker implements the FactChecker protocol and returns VerificationResult
with supporting evidence.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from reasoning_mcp.models.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext


class FactChecker(Protocol):
    """Protocol for fact checking claims.

    All fact checkers must implement the check method which takes a claim
    and execution context and returns a verification result with evidence.

    Examples:
        >>> # Any class implementing this protocol can be used as a checker
        >>> class MyChecker:
        ...     async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        ...         # Implementation here
        ...         pass
    """

    async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check a claim and return verification result.

        Args:
            claim: The claim to verify
            ctx: Execution context with session, registry, and optional LLM access

        Returns:
            Verification result with status, confidence, evidence, and reasoning
        """
        ...


class SelfConsistencyChecker:
    """Checks claim consistency across multiple LLM responses.

    This checker samples the LLM multiple times with the same query about a claim
    and measures agreement across responses. High agreement indicates the claim
    is likely correct, while disagreement suggests uncertainty or incorrectness.

    Agreement thresholds:
        - >80% agreement: VERIFIED
        - 50-80% agreement: UNCERTAIN
        - <50% agreement: REFUTED

    Attributes:
        num_samples: Number of LLM samples to generate (default: 5)
        temperature: Sampling temperature for diversity (default: 0.7)
    """

    def __init__(self, num_samples: int = 5, temperature: float = 0.7) -> None:
        """Initialize the self-consistency checker.

        Args:
            num_samples: Number of LLM samples to generate (3-10 recommended)
            temperature: Sampling temperature (0.0-2.0, higher = more diverse)
        """
        self.num_samples = max(3, min(10, num_samples))  # Clamp to 3-10
        self.temperature = max(0.0, min(2.0, temperature))  # Clamp to 0.0-2.0

    async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check claim using self-consistency across multiple LLM samples.

        Args:
            claim: The claim to verify
            ctx: Execution context with LLM sampling capability

        Returns:
            Verification result with evidence from each sample

        Examples:
            >>> checker = SelfConsistencyChecker(num_samples=5)
            >>> result = await checker.check(claim, ctx)
            >>> assert result.status in [
            ...     VerificationStatus.VERIFIED,
            ...     VerificationStatus.UNCERTAIN,
            ...     VerificationStatus.REFUTED,
            ...     VerificationStatus.UNVERIFIABLE
            ... ]
        """
        if not ctx.can_sample:
            # Cannot verify without LLM access
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning="Self-consistency checking requires LLM sampling, which is not available",
            )

        # Prompt the LLM to evaluate the claim
        prompt = f"""Evaluate the following claim and respond with either "AGREE" if the claim is \
correct, or "DISAGREE" if it is incorrect.

Claim: {claim.text}

Your response (AGREE or DISAGREE):"""

        # Collect multiple samples
        responses: list[str] = []
        evidence_list: list[Evidence] = []

        for i in range(self.num_samples):
            try:
                response = await ctx.sample(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=256,
                )
                response_str = str(response).strip().upper()
                responses.append(response_str)

                # Create evidence from this sample
                evidence = Evidence(
                    evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                    source=EvidenceSource.INTERNAL,
                    content=f"Sample {i + 1}/{self.num_samples}: {response_str}",
                    relevance_score=1.0,
                )
                evidence_list.append(evidence)

            except Exception as e:
                # If sampling fails, record the error but continue
                evidence = Evidence(
                    evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                    source=EvidenceSource.INTERNAL,
                    content=f"Sample {i + 1}/{self.num_samples} failed: {str(e)}",
                    relevance_score=0.0,
                )
                evidence_list.append(evidence)

        # Count agreements
        if not responses:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=evidence_list,
                reasoning="All LLM samples failed to produce responses",
            )

        # Count AGREE vs DISAGREE, being precise about what we're looking for
        agree_count = sum(1 for r in responses if "AGREE" in r and "DISAGREE" not in r)
        total_count = len(responses)
        agreement_ratio = agree_count / total_count if total_count > 0 else 0.0

        # Determine status based on agreement ratio
        if agreement_ratio >= 0.8:
            status = VerificationStatus.VERIFIED
            confidence = agreement_ratio
            reasoning = (
                f"{agree_count}/{total_count} samples agree with the claim (>=80% agreement)"
            )
        elif agreement_ratio >= 0.5:
            status = VerificationStatus.UNCERTAIN
            confidence = 0.5
            reasoning = (
                f"{agree_count}/{total_count} samples agree with the claim (50-80% agreement)"
            )
        else:
            status = VerificationStatus.REFUTED
            confidence = 1.0 - agreement_ratio
            reasoning = (
                f"Only {agree_count}/{total_count} samples agree with the claim (<50% agreement)"
            )

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence_list,
            reasoning=reasoning,
        )


class LogicalConsistencyChecker:
    """Checks logical consistency within a reasoning chain.

    This checker prompts the LLM to analyze whether a claim follows logically
    from given premises, checking for contradictions, non-sequiturs, and
    invalid inferences.
    """

    async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check claim for logical consistency.

        Args:
            claim: The claim to verify
            ctx: Execution context with LLM sampling capability

        Returns:
            Verification result with logical analysis

        Examples:
            >>> checker = LogicalConsistencyChecker()
            >>> result = await checker.check(claim, ctx)
            >>> assert result.status in [
            ...     VerificationStatus.VERIFIED,
            ...     VerificationStatus.REFUTED,
            ...     VerificationStatus.UNVERIFIABLE
            ... ]
        """
        if not ctx.can_sample:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=(
                    "Logical consistency checking requires LLM sampling, which is not available"
                ),
            )

        # Get the reasoning context from input data if available
        context_text = ctx.input_data.get("reasoning_text", "")
        if not context_text:
            # Use the claim text itself as minimal context
            context_text = claim.text

        prompt = f"""Analyze the logical consistency of the following claim within its \
reasoning context.

Context: {context_text}

Claim to evaluate: {claim.text}

Check for:
1. Does the claim follow logically from the context?
2. Are there any contradictions?
3. Are there any non-sequiturs (conclusions that don't follow)?
4. Are there any invalid logical inferences?

Respond with:
- "VALID" if the claim is logically consistent
- "INVALID" if there are logical errors
- "UNCERTAIN" if you cannot determine

Then explain your reasoning in 1-2 sentences.

Format your response as:
[VALID/INVALID/UNCERTAIN]: <explanation>"""

        try:
            response = await ctx.sample(
                prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=512,
            )
            response_str = str(response).strip()

            # Parse response
            evidence = Evidence(
                evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                source=EvidenceSource.INTERNAL,
                content=response_str,
                relevance_score=1.0,
            )

            # Determine status from response - check for keywords in priority order
            response_upper = response_str.upper()
            if "UNCERTAIN" in response_upper:
                # Check for UNCERTAIN first to avoid confusion with VALID/INVALID
                status = VerificationStatus.UNCERTAIN
                confidence = 0.5
            elif "INVALID" in response_upper:
                status = VerificationStatus.REFUTED
                confidence = 0.9
            elif "VALID" in response_upper:
                status = VerificationStatus.VERIFIED
                confidence = 0.9
            else:
                status = VerificationStatus.UNCERTAIN
                confidence = 0.3

            return VerificationResult(
                claim=claim,
                status=status,
                confidence=confidence,
                evidence=[evidence],
                reasoning=response_str,
            )

        except Exception as e:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=f"Logical consistency check failed: {str(e)}",
            )


class NumericalChecker:
    """Validates numerical claims through computation.

    This checker extracts numbers and operators from NUMERICAL type claims,
    performs the computation, and compares the result with the claimed result.

    For simple arithmetic, it uses safe evaluation. For complex mathematics,
    it may prompt the LLM for assistance.
    """

    async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check numerical claim by computing the result.

        Args:
            claim: The claim to verify (should be NUMERICAL type)
            ctx: Execution context

        Returns:
            Verification result with computation trace

        Examples:
            >>> checker = NumericalChecker()
            >>> claim = Claim(
            ...     claim_id="c1",
            ...     text="2 + 2 = 4",
            ...     claim_type=ClaimType.NUMERICAL,
            ...     confidence=0.9
            ... )
            >>> result = await checker.check(claim, ctx)
            >>> assert result.status == VerificationStatus.VERIFIED
        """
        if claim.claim_type != ClaimType.NUMERICAL:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=(
                    f"NumericalChecker only handles NUMERICAL claims, got {claim.claim_type.value}"
                ),
            )

        # Try to extract and verify numerical computation
        try:
            # Look for equation pattern: expression = result
            equation_pattern = r"^(.+?)\s*=\s*(.+?)$"
            match = re.search(equation_pattern, claim.text)

            if not match:
                # Try to use LLM to extract the computation if available
                if ctx.can_sample:
                    return await self._check_with_llm(claim, ctx)
                else:
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.UNVERIFIABLE,
                        confidence=0.0,
                        evidence=[],
                        reasoning=(
                            "Could not parse numerical equation format. "
                            "Expected 'expression = result'"
                        ),
                    )

            expression = match.group(1).strip()
            claimed_result = match.group(2).strip()

            # Try to evaluate the expression safely
            try:
                computed_result = self._safe_eval(expression)
                claimed_value = self._safe_eval(claimed_result)

                # Compare results (with small tolerance for floating point)
                tolerance = 1e-6
                if abs(computed_result - claimed_value) < tolerance:
                    status = VerificationStatus.VERIFIED
                    confidence = 1.0
                    reasoning = f"Computation verified: {expression} = {computed_result}"
                else:
                    status = VerificationStatus.REFUTED
                    confidence = 1.0
                    reasoning = (
                        f"Computation error: {expression} = {computed_result}, "
                        f"but claim states {claimed_value}"
                    )

                evidence = Evidence(
                    evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                    source=EvidenceSource.COMPUTED,
                    content=f"Computed {expression} = {computed_result}; Claimed = {claimed_value}",
                    relevance_score=1.0,
                )

                return VerificationResult(
                    claim=claim,
                    status=status,
                    confidence=confidence,
                    evidence=[evidence],
                    reasoning=reasoning,
                )

            except Exception as eval_error:
                # If evaluation fails, try LLM if available
                if ctx.can_sample:
                    return await self._check_with_llm(claim, ctx)
                else:
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.UNVERIFIABLE,
                        confidence=0.0,
                        evidence=[],
                        reasoning=f"Could not evaluate expression: {str(eval_error)}",
                    )

        except Exception as e:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=f"Numerical verification failed: {str(e)}",
            )

    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate a mathematical expression.

        Args:
            expr: Expression string to evaluate

        Returns:
            Computed result as float

        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression is malformed
        """
        # Remove whitespace
        expr = expr.strip()

        # Create a safe namespace with only math operations
        safe_dict: dict[str, Any] = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            "round": round,
        }

        # Check for unsafe patterns
        unsafe_patterns = [
            "__",  # Dunder methods
            "import",
            "exec",
            "eval",
            "compile",
            "open",
            "file",
        ]

        expr_lower = expr.lower()
        for pattern in unsafe_patterns:
            if pattern in expr_lower:
                raise ValueError(f"Unsafe pattern detected: {pattern}")

        # Evaluate expression
        result = eval(expr, safe_dict, {})  # noqa: S307

        # Convert to float
        return float(result)

    async def _check_with_llm(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check numerical claim using LLM when direct computation fails.

        Args:
            claim: The numerical claim to verify
            ctx: Execution context with LLM access

        Returns:
            Verification result from LLM analysis
        """
        prompt = f"""Verify the following numerical claim. Compute the result and check if \
it's correct.

Claim: {claim.text}

Respond with:
- "CORRECT" if the calculation is correct
- "INCORRECT" if the calculation is wrong
- "UNCLEAR" if you cannot verify

Then show your computation and explanation.

Format: [CORRECT/INCORRECT/UNCLEAR]: <computation and explanation>"""

        try:
            response = await ctx.sample(
                prompt,
                temperature=0.1,  # Very low temperature for numerical accuracy
                max_tokens=512,
            )
            response_str = str(response).strip()

            evidence = Evidence(
                evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                source=EvidenceSource.INTERNAL,
                content=response_str,
                relevance_score=1.0,
            )

            # Determine status from response - check for keywords in priority order
            response_upper = response_str.upper()
            if "UNCLEAR" in response_upper or "UNCERTAIN" in response_upper:
                status = VerificationStatus.UNCERTAIN
                confidence = 0.5
            elif "INCORRECT" in response_upper:
                status = VerificationStatus.REFUTED
                confidence = 0.9
            elif "CORRECT" in response_upper:
                status = VerificationStatus.VERIFIED
                confidence = 0.9
            else:
                status = VerificationStatus.UNCERTAIN
                confidence = 0.5

            return VerificationResult(
                claim=claim,
                status=status,
                confidence=confidence,
                evidence=[evidence],
                reasoning=response_str,
            )

        except Exception as e:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning=f"LLM numerical verification failed: {str(e)}",
            )


class ExternalSourceChecker:
    """Checks claims against external sources.

    This is a placeholder implementation designed for future integration
    with external search APIs, knowledge bases, or fact-checking services.

    Currently returns UNVERIFIABLE with an explanation that external
    verification is not yet implemented.
    """

    async def check(self, claim: Claim, ctx: ExecutionContext) -> VerificationResult:
        """Check claim against external sources (placeholder).

        Args:
            claim: The claim to verify
            ctx: Execution context

        Returns:
            Verification result indicating external verification is not available

        Examples:
            >>> checker = ExternalSourceChecker()
            >>> result = await checker.check(claim, ctx)
            >>> assert result.status == VerificationStatus.UNVERIFIABLE
        """
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            reasoning=(
                "External source verification is not yet implemented. "
                "Future integration will support checking claims against "
                "search engines, knowledge bases, and fact-checking APIs."
            ),
        )
