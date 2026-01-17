"""Synthesis voting strategy for ensemble decision-making.

This module implements a synthesis voting strategy that uses an LLM to
intelligently combine multiple reasoning results into a unified, coherent
answer. Unlike simple voting strategies, synthesis considers the content
and reasoning of each member to produce a more nuanced final result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models.ensemble import MemberResult


class SynthesisVoting:
    """Synthesis voting strategy.

    Uses an LLM to synthesize all member results into a unified answer.
    This strategy goes beyond simple voting to create an integrated response
    that combines the strengths of multiple reasoning approaches.

    The synthesis process:
    1. Collects all member results with their confidence scores
    2. Formats them into a structured prompt for the LLM
    3. Uses execution_context.sample() to generate a synthesized answer
    4. Returns the synthesis with average confidence

    When no execution context is available (e.g., during testing), falls back to
    a simple concatenation of unique answers.

    Examples:
        Create a synthesis voting strategy:
        >>> strategy = SynthesisVoting()
        >>> # Set context when available
        >>> strategy.set_context(ctx)

        Use with context for LLM-powered synthesis:
        >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
        >>> results = [
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="cot"),
        ...         result="The answer is 42 because...",
        ...         confidence=0.9,
        ...         execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="tot"),
        ...         result="After exploring paths, 42 is correct...",
        ...         confidence=0.95,
        ...         execution_time_ms=200
        ...     ),
        ... ]
        >>> answer, confidence, details = await strategy.aggregate_async(results)
        >>> assert "42" in answer
        >>> assert 0.0 <= confidence <= 1.0

        Use without context (fallback mode):
        >>> strategy = SynthesisVoting()
        >>> answer, confidence, details = strategy.aggregate(results)
        >>> assert details["fallback_mode"] is True
    """

    def __init__(self, execution_context: ExecutionContext | None = None) -> None:
        """Initialize synthesis voting strategy.

        Args:
            execution_context: Optional execution context for LLM access. Can be set later
                via set_context().
        """
        self._execution_context = execution_context

    def set_context(self, execution_context: ExecutionContext) -> None:
        """Set the execution context for LLM sampling.

        Args:
            execution_context: Execution context providing access to LLM via sample() method.

        Examples:
            >>> strategy = SynthesisVoting()
            >>> strategy.set_context(exec_ctx)
            >>> # Now strategy can use LLM for synthesis
        """
        self._execution_context = execution_context

    async def aggregate_async(
        self, results: list[MemberResult]
    ) -> tuple[str, float, dict[str, Any]]:
        """Synthesize results using LLM (async version).

        Uses execution_context.sample() to ask LLM to intelligently synthesize all member
        answers into a unified response that combines their insights.

        Args:
            results: List of member results to synthesize. Must not be empty.

        Returns:
            Tuple of (synthesized_answer, confidence, details) where:
            - synthesized_answer: LLM-generated synthesis of all results
            - confidence: Average confidence across all member results
            - details: Dict containing:
                - strategy: "synthesis"
                - num_members: Number of results synthesized
                - member_summaries: Brief summary of each member's result
                - avg_confidence: Average confidence score
                - confidence_range: (min, max) confidence values
                - fallback_mode: False (LLM synthesis was used)

        Raises:
            ValueError: If results list is empty or execution_context is not set.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = SynthesisVoting(context=ctx)
            >>> results = [
            ...     MemberResult(
            ...         member=EnsembleMember(method_name="cot"),
            ...         result="Analysis shows X because Y",
            ...         confidence=0.9,
            ...         execution_time_ms=100
            ...     ),
            ...     MemberResult(
            ...         member=EnsembleMember(method_name="tot"),
            ...         result="Tree exploration confirms X via Z",
            ...         confidence=0.85,
            ...         execution_time_ms=200
            ...     ),
            ... ]
            >>> answer, conf, details = await strategy.aggregate_async(results)
            >>> assert isinstance(answer, str)
            >>> assert 0.0 <= conf <= 1.0
            >>> assert details["strategy"] == "synthesis"
            >>> assert details["fallback_mode"] is False
        """
        # Validate inputs
        if not results:
            raise ValueError("Cannot synthesize empty results list")

        if self._execution_context is None:
            raise ValueError(
                "ExecutionContext is required for LLM synthesis. "
                "Use set_context() or call aggregate() for fallback mode."
            )

        # Calculate confidence statistics
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        # Format member results for LLM prompt
        member_summaries = []
        for i, result in enumerate(results, 1):
            summary = (
                f"{i}. Method: {result.member.method_name}\n"
                f"   Result: {result.result}\n"
                f"   Confidence: {result.confidence:.2f}\n"
                f"   Execution time: {result.execution_time_ms}ms"
            )
            member_summaries.append(summary)

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(member_summaries)

        # Use LLM to synthesize
        system_prompt = (
            "You are an expert at synthesizing multiple reasoning results into "
            "a unified, coherent answer. Combine the insights from different "
            "reasoning methods while preserving important details and nuances."
        )

        response = await self._execution_context.sample(
            prompt,
            system_prompt=system_prompt,
        )

        # Extract string from response (sample returns str or BaseModel)
        synthesized_answer = str(response) if not isinstance(response, str) else response

        # Build detailed results
        details: dict[str, Any] = {
            "strategy": "synthesis",
            "num_members": len(results),
            "member_summaries": member_summaries,
            "avg_confidence": avg_confidence,
            "confidence_range": (min_confidence, max_confidence),
            "fallback_mode": False,
        }

        return synthesized_answer, avg_confidence, details

    def aggregate(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Synchronous fallback - combines results textually without LLM.

        When no context is available, this method concatenates unique answers
        with a delimiter. This provides a simple aggregation when LLM synthesis
        is not possible (e.g., during testing or initialization).

        Args:
            results: List of member results to combine. Must not be empty.

        Returns:
            Tuple of (combined_answer, confidence, details) where:
            - combined_answer: Unique answers joined with separator
            - confidence: Average confidence across all member results
            - details: Dict containing:
                - strategy: "synthesis"
                - num_members: Number of results combined
                - unique_answers: Number of distinct answers
                - avg_confidence: Average confidence score
                - confidence_range: (min, max) confidence values
                - fallback_mode: True (textual combination was used)

        Raises:
            ValueError: If results list is empty.

        Examples:
            >>> from reasoning_mcp.models.ensemble import EnsembleMember, MemberResult
            >>> strategy = SynthesisVoting()  # No context
            >>> member1 = EnsembleMember(method_name="cot")
            >>> member2 = EnsembleMember(method_name="tot")
            >>> results = [
            ...     MemberResult(
            ...         member=member1,
            ...         result="Answer A",
            ...         confidence=0.9,
            ...         execution_time_ms=100
            ...     ),
            ...     MemberResult(
            ...         member=member2,
            ...         result="Answer B",
            ...         confidence=0.85,
            ...         execution_time_ms=200
            ...     ),
            ... ]
            >>> answer, conf, details = strategy.aggregate(results)
            >>> assert "Answer A" in answer
            >>> assert "Answer B" in answer
            >>> assert details["fallback_mode"] is True
            >>> assert details["unique_answers"] == 2
        """
        # Validate inputs
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Calculate confidence statistics
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        # Collect unique answers (preserve order)
        seen = set()
        unique_answers = []
        for result in results:
            answer = result.result.strip()
            if answer not in seen:
                seen.add(answer)
                unique_answers.append(answer)

        # Combine unique answers with delimiter
        combined_answer = "\n\n---\n\n".join(unique_answers)

        # Build detailed results
        details: dict[str, Any] = {
            "strategy": "synthesis",
            "num_members": len(results),
            "unique_answers": len(unique_answers),
            "avg_confidence": avg_confidence,
            "confidence_range": (min_confidence, max_confidence),
            "fallback_mode": True,
        }

        return combined_answer, avg_confidence, details

    def _build_synthesis_prompt(self, member_summaries: list[str]) -> str:
        """Build the LLM prompt for synthesizing member results.

        Args:
            member_summaries: List of formatted member result summaries.

        Returns:
            Prompt string for LLM synthesis.
        """
        summaries_text = "\n\n".join(member_summaries)

        prompt = f"""Synthesize the following reasoning results into a single, unified answer.

Your synthesis should:
1. Combine insights from all methods
2. Resolve any conflicts or contradictions
3. Preserve important details and reasoning
4. Produce a clear, coherent final answer

Member Results:
{summaries_text}

Please provide a synthesized answer that integrates these results:"""

        return prompt
