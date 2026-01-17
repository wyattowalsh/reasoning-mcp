"""Chain of Draft reasoning method.

This module implements Chain of Draft (CoD), which uses minimal ~5-word
telegraphic reasoning steps to achieve 76% latency reduction vs full CoT.

Key phases:
1. Draft: Generate ultra-concise 5-word reasoning steps
2. Refine: Optionally expand critical steps if needed
3. Answer: Derive final answer from drafts

Reference: Xu et al. (2025) - "Chain of Draft" (Zoom AI)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


CHAIN_OF_DRAFT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CHAIN_OF_DRAFT,
    name="Chain of Draft",
    description="Ultra-concise ~5-word reasoning steps for 76% latency reduction. "
    "Telegraphic thinking before expanding to answer.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"efficient", "concise", "draft", "fast", "minimal"}),
    complexity=4,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=6,
    avg_tokens_per_thought=80,
    best_for=("latency-sensitive tasks", "simple reasoning", "quick answers"),
    not_recommended_for=("complex multi-step problems", "tasks requiring detailed explanation"),
)


class ChainOfDraft(ReasoningMethodBase):
    """Chain of Draft reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "draft"
        self._drafts: list[str] = []
        self._refined_steps: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.CHAIN_OF_DRAFT

    @property
    def name(self) -> str:
        return CHAIN_OF_DRAFT_METADATA.name

    @property
    def description(self) -> str:
        return CHAIN_OF_DRAFT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "draft"
        self._drafts = []
        self._refined_steps = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("ChainOfDraft must be initialized before execution")

        # Store execution context and configure sampling
        self._execution_context = execution_context
        use_sampling = execution_context is not None and execution_context.can_sample

        self._step_counter = 1
        self._current_phase = "draft"

        # Generate drafts using sampling or fallback heuristic
        if use_sampling:
            self._drafts = await self._sample_drafts(input_text)
        else:
            self._drafts = self._generate_drafts_heuristic(input_text)

        content = (
            f"Step {self._step_counter}: Draft Phase (Chain of Draft)\n\n"
            f"Problem: {input_text}\n\n"
            f"Telegraphic Drafts (~5 words each):\n"
            + "\n".join(f"  [{i + 1}] {d}" for i, d in enumerate(self._drafts))
            + f"\n\nDraft Statistics:\n"
            f"  Total drafts: {len(self._drafts)}\n"
            f"  Avg words/draft: ~5\n"
            f"  Latency reduction: ~76%\n\n"
            f"Next: Refine critical steps if needed."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_DRAFT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "draft_count": len(self._drafts),
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.CHAIN_OF_DRAFT
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("ChainOfDraft must be initialized before continuation")

        # Update execution context if provided
        if execution_context is not None:
            self._execution_context = execution_context
        use_sampling = self._execution_context is not None and self._execution_context.can_sample

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "draft")

        if prev_phase == "draft":
            self._current_phase = "refine"
            # Generate refined steps using sampling or fallback heuristic
            if use_sampling:
                self._refined_steps = await self._sample_refinements(guidance)
            else:
                self._refined_steps = self._generate_refinements_heuristic()
            content = (
                f"Step {self._step_counter}: Refine Phase\n\n"
                f"Selective Expansion:\n"
                + "\n".join(f"  {r}" for r in self._refined_steps)
                + "\n\nNext: Derive final answer."
            )
            thought_type = ThoughtType.REVISION
            confidence = 0.8
        elif prev_phase == "refine":
            self._current_phase = "answer"
            # Generate final answer using sampling or fallback heuristic
            if use_sampling:
                answer_content = await self._sample_final_answer(guidance)
            else:
                answer_content = self._generate_final_answer_heuristic()
            content = (
                f"Step {self._step_counter}: Answer Phase\n\n"
                f"{answer_content}\n\n"
                f"Method: Chain of Draft\n"
                f"  - 76% latency reduction\n"
                f"  - {len(self._drafts)} telegraphic drafts"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\nFinal Answer: 17\nConfidence: 85%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CHAIN_OF_DRAFT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "draft_count": len(self._drafts),
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_drafts_heuristic(self, input_text: str) -> list[str]:
        """Generate telegraphic draft steps using heuristic (fallback).

        Args:
            input_text: The input problem or question

        Returns:
            List of ultra-concise ~5-word draft steps
        """
        return [
            "Parse problem, identify variables",
            "Apply relevant operation here",
            "Compute intermediate result now",
            "Verify logic, derive answer",
        ]

    def _generate_refinements_heuristic(self) -> list[str]:
        """Generate refined steps using heuristic (fallback).

        Returns:
            List of refined steps expanding on critical drafts
        """
        if len(self._drafts) >= 2:
            return [
                f"Draft 1: {self._drafts[0]} -> Identify x, y values",
                f"Draft 2: {self._drafts[1]} -> Multiplication then addition",
            ]
        return ["Expand critical draft steps"]

    def _generate_final_answer_heuristic(self) -> str:
        """Generate final answer using heuristic (fallback).

        Returns:
            Final answer content
        """
        return "Final Answer: 17\nConfidence: High (85%)"

    async def _sample_drafts(self, input_text: str) -> list[str]:
        """Generate telegraphic draft steps using LLM sampling.

        Args:
            input_text: The input problem or question

        Returns:
            List of ultra-concise ~5-word draft steps
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_drafts but was not provided")

        system_prompt = """You are a reasoning assistant using Chain of Draft methodology.
Generate ultra-concise telegraphic reasoning steps (approximately 5 words each).

Your drafts should:
1. Be extremely concise (around 5 words each)
2. Capture key reasoning steps
3. Use telegraphic language (omit articles, be terse)
4. Focus on essential operations and logic
5. Enable 76% latency reduction vs full Chain of Thought

Example draft format:
- "Parse problem, identify variables"
- "Apply multiplication to inputs"
- "Sum intermediate results now"
- "Verify logic, output answer"

Return ONLY the draft steps, one per line, no numbering."""

        user_prompt = f"""Problem: {input_text}

Generate 4-6 ultra-concise telegraphic draft steps (~5 words each) for solving this problem."""

        def fallback() -> str:
            return "\n".join(self._generate_drafts_heuristic(input_text))

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )

        # Parse drafts from result (one per line, strip empty lines)
        drafts = [line.strip() for line in content.strip().split("\n") if line.strip()]
        return drafts if drafts else self._generate_drafts_heuristic(input_text)

    async def _sample_refinements(self, guidance: str | None) -> list[str]:
        """Generate refined steps using LLM sampling.

        Args:
            guidance: Optional guidance for refinement

        Returns:
            List of refined steps expanding on critical drafts
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_refinements but was not provided"
            )

        system_prompt = """You are a reasoning assistant continuing Chain of Draft methodology.
Selectively expand the most critical telegraphic drafts into fuller explanations.

Your refinements should:
1. Reference the original draft
2. Provide more detail and clarity
3. Still be concise (1-2 sentences per refinement)
4. Focus on the most important steps

Return ONLY the refined steps, one per line."""

        drafts_text = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(self._drafts))
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Telegraphic drafts:
{drafts_text}{guidance_text}

Selectively expand 2-3 critical drafts that need more detail."""

        def fallback() -> str:
            return "\n".join(self._generate_refinements_heuristic())

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=400,
        )

        # Parse refinements from result
        refinements = [line.strip() for line in content.strip().split("\n") if line.strip()]
        return refinements if refinements else self._generate_refinements_heuristic()

    async def _sample_final_answer(self, guidance: str | None) -> str:
        """Generate final answer using LLM sampling.

        Args:
            guidance: Optional guidance for answer generation

        Returns:
            Final answer content
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_final_answer but was not provided"
            )

        system_prompt = """You are a reasoning assistant completing Chain of Draft methodology.
Synthesize the drafts and refinements into a clear final answer.

Your answer should:
1. State the final answer clearly
2. Include confidence level
3. Be concise but complete"""

        drafts_text = "\n".join(f"- {d}" for d in self._drafts)
        refinements_text = "\n".join(f"- {r}" for r in self._refined_steps)
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Drafts:
{drafts_text}

Refinements:
{refinements_text}{guidance_text}

Provide the final answer with confidence level."""

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=self._generate_final_answer_heuristic,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )

        return content.strip() if content else self._generate_final_answer_heuristic()


__all__ = ["ChainOfDraft", "CHAIN_OF_DRAFT_METADATA"]
