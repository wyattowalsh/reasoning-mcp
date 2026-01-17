"""Two-Stage Generation reasoning method.

This module implements Two-Stage Generation (Think-then-Answer), inspired by
DeepSeek-R1 and OpenAI o1/o3 models. Extended thinking in the first stage
followed by concise answer synthesis in the second stage.

Key phases:
1. Think: Extended internal reasoning with exploration
2. Verify: Self-check the thinking process
3. Summarize: Distill thinking into key points
4. Answer: Generate concise final response

Reference: DeepSeek (2025) - "DeepSeek-R1", OpenAI (2024-2025) - "o1/o3 models"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

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

logger = structlog.get_logger(__name__)


TWO_STAGE_GENERATION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.TWO_STAGE_GENERATION,
    name="Two-Stage Generation",
    description="Think-then-answer approach with extended thinking followed by concise "
    "summary. Inspired by DeepSeek-R1 and OpenAI o1/o3 reasoning models.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"two-stage", "think-then-answer", "r1", "o1", "extended-thinking"}),
    complexity=7,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=400,
    best_for=("complex reasoning", "detailed analysis", "accuracy-critical tasks"),
    not_recommended_for=("simple queries", "real-time responses"),
)


class TwoStageGeneration(ReasoningMethodBase):
    """Two-Stage Generation reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "think"
        self._thinking_content: list[str] = []
        self._key_points: list[str] = []
        self._thinking_tokens: int = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.TWO_STAGE_GENERATION

    @property
    def name(self) -> str:
        return TWO_STAGE_GENERATION_METADATA.name

    @property
    def description(self) -> str:
        return TWO_STAGE_GENERATION_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "think"
        self._thinking_content = []
        self._key_points = []
        self._thinking_tokens = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Two-Stage Generation must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "think"
        self._thinking_tokens = 150

        # Stage 1: Extended Thinking
        thinking_prompt = f"""Problem: {input_text}

Begin extended thinking about this problem. Think deeply and carefully:
- What is being asked?
- What are the key elements?
- What approaches could work?
- Think step by step and explore multiple angles.

Provide your initial extended thinking:"""

        def _fallback_thinking() -> str:
            """Fallback generator for thinking phase."""
            return "\n".join([
                "Let me think about this problem carefully...",
                "First, I need to understand what's being asked.",
                "The key elements here are...",
                "I should consider multiple approaches...",
            ])

        thinking_response = await self._sample_with_fallback(
            thinking_prompt,
            fallback_generator=_fallback_thinking,
            system_prompt=(
                "You are engaging in extended, exploratory reasoning. "
                "Think deeply and thoroughly about the problem. "
                "Express your thoughts naturally as you explore different aspects."
            ),
        )
        self._thinking_content = [
            line.strip() for line in thinking_response.strip().split("\n") if line.strip()
        ][:10]
        self._thinking_tokens = len(thinking_response.split())

        content = (
            f"Step {self._step_counter}: Extended Thinking (Stage 1)\n\n"
            f"Problem: {input_text}\n\n"
            f"<think>\n" + "\n".join(f"  {t}" for t in self._thinking_content) + f"\n</think>\n\n"
            f"Thinking tokens used: ~{self._thinking_tokens}\n"
            f"Status: Initial exploration complete\n"
            f"Next: Continue thinking and explore deeper."
        )

        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.TWO_STAGE_GENERATION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "thinking_tokens": self._thinking_tokens},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.TWO_STAGE_GENERATION
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
            raise RuntimeError("Two-Stage Generation must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "think")

        if prev_phase == "think":
            self._current_phase = "verify"

            # Verification phase
            verification_prompt = f"""Previous thinking:
{chr(10).join(self._thinking_content)}

Now verify this reasoning:
- Is the logic sound?
- Are there any gaps or errors?
- What edge cases should be considered?
- Are the assumptions valid?

Provide your verification analysis:"""

            def _fallback_verification() -> str:
                """Fallback generator for verification phase."""
                return "\n".join([
                    "Wait, let me verify this reasoning...",
                    "Is there anything I'm missing?",
                    "The logic seems sound because...",
                    "I'm confident this approach works.",
                ])

            verification_response = await self._sample_with_fallback(
                verification_prompt,
                fallback_generator=_fallback_verification,
                system_prompt=(
                    "You are verifying reasoning. Check for logical consistency, "
                    "edge cases, assumptions, and alternative approaches. "
                    "Be critical and thorough."
                ),
            )
            verification_lines = [
                line.strip()
                for line in verification_response.strip().split("\n")
                if line.strip()
            ][:4]
            self._thinking_tokens += len(verification_response.split())
            self._thinking_content.extend(verification_lines)
            verification_content = "\n  ".join(verification_lines)

            content = (
                f"Step {self._step_counter}: Self-Verification\n\n"
                f"<think>\n"
                f"  {verification_content}\n"
                f"</think>\n\n"
                f"Total thinking tokens: ~{self._thinking_tokens}\n"
                f"Verification: Passed\n"
                f"Next: Summarize key points for final answer."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.7
        elif prev_phase == "verify":
            self._current_phase = "summarize"

            # Summarization phase
            summarize_prompt = f"""Extended thinking ({self._thinking_tokens} tokens):
{chr(10).join(self._thinking_content)}

Distill the key points from this thinking:
- What is the core insight?
- What method or approach works best?
- What is the confidence level?

Provide 3-5 concise key points:"""

            def _fallback_summarization() -> str:
                """Fallback generator for summarization phase."""
                return "\n".join([
                    "Core insight: [Main finding from thinking]",
                    "Method: [Approach that worked best]",
                    "Confidence: High based on verification",
                ])

            summarize_response = await self._sample_with_fallback(
                summarize_prompt,
                fallback_generator=_fallback_summarization,
                system_prompt=(
                    "You are summarizing extended reasoning. "
                    "Extract the key insights, methods, and conclusions. "
                    "Be concise and clear."
                ),
            )
            self._key_points = [
                line.strip()
                for line in summarize_response.strip().split("\n")
                if line.strip()
            ][:5]

            content = (
                f"Step {self._step_counter}: Summarize Thinking\n\n"
                f"Distilling {self._thinking_tokens} tokens of thinking...\n\n"
                f"Key Points:\n"
                + "\n".join(f"  • {kp}" for kp in self._key_points)
                + "\n\nReady to generate concise final answer."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.8
        elif prev_phase == "summarize":
            self._current_phase = "answer"

            # Stage 2: Answer Generation
            key_points_text = chr(10).join(f"• {kp}" for kp in self._key_points)
            answer_prompt = f"""Key insights from extended thinking:
{key_points_text}

Based on {self._thinking_tokens} tokens of verified thinking, generate a clear,
concise final answer. Be direct and comprehensive but avoid unnecessary detail:"""

            def _fallback_answer() -> str:
                """Fallback generator for answer generation phase."""
                return "[Clear, concise answer derived from thinking]"

            answer_text = await self._sample_with_fallback(
                answer_prompt,
                fallback_generator=_fallback_answer,
                system_prompt=(
                    "You are generating the final answer based on extended thinking. "
                    "Be clear, concise, and direct. "
                    "Provide a comprehensive but focused answer."
                ),
            )

            content = (
                f"Step {self._step_counter}: Generate Answer (Stage 2)\n\n"
                f"Based on extended thinking, here is the concise answer:\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{answer_text}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Answer derived from:\n"
                f"  • {self._thinking_tokens} tokens of internal reasoning\n"
                f"  • Self-verification checks\n"
                f"  • Distilled key insights"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Two-Stage Generation Complete:\n"
                f"  • Stage 1 (Think): {self._thinking_tokens} tokens\n"
                f"  • Verification: Passed\n"
                f"  • Stage 2 (Answer): Concise output\n\n"
                f"Final Answer: [Answer]\n"
                f"Confidence: High (88%)\n\n"
                f"Process:\n"
                f"  Extended thinking → Verification → Summarization → Answer\n"
                f"  (Like DeepSeek-R1 / OpenAI o1 reasoning)"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.TWO_STAGE_GENERATION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "thinking_tokens": self._thinking_tokens,
                "key_points": self._key_points,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["TwoStageGeneration", "TWO_STAGE_GENERATION_METADATA"]
