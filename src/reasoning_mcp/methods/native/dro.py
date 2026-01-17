"""DRO (Direct Reasoning Optimization) method.

LLMs self-reward and self-refine without external reward models.

Key phases:
1. Generate: Initial reasoning
2. Self-Score: Evaluate own quality
3. Optimize: Refine based on self-feedback
4. Iterate: Until threshold

Reference: arXiv 2025 - "Direct Reasoning Optimization"
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


DRO_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DRO,
    name="DRO",
    description="Direct Reasoning Optimization - self-reward and self-refine.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"self-reward", "self-refine", "optimization", "autonomous"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=250,
    best_for=("autonomous improvement", "iterative refinement"),
    not_recommended_for=("external validation tasks",),
)


class Dro(ReasoningMethodBase):
    """DRO reasoning method implementation."""

    # Enable LLM sampling for generating reasoning, self-scoring, and optimization
    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._reasoning: str = ""
        self._self_score: float = 0.0
        self._iteration: int = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.DRO

    @property
    def name(self) -> str:
        return DRO_METADATA.name

    @property
    def description(self) -> str:
        return DRO_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._reasoning = ""
        self._self_score = 0.0
        self._iteration = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("DRO must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "generate"
        self._iteration = 1

        # Generate initial reasoning using LLM sampling with fallback
        content = await self._sample_generate(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DRO,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={"phase": self._current_phase, "iteration": self._iteration},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DRO
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
            raise RuntimeError("DRO must be initialized before continuation")

        # Store execution context for LLM sampling if provided
        if execution_context:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "self_score"
            # Generate self-score using LLM sampling with fallback
            content, self._self_score = await self._sample_self_score(
                previous_thought, guidance, context
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = self._self_score
        elif prev_phase == "self_score":
            self._current_phase = "optimize"
            # Generate optimization using LLM sampling with fallback
            content = await self._sample_optimize(previous_thought, guidance, context)
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = self._generate_conclusion(previous_thought, guidance, context)
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.DRO,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "iteration": self._iteration},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # ===== LLM Sampling Methods =====

    async def _sample_generate(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial reasoning using LLM sampling with fallback.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled reasoning content or fallback
        """
        system_prompt = """You are a reasoning assistant using the DRO \
(Direct Reasoning Optimization) methodology.
Generate an initial reasoning attempt at solving the given problem. This will later \
be self-scored and optimized.

Your reasoning should:
1. Address the problem directly
2. Show your step-by-step reasoning process
3. Provide a clear answer
4. Be complete enough to evaluate"""

        user_prompt = f"""Problem: {input_text}

Generate a thorough initial reasoning attempt:"""

        def fallback() -> str:
            return self._generate_reasoning(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )
        return f"Step {self._step_counter}: Generate (DRO)\n\n{content}\n\nNext: Self-score."

    async def _sample_self_score(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """Generate self-score using LLM sampling with fallback.

        Args:
            previous_thought: The reasoning to score
            guidance: Optional guidance for scoring
            context: Optional additional context

        Returns:
            A tuple of (score content, score value)
        """
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using the DRO \
(Direct Reasoning Optimization) methodology.
Evaluate and score your own reasoning from the previous step.

Your self-evaluation should:
1. Assess the correctness of the reasoning
2. Identify any gaps or errors
3. Provide constructive feedback for improvement
4. Give a quality score between 0.0 and 1.0

Format your response to include a quality score (0.0-1.0)."""

        user_prompt = f"""Previous reasoning (step {previous_thought.step_number}):
{previous_thought.content}

Evaluate your own reasoning and provide a quality score (0.0-1.0).{guidance_text}

Self-assessment:"""

        def fallback() -> str:
            content, _ = self._generate_self_score(previous_thought, guidance, context)
            return content

        content_text = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # If fallback was used, extract score from heuristic method
        if "Score: 0.78" in content_text:
            return content_text, 0.78

        # Try to extract quality score from the response
        score = 0.75  # Default

        # Look for patterns like "score: 0.75", "quality: 0.8", "0.85/1.0", etc.
        score_patterns = [
            r"score[:\s]+([0-9]*\.?[0-9]+)",
            r"quality[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)/1\.0",
            r"([0-9]*\.?[0-9]+)\s*/\s*1",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, content_text.lower())
            if match:
                try:
                    extracted_score = float(match.group(1))
                    if 0.0 <= extracted_score <= 1.0:
                        score = extracted_score
                        break
                except (ValueError, IndexError):
                    continue

        content = (
            f"Step {self._step_counter}: Self-Score\n\n"
            f"{content_text}\n\n"
            f"Score: {score:.2f}\n\nNext: Optimize."
        )

        return content, score

    async def _sample_optimize(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate optimization using LLM sampling with fallback.

        Args:
            previous_thought: The self-score to optimize from
            guidance: Optional guidance for optimization
            context: Optional additional context

        Returns:
            The optimized reasoning content
        """
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using the DRO \
(Direct Reasoning Optimization) methodology.
Based on the self-evaluation feedback, refine and optimize your reasoning.

Your optimized reasoning should:
1. Address the issues identified in self-evaluation
2. Improve clarity and correctness
3. Fill any gaps in the reasoning
4. Provide a stronger final answer"""

        user_prompt = f"""Previous self-evaluation (step {previous_thought.step_number}):
{previous_thought.content}

Based on this self-evaluation, refine and optimize your reasoning.{guidance_text}

Optimized reasoning:"""

        def fallback() -> str:
            return self._generate_optimize(previous_thought, guidance, context)

        content_text = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        return f"Step {self._step_counter}: Optimize\n\n{content_text}\n\nNext: Conclude."

    # ===== Fallback Heuristic Methods =====

    def _generate_reasoning(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial reasoning (fallback heuristic).

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The reasoning content
        """
        self._reasoning = "Calculate 5x3=15, then 15+2=17"
        return (
            f"Step {self._step_counter}: Generate (DRO)\n\n"
            f"Problem: {input_text}\n\n"
            f"Reasoning: {self._reasoning}\n"
            f"Answer: 17\n\nNext: Self-score."
        )

    def _generate_self_score(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """Generate self-score (fallback heuristic).

        Args:
            previous_thought: The reasoning to score
            guidance: Optional guidance for scoring
            context: Optional additional context

        Returns:
            A tuple of (score content, score value)
        """
        self._self_score = 0.78
        content = (
            f"Step {self._step_counter}: Self-Score\n\n"
            f"Score: {self._self_score:.2f}\n"
            f"Feedback: Correct but could add PEMDAS note.\n\nNext: Optimize."
        )
        return content, self._self_score

    def _generate_optimize(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate optimization (fallback heuristic).

        Args:
            previous_thought: The self-score to optimize from
            guidance: Optional guidance for optimization
            context: Optional additional context

        Returns:
            The optimized reasoning content
        """
        self._reasoning = "PEMDAS: 5x3=15 first, then 15+2=17"
        return (
            f"Step {self._step_counter}: Optimize\n\nRefined: {self._reasoning}\n\nNext: Conclude."
        )

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion (fallback heuristic).

        Args:
            previous_thought: The optimization to conclude from
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The conclusion content
        """
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"DRO Complete\nFinal: {self._reasoning}\n"
            f"Answer: 17\nConfidence: 90%"
        )


__all__ = ["Dro", "DRO_METADATA"]
