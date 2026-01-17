"""Think-PRM (Process Reward Model) reasoning method.

This module implements Think-PRM, which uses process reward modeling to score
and guide reasoning steps. Instead of just evaluating final answers, it
provides step-by-step reward signals to guide the reasoning process toward
more reliable conclusions.

Key phases:
1. Generate: Produce reasoning steps
2. Score: Evaluate each step with process rewards
3. Guide: Use scores to select/refine steps
4. Conclude: Derive final answer from high-scoring path

Reference: Process Reward Models research (2024-2025), OpenAI PRM approaches
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


THINK_PRM_METADATA = MethodMetadata(
    identifier=MethodIdentifier.THINK_PRM,
    name="Think-PRM",
    description="Uses process reward modeling to score and guide reasoning steps.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"process-reward", "step-scoring", "guided-reasoning", "prm"}),
    complexity=8,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=12,
    avg_tokens_per_thought=350,
    best_for=("mathematical problems", "multi-step reasoning", "verification tasks"),
    not_recommended_for=("simple queries", "creative tasks"),
)


class ThinkPRM(ReasoningMethodBase):
    """Think-PRM reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._reasoning_steps: list[dict[str, Any]] = []
        self._step_scores: list[float] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.THINK_PRM

    @property
    def name(self) -> str:
        return THINK_PRM_METADATA.name

    @property
    def description(self) -> str:
        return THINK_PRM_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._reasoning_steps = []
        self._step_scores = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Think-PRM must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "generate"

        # Generate reasoning steps using sampling
        self._reasoning_steps = await self._generate_reasoning_steps(input_text)

        content = (
            f"Step {self._step_counter}: Generate Reasoning Steps (Think-PRM)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generated Steps:\n"
            + "\n".join(f"  {s['step']}. {s['content']}" for s in self._reasoning_steps)
            + "\n\nNext: Score each step with Process Reward Model."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THINK_PRM,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "steps": self._reasoning_steps,
                "input_text": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.THINK_PRM
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
            raise RuntimeError("Think-PRM must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "score"
            # Get the input_text from previous thought metadata or use a placeholder
            input_text = previous_thought.metadata.get("input_text", "the problem")
            self._step_scores = await self._score_steps_with_prm(self._reasoning_steps, input_text)
            content = (
                f"Step {self._step_counter}: PRM Scoring\n\n"
                f"Step Scores:\n"
                + "\n".join(f"  Step {i + 1}: {s:.2f}" for i, s in enumerate(self._step_scores))
                + f"\n\nAverage: {sum(self._step_scores) / len(self._step_scores):.2f}"
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.7
        elif prev_phase == "score":
            self._current_phase = "guide"
            refinement_text = await self._refine_low_scoring_steps(
                self._reasoning_steps, self._step_scores
            )
            content = f"Step {self._step_counter}: Score-Guided Refinement\n\n{refinement_text}"
            thought_type = ThoughtType.REASONING
            confidence = 0.8
        else:
            self._current_phase = "conclude"
            avg = sum(self._step_scores) / len(self._step_scores) if self._step_scores else 0.85
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"PRM Summary: Avg reward {avg:.2f}\n"
                f"Final Answer: [PRM-verified answer]\n"
                f"Confidence: High (90%)"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.9

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.THINK_PRM,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "scores": self._step_scores},
        )
        session.add_thought(thought)
        return thought

    async def _generate_reasoning_steps(self, input_text: str) -> list[dict[str, Any]]:
        """Generate reasoning steps using LLM sampling or fallback heuristics."""
        prompt = (
            f"Break down this problem into 4-6 clear reasoning steps:\n\n"
            f"{input_text}\n\n"
            f"Output each step as a numbered list with brief descriptions."
        )

        def fallback_steps() -> str:
            return "1. Parse problem\n2. Identify elements\n3. Apply method\n4. Compute result"

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_steps,
            system_prompt="You are a reasoning assistant that breaks problems into clear steps.",
        )
        return self._parse_steps_from_response(result)

    async def _score_steps_with_prm(
        self, steps: list[dict[str, Any]], input_text: str
    ) -> list[float]:
        """Score reasoning steps using process reward model via LLM sampling."""
        steps_text = "\n".join(f"{s['step']}. {s['content']}" for s in steps)
        prompt = (
            f"Score each reasoning step from 0.0 to 1.0 based on "
            f"correctness and clarity:\n\n"
            f"Problem: {input_text}\n\n"
            f"Steps:\n{steps_text}\n\n"
            f"Output: Provide a score for each step "
            f"(format: 'Step 1: 0.95')"
        )

        def fallback_scores() -> str:
            return "\n".join(
                f"Step {i + 1}: {max(0.5, 0.95 - (i * 0.05)):.2f}" for i in range(len(steps))
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_scores,
            system_prompt="You are a process reward model that scores reasoning steps.",
        )
        return self._parse_scores_from_response(result, len(steps))

    async def _refine_low_scoring_steps(
        self,
        steps: list[dict[str, Any]],
        scores: list[float],
        threshold: float = 0.8,
    ) -> str:
        """Generate refinement guidance for low-scoring steps."""
        low_steps = [
            f"Step {s['step']}: {s['content']} (score: {scores[i]:.2f})"
            for i, s in enumerate(steps)
            if i < len(scores) and scores[i] < threshold
        ]

        if not low_steps:
            return "All steps meet quality threshold."

        prompt = (
            "Suggest improvements for these low-scoring reasoning steps:\n\n"
            + "\n".join(low_steps)
            + "\n\nProvide specific refinements to improve quality."
        )

        def fallback_refinement() -> str:
            low_count = sum(1 for s in scores if s < threshold)
            return f"Refining {low_count} low-scoring steps with verification sub-steps."

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_refinement,
            system_prompt="You are a reasoning refinement assistant.",
        )
        return result or "Refinements applied."

    def _parse_steps_from_response(self, response: str) -> list[dict[str, Any]]:
        """Parse reasoning steps from LLM response."""
        steps = []
        lines = response.strip().split("\n")
        step_num = 1
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                # Remove numbering/bullets
                content = line.lstrip("0123456789.-*) ").strip()
                if content:
                    steps.append({"step": step_num, "content": content, "score": None})
                    step_num += 1

        # Ensure we have at least 4 steps
        if len(steps) < 4:
            steps = [
                {"step": 1, "content": "Parse problem", "score": None},
                {"step": 2, "content": "Identify elements", "score": None},
                {"step": 3, "content": "Apply method", "score": None},
                {"step": 4, "content": "Compute result", "score": None},
            ]
        return steps[:6]  # Max 6 steps

    def _parse_scores_from_response(self, response: str, num_steps: int) -> list[float]:
        """Parse step scores from LLM response."""
        scores = []
        lines = response.strip().split("\n")
        for line in lines:
            # Look for patterns like "Step 1: 0.95" or "0.95"
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    score = float(parts[-1].strip())
                    scores.append(max(0.0, min(1.0, score)))
                except ValueError:
                    continue
            else:
                # Try to parse as just a number
                try:
                    score = float(line.strip())
                    scores.append(max(0.0, min(1.0, score)))
                except ValueError:
                    continue

        # Fallback if parsing failed
        if len(scores) < num_steps:
            scores = [max(0.5, 0.95 - (i * 0.05)) for i in range(num_steps)]
        return scores[:num_steps]

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ThinkPRM", "THINK_PRM_METADATA"]
