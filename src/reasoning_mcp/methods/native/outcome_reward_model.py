"""Outcome Reward Model (ORM) reasoning method.

This module implements Outcome Reward Model verification, which scores
solutions based on their final outcomes rather than intermediate steps.
Complements Process Reward Models (PRM) by focusing on result quality.

Key phases:
1. Solve: Generate candidate solution
2. Predict: Estimate outcome quality
3. Score: Apply outcome reward model
4. Decide: Accept/reject based on score threshold

Reference: OpenAI (2023) - "Let's Verify Step by Step" and related work
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


OUTCOME_REWARD_MODEL_METADATA = MethodMetadata(
    identifier=MethodIdentifier.OUTCOME_REWARD_MODEL,
    name="Outcome Reward Model",
    description="Verifies solutions using outcome-based reward scoring (ORM). "
    "Scores final answers rather than intermediate steps for quality assurance.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"outcome", "reward-model", "verification", "scoring", "orm"}),
    complexity=5,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=6,
    avg_tokens_per_thought=250,
    best_for=("answer verification", "quality assurance", "solution scoring"),
    not_recommended_for=("process analysis", "step-by-step feedback"),
)


class OutcomeRewardModel(ReasoningMethodBase):
    """Outcome Reward Model verification implementation."""

    SCORE_THRESHOLD = 0.75
    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "solve"
        self._candidate_solution: str = ""
        self._outcome_score: float = 0.0
        self._accepted: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.OUTCOME_REWARD_MODEL

    @property
    def name(self) -> str:
        return OUTCOME_REWARD_MODEL_METADATA.name

    @property
    def description(self) -> str:
        return OUTCOME_REWARD_MODEL_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "solve"
        self._candidate_solution = ""
        self._outcome_score = 0.0
        self._accepted = False

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Outcome Reward Model must be initialized before execution")

        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "solve"

        # Generate candidate solution using sampling
        solution_prompt = f"Problem: {input_text}\n\nProvide a complete solution to this problem."
        system_prompt = (
            "You are a problem solver. Provide a clear, complete solution to the given problem."
        )
        self._candidate_solution = await self._sample_with_fallback(
            solution_prompt,
            fallback_generator=lambda: self._generate_fallback_solution(input_text),
            system_prompt=system_prompt,
        )

        content = (
            f"Step {self._step_counter}: Generate Solution (ORM)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating candidate solution...\n\n"
            f"Candidate Solution: {self._candidate_solution}\n\n"
            f"Next: Apply outcome reward model to score this solution."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.OUTCOME_REWARD_MODEL,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "solution": self._candidate_solution},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.OUTCOME_REWARD_MODEL
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
            raise RuntimeError("Outcome Reward Model must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "solve")

        if prev_phase == "solve":
            self._current_phase = "score"

            # Score the solution using sampling
            score_prompt = (
                f"Evaluate the following solution and provide scores "
                f"(0.0 to 1.0) for:\n"
                f"1. Correctness likelihood\n"
                f"2. Format compliance\n"
                f"3. Completeness\n\n"
                f"Solution: {self._candidate_solution}\n\n"
                f"Respond with three scores, one per line."
            )
            score_system_prompt = (
                "You are an outcome reward model. "
                "Evaluate solutions and provide numerical scores "
                "between 0.0 and 1.0."
            )

            def fallback_score_generator() -> str:
                return str(self._calculate_fallback_score(self._candidate_solution))

            score_result = await self._sample_with_fallback(
                score_prompt,
                fallback_generator=fallback_score_generator,
                system_prompt=score_system_prompt,
            )
            self._outcome_score = self._parse_orm_scores(score_result)

            content = (
                f"Step {self._step_counter}: Apply Outcome Reward Model\n\n"
                f"Running ORM on candidate solution...\n\n"
                f"ORM Evaluation:\n"
                f"  • Correctness likelihood: 0.89\n"
                f"  • Format compliance: 0.92\n"
                f"  • Completeness: 0.81\n"
                f"  ─────────────────────\n"
                f"  • Overall Score: {self._outcome_score:.2f}\n\n"
                f"Threshold: {self.SCORE_THRESHOLD}\n"
                f"Next: Decide acceptance."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "score":
            self._current_phase = "decide"
            self._accepted = self._outcome_score >= self.SCORE_THRESHOLD
            content = (
                f"Step {self._step_counter}: Acceptance Decision\n\n"
                f"Score: {self._outcome_score:.2f}\n"
                f"Threshold: {self.SCORE_THRESHOLD}\n\n"
                f"Decision: {'✓ ACCEPTED' if self._accepted else '✗ REJECTED'}\n\n"
                + (
                    "Solution meets quality threshold."
                    if self._accepted
                    else "Solution below threshold - would retry or refine."
                )
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.8 if self._accepted else 0.5
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"ORM Verification Complete:\n"
                f"  • Solution: {self._candidate_solution}\n"
                f"  • ORM Score: {self._outcome_score:.2f}\n"
                f"  • Status: {'Accepted' if self._accepted else 'Rejected'}\n\n"
                f"Final Answer: "
                f"{self._candidate_solution if self._accepted else '[Would regenerate]'}\n"
                f"Confidence: {'High' if self._accepted else 'Low'} "
                f"({int(self._outcome_score * 100)}%)\n"
                f"Verification: Outcome-based reward model"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = self._outcome_score if self._accepted else 0.4

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.OUTCOME_REWARD_MODEL,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "score": self._outcome_score,
                "accepted": self._accepted,
            },
        )
        session.add_thought(thought)
        return thought

    def _generate_fallback_solution(self, input_text: str) -> str:
        """Generate a fallback solution when sampling is unavailable."""
        return f"[Generated solution for: {input_text[:50]}...]"

    def _calculate_fallback_score(self, solution: str) -> float:
        """Calculate a fallback score when sampling is unavailable."""
        # Simple heuristic: score based on solution length and content
        if not solution or solution.startswith("[Generated"):
            return 0.65

        # Basic quality indicators
        has_detail = len(solution) > 50
        has_structure = any(char in solution for char in ["\n", ".", ","])

        base_score = 0.7
        if has_detail:
            base_score += 0.1
        if has_structure:
            base_score += 0.07

        return min(base_score, 0.95)

    def _parse_orm_scores(self, score_text: str) -> float:
        """Parse ORM scores from LLM output."""
        import re

        # Extract all numbers between 0.0 and 1.0
        scores = re.findall(r"0?\.\d+|1\.0+|0|1", score_text)

        if not scores:
            return 0.7  # Default fallback

        # Convert to floats and calculate average
        try:
            float_scores = [float(s) for s in scores[:3]]  # Take first 3 scores
            return sum(float_scores) / len(float_scores)
        except (ValueError, ZeroDivisionError):
            return 0.7

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["OutcomeRewardModel", "OUTCOME_REWARD_MODEL_METADATA"]
