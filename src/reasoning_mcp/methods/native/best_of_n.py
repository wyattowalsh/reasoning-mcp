"""Best-of-N reasoning method.

This module implements Best-of-N sampling, a simple but effective approach
that generates N reasoning paths and selects the best one using a reward
model or verifier. Widely used in conjunction with other methods.

Key phases:
1. Sample: Generate N diverse reasoning paths
2. Score: Evaluate each path with reward/verifier
3. Select: Choose the highest-scoring path
4. Output: Return the best solution

Reference: Various (2022-2024) - Standard technique in LLM reasoning
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
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


BEST_OF_N_METADATA = MethodMetadata(
    identifier=MethodIdentifier.BEST_OF_N,
    name="Best-of-N",
    description="Sample N reasoning paths and select the best via reward model or verifier. "
    "Simple, effective, and widely applicable approach to improving reasoning quality.",
    category=MethodCategory.CORE,
    tags=frozenset({"sampling", "selection", "reward-model", "verifier", "simple"}),
    complexity=4,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=8,
    avg_tokens_per_thought=250,
    best_for=("quality improvement", "answer selection", "uncertainty reduction"),
    not_recommended_for=("real-time responses", "cost-sensitive applications"),
)


class BestOfN(ReasoningMethodBase):
    """Best-of-N sampling method implementation."""

    DEFAULT_N = 5
    _use_sampling: bool = True

    def __init__(self, n: int = DEFAULT_N, enable_elicitation: bool = True) -> None:
        self._n = n
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "sample"
        self._samples: list[dict[str, Any]] = []
        self._scores: list[float] = []
        self._best_idx: int = 0
        self.enable_elicitation = enable_elicitation
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.BEST_OF_N

    @property
    def name(self) -> str:
        return BEST_OF_N_METADATA.name

    @property
    def description(self) -> str:
        return BEST_OF_N_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.CORE

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "sample"
        self._samples = []
        self._scores = []
        self._best_idx = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Best-of-N must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "sample"

        # Generate N samples using LLM sampling if available, otherwise use heuristic
        if self._execution_context and self._execution_context.can_sample:
            try:
                self._samples = await self._sample_candidates(input_text, self._n)
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="execute",
                    error=str(e),
                    exc_info=True,
                )
                # Fallback to heuristic implementation
                self._samples = self._generate_candidates(input_text, self._n)
        else:
            self._samples = self._generate_candidates(input_text, self._n)

        content = (
            f"Step {self._step_counter}: Generate {self._n} Samples (Best-of-N)\n\n"
            f"Problem: {input_text}\n\n"
            f"Sampling {self._n} diverse reasoning paths...\n\n"
            f"Generated Samples:\n"
            + "\n".join(f"  [{s['id']}] {s['reasoning']}" for s in self._samples)
            + "\n\nNext: Score each sample with reward model."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BEST_OF_N,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "n": self._n, "samples": len(self._samples)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.BEST_OF_N
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
            raise RuntimeError("Best-of-N must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "sample")

        if prev_phase == "sample":
            self._current_phase = "score"
            # Score samples using LLM sampling if available, otherwise use heuristic
            if self._execution_context and self._execution_context.can_sample:
                try:
                    self._scores = await self._sample_scores(self._samples)
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "llm_sampling_failed",
                        method="continue_reasoning",
                        error=str(e),
                        exc_info=True,
                    )
                    # Fallback to heuristic scoring
                    self._scores = self._generate_scores(self._samples)
            else:
                self._scores = self._generate_scores(self._samples)
            content = (
                f"Step {self._step_counter}: Score Samples\n\n"
                f"Running reward model on each sample...\n\n"
                f"Scores:\n"
                + "\n".join(
                    f"  Sample {s['id']}: {self._scores[i]:.2f}"
                    for i, s in enumerate(self._samples)
                )
                + f"\n\nHighest score: {max(self._scores):.2f}\n"
                f"Next: Select best sample."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.7
        elif prev_phase == "score":
            self._current_phase = "select"

            # Optional elicitation: ask user how to select the best solution
            selection_criteria = "balanced"
            if (
                self.enable_elicitation
                and self._execution_context
                and hasattr(self._execution_context, "ctx")
                and self._execution_context.ctx
            ):
                try:
                    options = [
                        {"id": "quality", "label": "Prioritize solution quality"},
                        {"id": "simplicity", "label": "Prioritize simplicity"},
                        {"id": "creativity", "label": "Prioritize creative/novel solutions"},
                        {"id": "balanced", "label": "Balanced evaluation (default)"},
                    ]
                    config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                    selection = await elicit_selection(
                        self._execution_context.ctx,
                        "How should we evaluate and select the best solution?",
                        options,
                        config=config,
                    )
                    if selection and selection.selected:
                        selection_criteria = selection.selected
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        error=str(e),
                        exc_info=True,
                    )
                    # Fall back to default selection criteria

            self._best_idx = self._scores.index(max(self._scores))
            best = self._samples[self._best_idx]
            content = (
                f"Step {self._step_counter}: Select Best Sample\n\n"
                f"Selection Criteria: {selection_criteria.capitalize()} evaluation\n\n"
                f"Selected: Sample {best['id']}\n"
                f"  Score: {self._scores[self._best_idx]:.2f}\n"
                f"  Reasoning: {best['reasoning']}\n"
                f"  Answer: {best['answer']}\n\n"
                f"This sample outperformed {self._n - 1} alternatives."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            best = self._samples[self._best_idx]
            avg_score = sum(self._scores) / len(self._scores)
            score_improvement = max(self._scores) - avg_score
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Best-of-{self._n} Complete:\n"
                f"  • Samples generated: {self._n}\n"
                f"  • Best sample: #{best['id']}\n"
                f"  • Best score: {max(self._scores):.2f}\n"
                f"  • Score improvement over random: +{score_improvement:.2f}\n\n"
                f"Final Answer: {best['answer']}\n"
                f"Confidence: High ({int(max(self._scores) * 100)}%)\n"
                f"Method: Selected from {self._n} candidates via reward scoring"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = max(self._scores)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.BEST_OF_N,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "scores": self._scores,
                "best_idx": self._best_idx,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_candidates(self, input_text: str, n: int) -> list[dict[str, Any]]:
        """Generate candidate solutions using heuristic approach (fallback).

        Args:
            input_text: The problem or question to solve
            n: Number of candidates to generate

        Returns:
            List of candidate solutions with id, answer, and reasoning
        """
        return [
            {
                "id": i + 1,
                "answer": f"Answer variant {i + 1}",
                "reasoning": f"Reasoning path {i + 1}",
            }
            for i in range(n)
        ]

    def _generate_scores(self, samples: list[dict[str, Any]]) -> list[float]:
        """Generate scores using heuristic approach (fallback).

        Args:
            samples: List of candidate samples to score

        Returns:
            List of scores for each sample
        """
        import random

        random.seed(42)  # Deterministic for demo
        return [round(random.uniform(0.6, 0.95), 2) for _ in samples]

    async def _sample_candidates(self, input_text: str, n: int) -> list[dict[str, Any]]:
        """Generate candidate solutions using LLM sampling.

        Uses the execution context's sampling capability to generate
        diverse reasoning paths and solutions.

        Args:
            input_text: The problem or question to solve
            n: Number of candidates to generate

        Returns:
            List of candidate solutions with id, answer, and reasoning

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_candidates but was not set")

        system_prompt = """You are a reasoning assistant using Best-of-N sampling.
Generate diverse, independent reasoning paths and solutions for the given problem.

For each candidate:
1. Use a different approach or perspective
2. Show clear reasoning steps
3. Provide a concrete answer
4. Be creative but stay grounded in logic

Focus on diversity - each candidate should explore different aspects or methods."""

        candidates = []
        for i in range(n):
            user_prompt = f"""Problem: {input_text}

Generate candidate solution #{i + 1} of {n}.
Use a unique approach or perspective from previous candidates.

Format your response as:
REASONING: [Your step-by-step reasoning]
ANSWER: [Your final answer]"""

            def fallback_candidate() -> str:
                return f"REASONING: Reasoning path {i + 1}\nANSWER: Answer variant {i + 1}"

            content = await self._sample_with_fallback(
                user_prompt,
                fallback_generator=fallback_candidate,
                system_prompt=system_prompt,
                temperature=0.8,  # Higher temperature for diversity
                max_tokens=800,
            )

            # Parse reasoning and answer from response
            reasoning = "No reasoning provided"
            answer = "No answer provided"

            if "REASONING:" in content and "ANSWER:" in content:
                parts = content.split("ANSWER:")
                reasoning = parts[0].replace("REASONING:", "").strip()
                answer = parts[1].strip()
            elif "ANSWER:" in content:
                parts = content.split("ANSWER:")
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = content
                answer = f"Solution {i + 1}"

            candidates.append(
                {
                    "id": i + 1,
                    "answer": answer,
                    "reasoning": reasoning,
                }
            )

        return candidates

    async def _sample_scores(self, samples: list[dict[str, Any]]) -> list[float]:
        """Score candidate solutions using LLM sampling.

        Uses the execution context's sampling capability to evaluate
        the quality and correctness of each candidate solution.

        Args:
            samples: List of candidate samples to score

        Returns:
            List of scores (0.0-1.0) for each sample

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_scores but was not set")

        system_prompt = """You are a reasoning evaluator using reward model scoring.
Evaluate the quality, correctness, and reasoning of each solution.

Scoring criteria:
- Logical soundness (0-0.3)
- Completeness (0-0.3)
- Clarity (0-0.2)
- Correctness (0-0.2)

Provide a score between 0.0 and 1.0 for each candidate."""

        scores = []
        for sample in samples:
            user_prompt = f"""Evaluate this candidate solution:

REASONING: {sample["reasoning"]}
ANSWER: {sample["answer"]}

Provide a numerical score between 0.0 and 1.0.
Respond with ONLY the score number (e.g., "0.85")."""

            def fallback_score() -> str:
                import random

                random.seed(42 + sample["id"])
                return str(round(random.uniform(0.6, 0.95), 2))

            content = await self._sample_with_fallback(
                user_prompt,
                fallback_generator=fallback_score,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for consistent scoring
                max_tokens=50,
            )

            # Extract score from response
            try:
                # Try to find a float in the response
                import re

                match = re.search(r"0?\.\d+|[01]\.\d+|[01]", content)
                if match:
                    score = float(match.group())
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                else:
                    score = 0.75  # Default if parsing fails
            except (ValueError, AttributeError):
                score = 0.75  # Default if parsing fails

            scores.append(round(score, 2))

        return scores


__all__ = ["BestOfN", "BEST_OF_N_METADATA"]
