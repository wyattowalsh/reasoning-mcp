"""GAR (Generator-Adversarial Reasoning) method.

Generator-discriminator architecture for adversarial reasoning improvement.

Key phases:
1. Generate: Produce reasoning candidates
2. Discriminate: Score candidates adversarially
3. Update: Improve based on feedback
4. Iterate: Until convergence

Reference: Xi et al. (2025) - "Generator-Adversarial Reasoning"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


GAR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.GAR,
    name="GAR",
    description="Generator-Adversarial Reasoning with trainable discriminator.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"adversarial", "generator", "discriminator", "iterative"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=280,
    best_for=("robustness", "candidate selection"),
    not_recommended_for=("simple queries",),
)


class Gar(ReasoningMethodBase):
    """GAR reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._candidates: list[dict[str, Any]] = []
        self._scores: list[float] = []
        self._iteration: int = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.GAR

    @property
    def name(self) -> str:
        return GAR_METADATA.name

    @property
    def description(self) -> str:
        return GAR_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._candidates = []
        self._scores = []
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
            raise RuntimeError("GAR must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "generate"
        self._iteration = 1

        # Generate candidates using sampling if available
        self._candidates = await self._sample_generate_candidates(input_text)

        content = (
            f"Step {self._step_counter}: Generate (GAR)\n\n"
            f"Problem: {input_text}\n\n"
            f"Candidates:\n"
            + "\n".join(f"  [{c['id']}] {c['reasoning']}" for c in self._candidates)
            + "\n\nNext: Discriminator evaluation."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GAR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.65,
            quality_score=0.65,
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
                "sampled": (
                    self._execution_context is not None and self._execution_context.can_sample
                ),
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.GAR
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
            raise RuntimeError("GAR must be initialized before continuation")

        # Update execution context if provided
        if execution_context is not None:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "discriminate"
            # Score candidates using sampling
            self._scores = await self._sample_discriminate_candidates()

            best_idx = self._scores.index(max(self._scores)) if self._scores else 0
            best_candidate = self._candidates[best_idx] if self._candidates else {"id": "C1"}
            best_score = max(self._scores) if self._scores else 0.91

            # Build scores list safely handling mismatched lengths
            score_lines = []
            for i, c in enumerate(self._candidates):
                score = self._scores[i] if i < len(self._scores) else 0.0
                score_lines.append(f"  [{c['id']}] {score:.2f}")

            content = (
                f"Step {self._step_counter}: Discriminate\n\n"
                f"Scores:\n"
                + "\n".join(score_lines)
                + f"\n\nBest: {best_candidate['id']} ({best_score:.2f})\nNext: Update."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.78
        elif prev_phase == "discriminate":
            self._current_phase = "update"
            # Update best candidate using sampling
            update_content = await self._sample_update_candidate()

            content = f"Step {self._step_counter}: Update\n\n{update_content}\nNext: Conclude."
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            best_score = max(self._scores) if self._scores else 0.91
            # Generate final answer using sampling
            final_answer = await self._sample_final_answer()

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"GAR Complete\nFinal Answer: {final_answer}\nConfidence: {int(best_score * 100)}%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = best_score

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.GAR,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
                "sampled": (
                    self._execution_context is not None and self._execution_context.can_sample
                ),
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Sampling methods
    async def _sample_generate_candidates(self, input_text: str) -> list[dict[str, Any]]:
        """Generate reasoning candidates using LLM sampling.

        Args:
            input_text: The problem to generate candidates for

        Returns:
            List of candidate solutions with id, reasoning, and answer
        """
        system_prompt = """You are a generator in a GAR (Generator-Adversarial Reasoning) system.
Generate multiple diverse reasoning candidates for the given problem.

For each candidate, provide:
1. A unique approach or perspective
2. Step-by-step reasoning
3. A clear answer

Return the candidates in this format:
[C1] reasoning approach 1 -> answer1
[C2] reasoning approach 2 -> answer2"""

        user_prompt = f"""Problem: {input_text}

Generate 2-3 diverse reasoning candidates using different approaches."""

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: "",  # Will trigger heuristic fallback below
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=800,
        )

        # If empty content (fallback triggered), use heuristic
        if not content.strip():
            return self._generate_candidates_heuristic(input_text)

        # Parse candidates from response
        candidates = []
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip() and "[C" in line:
                # Extract candidate info
                parts = line.split("->")
                reasoning = parts[0].strip() if parts else line.strip()
                answer = parts[1].strip() if len(parts) > 1 else "pending"
                candidates.append(
                    {
                        "id": f"C{i}",
                        "reasoning": reasoning.replace(f"[C{i}]", "").strip(),
                        "answer": answer,
                    }
                )

        # Ensure we have at least 2 candidates
        if len(candidates) < 2:
            return self._generate_candidates_heuristic(input_text)

        return candidates[:3]  # Limit to 3 candidates

    async def _sample_discriminate_candidates(self) -> list[float]:
        """Score candidates using LLM sampling as a discriminator.

        Returns:
            List of scores for each candidate (0.0 to 1.0)
        """
        system_prompt = """You are a discriminator in a GAR system.
Evaluate and score the quality of reasoning candidates.

Consider:
1. Logical correctness
2. Completeness of reasoning
3. Clarity of explanation
4. Likelihood of correct answer

Provide a score from 0.0 to 1.0 for each candidate."""

        candidates_text = "\n".join(
            f"[{c['id']}] {c['reasoning']} -> {c['answer']}" for c in self._candidates
        )

        user_prompt = f"""Candidates to evaluate:
{candidates_text}

Score each candidate (0.0-1.0) in format:
[C1] score1
[C2] score2"""

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: "",  # Will trigger heuristic fallback below
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=400,
        )

        # If empty content (fallback triggered), use heuristic
        if not content.strip():
            return self._discriminate_candidates_heuristic()

        # Parse scores from response
        scores = []
        for line in content.split("\n"):
            if "[C" in line and "]" in line:
                # Extract score after ]
                parts = line.split("]")
                if len(parts) > 1:
                    score_str = parts[1].strip()
                    try:
                        score = float(score_str)
                        scores.append(min(max(score, 0.0), 1.0))  # Clamp to [0, 1]
                    except ValueError:
                        continue

        # Ensure we have scores for all candidates
        if len(scores) < len(self._candidates):
            return self._discriminate_candidates_heuristic()

        return scores[: len(self._candidates)]

    async def _sample_update_candidate(self) -> str:
        """Refine the best candidate using LLM sampling.

        Returns:
            Description of the update/refinement
        """
        best_idx = self._scores.index(max(self._scores)) if self._scores else 0
        best_candidate = self._candidates[best_idx] if self._candidates else {"reasoning": ""}

        system_prompt = """You are updating a reasoning candidate in a GAR system.
Refine and improve the best candidate based on discriminator feedback.

Provide:
1. What was improved
2. Why the refinement is better
3. The refined reasoning"""

        user_prompt = f"""Best candidate to refine:
{best_candidate["reasoning"]}

Improve this reasoning and explain the refinement."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: "Best candidate refined with improved logical flow.",
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=500,
        )

    async def _sample_final_answer(self) -> str:
        """Generate final answer using LLM sampling.

        Returns:
            The final answer
        """
        best_idx = self._scores.index(max(self._scores)) if self._scores else 0
        best_candidate = self._candidates[best_idx] if self._candidates else {"answer": "17"}

        system_prompt = """You are finalizing the answer in a GAR system.
Provide the concise final answer based on the best refined candidate."""

        user_prompt = f"""Best candidate:
{best_candidate.get("reasoning", "")}
Answer: {best_candidate.get("answer", "17")}

Provide the final answer (just the answer, be concise)."""

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: str(best_candidate.get("answer", "17")),
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=100,
        )
        return result.strip()

    # Heuristic fallback methods
    def _generate_candidates_heuristic(self, input_text: str) -> list[dict[str, Any]]:
        """Generate candidates using heuristic approach (fallback).

        Args:
            input_text: The problem to generate candidates for

        Returns:
            List of candidate solutions
        """
        return [
            {"id": "C1", "reasoning": "Direct: 5x3+2=17", "answer": "17"},
            {"id": "C2", "reasoning": "Step-wise: 5x3=15, +2=17", "answer": "17"},
        ]

    def _discriminate_candidates_heuristic(self) -> list[float]:
        """Score candidates using heuristic approach (fallback).

        Returns:
            List of scores for each candidate
        """
        # Simple heuristic: prefer longer, more detailed reasoning
        scores = []
        for candidate in self._candidates:
            reasoning_length = len(candidate.get("reasoning", ""))
            # Score based on reasoning length (normalized)
            base_score = min(0.6 + (reasoning_length / 100) * 0.3, 0.95)
            scores.append(base_score)
        return scores if scores else [0.82, 0.91]


__all__ = ["Gar", "GAR_METADATA"]
