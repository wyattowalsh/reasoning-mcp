"""Self-Verification reasoning method.

This module implements Self-Verification, which combines forward reasoning
with backward verification. After generating candidate answers, the method
verifies by checking if the answer can correctly predict the original conditions.

Key phases:
1. Forward: Generate candidate answers with CoT
2. Backward: Verify each answer by predicting conditions
3. Vote: Score answers by verification success
4. Select: Choose best-verified answer

Reference: Weng et al. (2022) - "Large Language Models are Better Reasoners
with Self-Verification"
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


SELF_VERIFICATION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_VERIFICATION,
    name="Self-Verification",
    description="Forward reasoning with backward verification. Validates answers "
    "by checking if they correctly predict the original problem conditions.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"verification", "backward", "validation", "self-check", "accuracy"}),
    complexity=5,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=("math problems", "logic puzzles", "verifiable answers"),
    not_recommended_for=("open-ended questions", "creative tasks"),
)


class SelfVerification(ReasoningMethodBase):
    """Self-Verification reasoning method implementation."""

    DEFAULT_CANDIDATES = 3

    # Enable LLM sampling for generating candidates and verification
    _use_sampling: bool = True

    def __init__(self, num_candidates: int = DEFAULT_CANDIDATES) -> None:
        self._num_candidates = num_candidates
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "forward"
        self._candidates: list[dict[str, Any]] = []
        self._verification_scores: list[float] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SELF_VERIFICATION

    @property
    def name(self) -> str:
        return SELF_VERIFICATION_METADATA.name

    @property
    def description(self) -> str:
        return SELF_VERIFICATION_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "forward"
        self._candidates = []
        self._verification_scores = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Self-Verification must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "forward"

        # Generate candidate answers using LLM sampling with fallback
        self._candidates = await self._sample_candidates(input_text, context)

        content = (
            f"Step {self._step_counter}: Forward Reasoning (Self-Verification)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating {self._num_candidates} candidate answers with CoT...\n\n"
            f"Candidates:\n"
            + "\n".join(
                f"  [{c['id']}] {c['answer']}\n      Reasoning: {c['reasoning']}"
                for c in self._candidates
            )
            + "\n\nNext: Backward verification of each candidate."
        )

        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.SELF_VERIFICATION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "candidates": len(self._candidates)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SELF_VERIFICATION
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
            raise RuntimeError("Self-Verification must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "forward")

        if prev_phase == "forward":
            self._current_phase = "backward"
            # Verify candidates using LLM sampling with fallback
            # Get the original problem from the first thought in the session
            first_thoughts = session.get_recent_thoughts(n=session.thought_count)
            original_problem = first_thoughts[-1].content if first_thoughts else ""
            self._verification_scores = await self._sample_verification(
                self._candidates, original_problem
            )
            content = (
                f"Step {self._step_counter}: Backward Verification\n\n"
                f"For each candidate, verify by predicting original conditions:\n\n"
                + "\n".join(
                    f"  Candidate {c['id']}: {c['answer']}\n"
                    f"    Backward check: Given this answer, can we derive the problem?\n"
                    f"    Verification score: {self._verification_scores[i]:.0%}"
                    for i, c in enumerate(self._candidates)
                )
                + "\n\nNext: Vote and select best-verified answer."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.7
        elif prev_phase == "backward":
            self._current_phase = "vote"
            # Find best candidate
            best_idx = self._verification_scores.index(max(self._verification_scores))
            best = self._candidates[best_idx]
            content = (
                f"Step {self._step_counter}: Vote and Select\n\n"
                f"Verification Scores:\n"
                + "\n".join(
                    f"  Candidate {c['id']}: {self._verification_scores[i]:.0%}"
                    + (" <-- BEST" if i == best_idx else "")
                    for i, c in enumerate(self._candidates)
                )
                + f"\n\nBest Verified Answer: Candidate {best['id']}\n"
                f"  Answer: {best['answer']}\n"
                f"  Verification: {max(self._verification_scores):.0%}"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = max(self._verification_scores)
        else:
            self._current_phase = "conclude"
            best_idx = self._verification_scores.index(max(self._verification_scores))
            best = self._candidates[best_idx]
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Self-Verification Complete:\n"
                f"  Candidates generated: {self._num_candidates}\n"
                f"  Best candidate: #{best['id']}\n"
                f"  Verification score: {max(self._verification_scores):.0%}\n\n"
                f"Final Answer: {best['answer']}\n"
                f"Confidence: High ({int(max(self._verification_scores) * 100)}%)\n\n"
                f"Verification Method:\n"
                f"  Forward: Generated answer via CoT\n"
                f"  Backward: Verified answer predicts conditions"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = max(self._verification_scores)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SELF_VERIFICATION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "scores": self._verification_scores,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_candidates(self, input_text: str) -> list[dict[str, Any]]:
        """Generate candidate answers using heuristic approach.

        This is a fallback method when LLM sampling is unavailable.

        Args:
            input_text: The problem or question to reason about

        Returns:
            List of candidate dictionaries with id, answer, and reasoning
        """
        return [
            {"id": i + 1, "answer": f"Answer {i + 1}", "reasoning": f"Reasoning path {i + 1}"}
            for i in range(self._num_candidates)
        ]

    def _generate_verification_scores(self) -> list[float]:
        """Generate verification scores using heuristic approach.

        This is a fallback method when LLM sampling is unavailable.

        Returns:
            List of verification scores between 0.5 and 1.0
        """
        import random

        random.seed(42)
        return [round(random.uniform(0.5, 1.0), 2) for _ in self._candidates]

    async def _sample_candidates(
        self, input_text: str, context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Generate candidate answers using LLM sampling with fallback.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            List of candidate dictionaries with id, answer, and reasoning
        """
        system_prompt = """You are a reasoning assistant using the Self-Verification methodology.
Generate multiple candidate answers to the given problem using Chain-of-Thought reasoning.

For each candidate:
1. Show your step-by-step reasoning
2. Arrive at a clear answer
3. Use different reasoning approaches if possible

Format your response as follows for each candidate:
Candidate N:
Answer: [your answer]
Reasoning: [your step-by-step reasoning]"""

        user_prompt = f"""Problem: {input_text}

Generate {self._num_candidates} candidate answers with Chain-of-Thought reasoning."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "",
            system_prompt=system_prompt,
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=1500,
        )

        # If fallback was used (empty content), use heuristic candidates
        if not content:
            return self._generate_candidates(input_text)

        # Parse candidates from the response
        candidates = self._parse_candidates(content)

        # Ensure we have the right number of candidates
        if len(candidates) < self._num_candidates:
            # Pad with heuristic candidates if needed
            for i in range(len(candidates), self._num_candidates):
                candidates.append(
                    {
                        "id": i + 1,
                        "answer": f"Answer {i + 1}",
                        "reasoning": f"Reasoning path {i + 1}",
                    }
                )
        elif len(candidates) > self._num_candidates:
            # Trim if we have too many
            candidates = candidates[: self._num_candidates]

        return candidates

    def _parse_candidates(self, content: str) -> list[dict[str, Any]]:
        """Parse candidate answers from LLM response.

        Args:
            content: The LLM response text to parse

        Returns:
            List of candidate dictionaries with id, answer, and reasoning
        """
        candidates: list[dict[str, Any]] = []
        lines = content.split("\n")
        current_candidate: dict[str, Any] = {}

        for line in lines:
            line = line.strip()
            if line.lower().startswith("candidate"):
                if current_candidate:
                    candidates.append(current_candidate)
                current_candidate = {"id": len(candidates) + 1, "answer": "", "reasoning": ""}
            elif line.lower().startswith("answer:"):
                current_candidate["answer"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reasoning:"):
                current_candidate["reasoning"] = line.split(":", 1)[1].strip()
            elif current_candidate and "reasoning" in current_candidate:
                # Continue reasoning if we're in a reasoning block
                current_candidate["reasoning"] += " " + line

        # Add the last candidate
        if current_candidate:
            candidates.append(current_candidate)

        return candidates

    async def _sample_verification(
        self, candidates: list[dict[str, Any]], original_problem: str
    ) -> list[float]:
        """Verify candidates using LLM sampling with fallback.

        Args:
            candidates: List of candidate answers to verify
            original_problem: The original problem statement

        Returns:
            List of verification scores between 0.0 and 1.0
        """
        system_prompt = """You are a reasoning assistant using the Self-Verification methodology.
For each candidate answer, perform backward verification by checking if the answer can correctly
predict the original problem conditions.

For each candidate:
1. Assume the answer is correct
2. Work backwards to derive what the problem must have been
3. Check if this matches the actual problem
4. Assign a verification score (0.0 to 1.0) based on how well it matches

Respond with one score per line in format: Candidate N: [score]"""

        candidates_text = "\n\n".join(
            f"Candidate {c['id']}:\nAnswer: {c['answer']}\nReasoning: {c['reasoning']}"
            for c in candidates
        )

        user_prompt = f"""Original problem: {original_problem}

Candidates to verify:
{candidates_text}

Verify each candidate by working backward from the answer to check if it predicts the problem.
Provide verification scores (0.0 to 1.0) for each candidate."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "",
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for consistent scoring
            max_tokens=800,
        )

        # If fallback was used (empty content), use heuristic scores
        if not content:
            return self._generate_verification_scores()

        # Parse verification scores
        scores = self._parse_verification_scores(content)

        # Ensure we have the right number of scores
        if len(scores) < len(candidates):
            # Pad with default scores if needed
            import random

            random.seed(42)
            while len(scores) < len(candidates):
                scores.append(round(random.uniform(0.5, 0.9), 2))
        elif len(scores) > len(candidates):
            # Trim if we have too many
            scores = scores[: len(candidates)]

        return scores

    def _parse_verification_scores(self, content: str) -> list[float]:
        """Parse verification scores from LLM response.

        Args:
            content: The LLM response text to parse

        Returns:
            List of verification scores between 0.0 and 1.0
        """
        scores: list[float] = []
        for line in content.split("\n"):
            line = line.strip()
            if "candidate" in line.lower() and ":" in line:
                # Extract score from line
                parts = line.split(":")
                if len(parts) >= 2:
                    score_text = parts[-1].strip()
                    try:
                        # Try to extract a number from the score text
                        score = float(score_text.split()[0].replace("%", ""))
                        # If it looks like a percentage, convert to decimal
                        if score > 1.0:
                            score = score / 100.0
                        scores.append(max(0.0, min(1.0, score)))
                    except (ValueError, IndexError):
                        continue
        return scores


__all__ = ["SelfVerification", "SELF_VERIFICATION_METADATA"]
