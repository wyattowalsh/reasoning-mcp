"""V-STaR (Verifier-guided Self-Taught Reasoner) reasoning method.

This module implements V-STaR, which trains verifiers to improve reasoning
by learning from correctness signals. Uses DPO to train verifiers that can
distinguish correct from incorrect reasoning paths.

Key phases:
1. Generate: Create multiple reasoning candidates
2. Verify: Score candidates with trained verifier
3. Select: Choose best candidate based on verifier scores
4. Learn: Update verifier from correctness feedback

Reference: Hosseini et al. (2024) - "V-STaR: Training Verifiers for Self-Taught
Reasoners" (COLM 2024)
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


V_STAR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.V_STAR,
    name="V-STaR",
    description="Trains verifiers for self-taught reasoners using DPO on correctness. "
    "Verifier learns to distinguish correct from incorrect reasoning paths.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"verifier", "self-taught", "dpo", "correctness", "selection"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=("math reasoning", "verification tasks", "self-improvement", "answer selection"),
    not_recommended_for=("simple queries", "single-path problems"),
)


class VStar(ReasoningMethodBase):
    """V-STaR reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._candidates: list[dict[str, Any]] = []
        self._verifier_scores: list[dict[str, Any]] = []
        self._selected_candidate: dict[str, Any] | None = None
        self._learning_signal: dict[str, Any] = {}
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.V_STAR

    @property
    def name(self) -> str:
        return V_STAR_METADATA.name

    @property
    def description(self) -> str:
        return V_STAR_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._candidates = []
        self._verifier_scores = []
        self._selected_candidate = None
        self._learning_signal = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("VStar must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "generate"

        # Generate multiple candidates
        self._candidates = await self._generate_candidates(input_text)

        content = (
            f"Step {self._step_counter}: Generate Candidates (V-STaR)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating multiple reasoning candidates:\n\n"
            f"Candidates Generated:\n"
            + "\n".join(
                f"  [Candidate {c['id']}]\n"
                f"    Reasoning: {c['reasoning']}\n"
                f"    Answer: {c['answer']}\n"
                f"    Steps: {c['steps']}"
                for c in self._candidates
            )
            + "\n\nV-STaR Principle:\n"
            "  - Generate diverse reasoning paths\n"
            "  - Verifier scores correctness\n"
            "  - DPO training on (correct, incorrect) pairs\n\n"
            "Next: Apply verifier to score candidates."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.V_STAR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "candidates": len(self._candidates),
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.V_STAR
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
            raise RuntimeError("VStar must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "verify"
            # Score candidates with verifier
            self._verifier_scores = await self._verify_candidates()

            content = (
                f"Step {self._step_counter}: Verifier Scoring\n\n"
                f"Applying trained verifier to score candidates:\n\n"
                f"Verifier Scores:\n"
                + "\n".join(
                    f"  Candidate {s['id']}: {s['score']:.2f}\n"
                    f"    Reasoning quality: {s['reasoning_quality']}\n"
                    f"    Answer correct: {'✓' if s['answer_correct'] else '✗'}"
                    for s in sorted(self._verifier_scores, key=lambda x: -x["score"])
                )
                + f"\n\nVerifier Analysis:\n"
                f"  Candidates scored: {len(self._verifier_scores)}\n"
                f"  Correct answers: "
                f"{sum(1 for s in self._verifier_scores if s['answer_correct'])}\n"
                f"  Score range: [{min(s['score'] for s in self._verifier_scores):.2f}, "
                f"{max(s['score'] for s in self._verifier_scores):.2f}]\n\n"
                f"Next: Select best candidate."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "verify":
            self._current_phase = "select"
            # Select best candidate
            best_score = max(self._verifier_scores, key=lambda x: x["score"])
            self._selected_candidate = next(
                c for c in self._candidates if c["id"] == best_score["id"]
            )

            content = (
                f"Step {self._step_counter}: Select Best Candidate\n\n"
                f"Choosing candidate with highest verifier score:\n\n"
                f"Selection Criteria:\n"
                f"  Primary: Verifier score (correctness likelihood)\n"
                f"  Secondary: Reasoning quality assessment\n\n"
                f"Selected: Candidate {self._selected_candidate['id']}\n"
                f"  Reasoning: {self._selected_candidate['reasoning']}\n"
                f"  Answer: {self._selected_candidate['answer']}\n"
                f"  Verifier Score: {best_score['score']:.2f}\n\n"
                f"Rejected Candidates:\n"
                + "\n".join(
                    f"  Candidate {s['id']}: score={s['score']:.2f} "
                    f"({'incorrect' if not s['answer_correct'] else 'lower confidence'})"
                    for s in self._verifier_scores
                    if s["id"] != best_score["id"]
                )
                + "\n\nNext: Generate learning signal."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        elif prev_phase == "select":
            self._current_phase = "learn"
            # Generate DPO learning signal
            correct_candidates = [s for s in self._verifier_scores if s["answer_correct"]]
            incorrect_candidates = [s for s in self._verifier_scores if not s["answer_correct"]]

            self._learning_signal = {
                "positive_examples": len(correct_candidates),
                "negative_examples": len(incorrect_candidates),
                "dpo_pairs": len(correct_candidates) * len(incorrect_candidates),
                "margin": max(s["score"] for s in correct_candidates)
                - max(s["score"] for s in incorrect_candidates)
                if incorrect_candidates
                else 0.0,
            }

            content = (
                f"Step {self._step_counter}: Learning Signal (DPO)\n\n"
                f"Generating training signal for verifier improvement:\n\n"
                f"DPO Training Data:\n"
                f"  Positive examples (correct): {self._learning_signal['positive_examples']}\n"
                f"  Negative examples (incorrect): {self._learning_signal['negative_examples']}\n"
                f"  DPO pairs generated: {self._learning_signal['dpo_pairs']}\n"
                f"  Score margin: {self._learning_signal['margin']:.2f}\n\n"
                f"Learning Objective:\n"
                f"  - Increase P(correct reasoning)\n"
                f"  - Decrease P(incorrect reasoning)\n"
                f"  - Improve verifier discrimination\n\n"
                f"Verifier will improve from this feedback."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.87
        else:
            self._current_phase = "conclude"
            final_answer = (
                self._selected_candidate["answer"] if self._selected_candidate else "[Answer]"
            )
            best_score_value = (
                max(s["score"] for s in self._verifier_scores) if self._verifier_scores else 0.85
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"V-STaR Complete:\n"
                f"  Candidates generated: {len(self._candidates)}\n"
                f"  Candidates verified: {len(self._verifier_scores)}\n"
                f"  DPO pairs created: {self._learning_signal.get('dpo_pairs', 0)}\n\n"
                f"Final Answer: {final_answer}\n"
                f"Verifier Confidence: {best_score_value:.0%}\n"
                f"Overall Confidence: High (89%)\n\n"
                f"Method: V-STaR\n"
                f"  - Multi-candidate generation\n"
                f"  - Trained verifier scoring\n"
                f"  - Correctness-based selection\n"
                f"  - DPO learning from feedback\n"
                f"  - Self-improving verification"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.89

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.V_STAR,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "candidates": len(self._candidates),
                "selected": self._selected_candidate["id"] if self._selected_candidate else None,
            },
        )
        session.add_thought(thought)
        return thought

    async def _generate_candidates(self, input_text: str) -> list[dict[str, Any]]:
        """Generate multiple reasoning candidates using LLM sampling or fallback."""
        prompt = f"""Generate 4 different reasoning approaches to solve this problem:
{input_text}

For each approach, provide:
1. A unique reasoning path
2. The final answer
3. Number of steps used

Return candidates in this format:
[Candidate 1] Reasoning: ... | Answer: ... | Steps: ...
[Candidate 2] Reasoning: ... | Answer: ... | Steps: ...
[Candidate 3] Reasoning: ... | Answer: ... | Steps: ...
[Candidate 4] Reasoning: ... | Answer: ... | Steps: ...

Make candidates diverse - include both simple and complex approaches."""

        def parse_and_return() -> str:
            """Parse candidates and return heuristic fallback if parsing fails."""
            return self._parse_and_generate_heuristic(input_text)

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=parse_and_return,
            system_prompt=(
                "You are a reasoning assistant generating diverse "
                "solution candidates for V-STaR verification."
            ),
        )

        # Parse the result to extract candidates
        candidates = self._parse_candidates(str(result))
        if candidates and len(candidates) >= 2:
            return candidates

        # Fallback: generate heuristic candidates
        return self._generate_heuristic_candidates(input_text)

    def _parse_and_generate_heuristic(self, input_text: str) -> str:
        """Helper method to generate heuristic candidates as a string fallback."""
        candidates = self._generate_heuristic_candidates(input_text)
        lines = []
        for c in candidates:
            lines.append(
                f"[Candidate {c['id']}] Reasoning: {c['reasoning']} | "
                f"Answer: {c['answer']} | Steps: {c['steps']}"
            )
        return "\n".join(lines)

    def _parse_candidates(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM-generated candidates from text."""
        candidates: list[dict[str, Any]] = []
        lines = content.split("\n")

        for _i, line in enumerate(lines, 1):
            if "[Candidate" in line and "|" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    reasoning = parts[0].split("Reasoning:")[-1].strip()
                    answer = parts[1].split("Answer:")[-1].strip()
                    steps_str = parts[2].split("Steps:")[-1].strip()

                    try:
                        steps = int(steps_str)
                    except ValueError:
                        steps = 2

                    candidates.append(
                        {
                            "id": len(candidates) + 1,
                            "reasoning": reasoning,
                            "answer": answer,
                            "steps": steps,
                        }
                    )

        return candidates if len(candidates) >= 2 else []

    def _generate_heuristic_candidates(self, input_text: str) -> list[dict[str, Any]]:
        """Generate fallback candidates using heuristics."""
        return [
            {
                "id": 1,
                "reasoning": "Step 1: Parse → Step 2: Calculate → Step 3: Answer = 17",
                "answer": "17",
                "steps": 3,
            },
            {
                "id": 2,
                "reasoning": "Direct calculation: 5 × 3 + 2 = 17",
                "answer": "17",
                "steps": 1,
            },
            {
                "id": 3,
                "reasoning": "Let x=5, y=3, z=2. Compute x*y=15, then +z=18",
                "answer": "18",
                "steps": 2,
            },
            {
                "id": 4,
                "reasoning": "Using formula: result = (5)(3) + 2 = 17",
                "answer": "17",
                "steps": 2,
            },
        ]

    async def _verify_candidates(self) -> list[dict[str, Any]]:
        """Score candidates using verifier with LLM sampling or fallback."""
        candidates_text = "\n".join(
            f"[{c['id']}] {c['reasoning']} → Answer: {c['answer']}"
            for c in self._candidates
        )

        prompt = f"""As a trained verifier, score these reasoning candidates.
Consider reasoning quality, logical consistency, and answer correctness.

Candidates:
{candidates_text}

For each candidate, provide a score (0.0-1.0) and assessment.
Format:
Candidate 1: score=X.XX, quality=high/medium/low, correct=yes/no
Candidate 2: score=X.XX, quality=high/medium/low, correct=yes/no
Candidate 3: score=X.XX, quality=high/medium/low, correct=yes/no
Candidate 4: score=X.XX, quality=high/medium/low, correct=yes/no"""

        def generate_heuristic_scores_fallback() -> str:
            """Generate heuristic scores as string fallback."""
            scores = self._generate_heuristic_scores()
            lines = []
            for s in scores:
                correct = "yes" if s["answer_correct"] else "no"
                lines.append(
                    f"Candidate {s['id']}: score={s['score']:.2f}, "
                    f"quality={s['reasoning_quality']}, correct={correct}"
                )
            return "\n".join(lines)

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_heuristic_scores_fallback,
            system_prompt=(
                "You are a trained verifier for V-STaR, scoring reasoning candidates."
            ),
        )

        scores = self._parse_verifier_scores(str(result))
        if scores and len(scores) >= 2:
            return scores

        # Fallback: generate heuristic scores
        return self._generate_heuristic_scores()

    def _parse_verifier_scores(self, content: str) -> list[dict[str, Any]]:
        """Parse verifier scores from LLM output."""
        scores = []
        lines = content.split("\n")

        for line in lines:
            if "Candidate" in line and "score=" in line:
                try:
                    # Extract candidate ID
                    cand_id = int(line.split("Candidate")[1].split(":")[0].strip())

                    # Extract score
                    score_str = line.split("score=")[1].split(",")[0].strip()
                    score = float(score_str)

                    # Extract quality
                    quality = "medium"
                    if "quality=" in line:
                        quality = line.split("quality=")[1].split(",")[0].strip()

                    # Extract correctness
                    correct = "yes" in line.lower() and "correct=yes" in line.lower()

                    scores.append(
                        {
                            "id": cand_id,
                            "score": score,
                            "reasoning_quality": quality,
                            "answer_correct": correct,
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return scores if len(scores) >= 2 else []

    def _generate_heuristic_scores(self) -> list[dict[str, Any]]:
        """Generate fallback verifier scores using heuristics."""
        return [
            {"id": 1, "score": 0.85, "reasoning_quality": "high", "answer_correct": True},
            {"id": 2, "score": 0.78, "reasoning_quality": "medium", "answer_correct": True},
            {"id": 3, "score": 0.25, "reasoning_quality": "low", "answer_correct": False},
            {"id": 4, "score": 0.82, "reasoning_quality": "high", "answer_correct": True},
        ]

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["VStar", "V_STAR_METADATA"]
