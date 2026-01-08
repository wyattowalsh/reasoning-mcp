"""OR-PRM (Outcome-aware Process Reward Model) verifier.

Process rewards with outcome awareness.

Reference: 2025 - "OR-PRM"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.verifiers.base import VerifierBase, VerifierMetadata
from reasoning_mcp.models.core import VerifierIdentifier


OR_PRM_METADATA = VerifierMetadata(
    identifier=VerifierIdentifier.OR_PRM,
    name="OR-PRM",
    description="Outcome-aware process reward model.",
    tags=frozenset({"outcome-aware", "process-reward", "predictive"}),
    complexity=6,
    supports_step_level=True,
    supports_outcome_level=True,
    supports_cot_verification=True,
    best_for=("outcome prediction", "forward-looking verification"),
    not_recommended_for=("step-only evaluation",),
)


class OrPrm:
    """OR-PRM verifier implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._outcome_weight: float = 0.4

    @property
    def identifier(self) -> str:
        return VerifierIdentifier.OR_PRM

    @property
    def name(self) -> str:
        return OR_PRM_METADATA.name

    @property
    def description(self) -> str:
        return OR_PRM_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._outcome_weight = 0.4

    async def verify(
        self, solution: str, context: dict[str, Any] | None = None
    ) -> tuple[float, str]:
        """Verify with outcome awareness."""
        if not self._initialized:
            raise RuntimeError("OR-PRM must be initialized")

        # Assess both process and outcome
        process_score = self._score_process(solution)
        outcome_score = self._predict_outcome(solution)
        
        combined = (
            process_score * (1 - self._outcome_weight) +
            outcome_score * self._outcome_weight
        )
        
        verification = (
            "Outcome-Aware Verification:\n"
            f"  Process quality: {process_score:.2f}\n"
            f"  Predicted outcome: {outcome_score:.2f}\n"
            f"  Outcome weight: {self._outcome_weight:.2f}\n"
            f"  Combined score: {combined:.2f}\n"
        )
        
        return combined, verification

    def _score_process(self, solution: str) -> float:
        """Score the reasoning process."""
        score = 0.6
        
        if len(solution) > 30:
            score += 0.1
        if any(word in solution.lower() for word in ["step", "first", "then", "finally"]):
            score += 0.1
        if "=" in solution:
            score += 0.1
        
        return min(1.0, score)

    def _predict_outcome(self, solution: str) -> float:
        """Predict outcome quality from solution."""
        score = 0.7
        
        # Outcome indicators
        if any(word in solution.lower() for word in ["answer", "result", "conclusion"]):
            score += 0.1
        if any(c.isdigit() for c in solution):
            score += 0.1
        if "therefore" in solution.lower() or "thus" in solution.lower():
            score += 0.05
        
        return min(1.0, score)

    async def score_with_outcome(
        self,
        steps: list[str],
        outcome: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score steps with outcome awareness."""
        if not self._initialized:
            raise RuntimeError("OR-PRM must be initialized")

        # If outcome provided, use it to adjust step scores
        outcome_bonus = 0.0
        if outcome:
            outcome_quality = self._predict_outcome(outcome)
            outcome_bonus = (outcome_quality - 0.7) * 0.2
        
        scores = []
        for i, step in enumerate(steps):
            process_score = self._score_process(step)
            # Later steps get more outcome influence
            step_outcome_weight = self._outcome_weight * (i + 1) / len(steps)
            adjusted = process_score + outcome_bonus * step_outcome_weight
            scores.append(min(1.0, max(0.0, adjusted)))
        
        return scores

    async def score_steps(
        self, steps: list[str], context: dict[str, Any] | None = None
    ) -> list[float]:
        """Score steps with implicit outcome prediction."""
        return await self.score_with_outcome(steps, None, context)

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["OrPrm", "OR_PRM_METADATA"]
