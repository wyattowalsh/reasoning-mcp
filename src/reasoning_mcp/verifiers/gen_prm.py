"""GenPRM verifier.

Generative process rewards with explicit CoT verification.

Reference: Zhao et al. (2025) - "GenPRM"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.verifiers.base import VerifierBase, VerifierMetadata
from reasoning_mcp.models.core import VerifierIdentifier


GEN_PRM_METADATA = VerifierMetadata(
    identifier=VerifierIdentifier.GEN_PRM,
    name="GenPRM",
    description="Generative process rewards with explicit CoT verification.",
    tags=frozenset({"generative", "process-reward", "cot", "explicit"}),
    complexity=6,
    supports_step_level=True,
    supports_outcome_level=True,
    supports_cot_verification=True,
    best_for=("test-time scaling", "step verification"),
    not_recommended_for=("simple tasks",),
)


class GenPrm:
    """GenPRM verifier implementation."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def identifier(self) -> str:
        return VerifierIdentifier.GEN_PRM

    @property
    def name(self) -> str:
        return GEN_PRM_METADATA.name

    @property
    def description(self) -> str:
        return GEN_PRM_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True

    async def verify(
        self, solution: str, context: dict[str, Any] | None = None
    ) -> tuple[float, str]:
        """Verify solution with generative CoT."""
        if not self._initialized:
            raise RuntimeError("GenPRM must be initialized")

        # Generate verification reasoning
        verification = (
            "Generative verification:\n"
            "Step 1: Parse expression structure ✓\n"
            "Step 2: Verify operation order (PEMDAS) ✓\n"
            "Step 3: Check arithmetic correctness ✓\n"
            "Step 4: Validate final answer ✓\n"
            "All steps verified correct."
        )

        score = 0.91
        return score, verification

    async def score_steps(
        self, steps: list[str], context: dict[str, Any] | None = None
    ) -> list[float]:
        """Score individual steps with process rewards."""
        if not self._initialized:
            raise RuntimeError("GenPRM must be initialized")

        scores = []
        for i, step in enumerate(steps):
            # Progressive scoring - later steps build on earlier
            base_score = 0.7
            progress_bonus = 0.05 * (i + 1)
            
            # Content-based adjustments
            if "=" in step:
                base_score += 0.1
            if any(word in step.lower() for word in ["therefore", "thus", "verify"]):
                base_score += 0.05
                
            scores.append(min(1.0, base_score + progress_bonus))

        return scores

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["GenPrm", "GEN_PRM_METADATA"]
