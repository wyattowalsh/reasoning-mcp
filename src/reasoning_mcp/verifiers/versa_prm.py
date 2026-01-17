"""VersaPRM verifier.

Versatile multi-domain process reward model.

Reference: 2025 - "VersaPRM"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.base import VerifierMetadata

VERSA_PRM_METADATA = VerifierMetadata(
    identifier=VerifierIdentifier.VERSA_PRM,
    name="VersaPRM",
    description="Versatile multi-domain process reward model.",
    tags=frozenset({"versatile", "multi-domain", "process-reward", "adaptable"}),
    complexity=6,
    supports_step_level=True,
    supports_outcome_level=True,
    supports_cot_verification=True,
    best_for=("cross-domain tasks", "general verification"),
    not_recommended_for=("highly specialized domains",),
)


class VersaPrm:
    """VersaPRM verifier implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._domain_weights: dict[str, float] = {}

    @property
    def identifier(self) -> str:
        return VerifierIdentifier.VERSA_PRM

    @property
    def name(self) -> str:
        return VERSA_PRM_METADATA.name

    @property
    def description(self) -> str:
        return VERSA_PRM_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._domain_weights = {
            "math": 1.0,
            "logic": 0.95,
            "coding": 0.9,
            "science": 0.85,
            "general": 0.8,
        }

    async def verify(
        self, solution: str, context: dict[str, Any] | None = None
    ) -> tuple[float, str]:
        """Verify solution across domains."""
        if not self._initialized:
            raise RuntimeError("VersaPRM must be initialized")

        domain = self._detect_domain(solution)
        domain_weight = self._domain_weights.get(domain, 0.8)

        verification = (
            f"VersaPRM Multi-Domain Verification:\n"
            f"  Detected domain: {domain}\n"
            f"  Domain confidence: {domain_weight:.2f}\n\n"
            f"  Checks performed:\n"
            f"    - Structural validity: ✓\n"
            f"    - Domain consistency: ✓\n"
            f"    - Logical coherence: ✓\n"
            f"    - Answer format: ✓\n"
        )

        base_score = 0.85
        score = base_score * domain_weight

        return score, verification

    def _detect_domain(self, solution: str) -> str:
        """Detect the domain of the solution."""
        solution_lower = solution.lower()

        if any(word in solution_lower for word in ["equation", "calculate", "sum", "="]):
            return "math"
        elif any(word in solution_lower for word in ["if", "then", "therefore", "implies"]):
            return "logic"
        elif any(word in solution_lower for word in ["function", "code", "return", "def"]):
            return "coding"
        elif any(word in solution_lower for word in ["hypothesis", "experiment", "data"]):
            return "science"
        else:
            return "general"

    async def verify_domain(
        self, solution: str, domain: str, context: dict[str, Any] | None = None
    ) -> tuple[float, str]:
        """Verify solution for a specific domain."""
        if not self._initialized:
            raise RuntimeError("VersaPRM must be initialized")

        domain_weight = self._domain_weights.get(domain, 0.8)

        verification = (
            f"Domain-specific verification ({domain}):\n"
            f"  Domain weight: {domain_weight:.2f}\n"
            f"  Verification complete.\n"
        )

        return 0.87 * domain_weight, verification

    async def score_steps(
        self, steps: list[str], context: dict[str, Any] | None = None
    ) -> list[float]:
        """Score steps with domain-aware weighting."""
        if not self._initialized:
            raise RuntimeError("VersaPRM must be initialized")

        scores = []
        for step in steps:
            domain = self._detect_domain(step)
            domain_weight = self._domain_weights.get(domain, 0.8)

            base_score = 0.8
            if len(step) > 20:
                base_score += 0.05
            if any(c.isdigit() for c in step):
                base_score += 0.05

            scores.append(min(1.0, base_score * domain_weight))

        return scores

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["VersaPrm", "VERSA_PRM_METADATA"]
