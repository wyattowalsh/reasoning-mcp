"""SelfBudgeter router.

Token allocation optimization based on problem difficulty.

Reference: Test-time compute scaling research (2025)
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.models.core import RouterIdentifier, MethodIdentifier


SELF_BUDGETER_METADATA = RouterMetadata(
    identifier=RouterIdentifier.SELF_BUDGETER,
    name="SelfBudgeter",
    description="Token allocation optimization based on problem difficulty.",
    tags=frozenset({"budget", "allocation", "difficulty", "optimization"}),
    complexity=4,
    supports_budget_control=True,
    supports_multi_model=True,
    best_for=("resource optimization", "multi-method allocation"),
    not_recommended_for=("fixed-budget tasks",),
)


class SelfBudgeter:
    """SelfBudgeter router implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._difficulty_estimate: float = 0.0

    @property
    def identifier(self) -> str:
        return RouterIdentifier.SELF_BUDGETER

    @property
    def name(self) -> str:
        return SELF_BUDGETER_METADATA.name

    @property
    def description(self) -> str:
        return SELF_BUDGETER_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._difficulty_estimate = 0.0

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Route to primary method based on difficulty."""
        if not self._initialized:
            raise RuntimeError("SelfBudgeter must be initialized")

        # Estimate difficulty
        self._difficulty_estimate = self._estimate_difficulty(query)

        if self._difficulty_estimate < 0.3:
            return MethodIdentifier.CHAIN_OF_THOUGHT
        elif self._difficulty_estimate < 0.7:
            return MethodIdentifier.SELF_CONSISTENCY
        else:
            return MethodIdentifier.MCTS

    def _estimate_difficulty(self, query: str) -> float:
        """Estimate problem difficulty from query."""
        # Simple heuristic based on length and keywords
        base = min(1.0, len(query) / 500)
        
        hard_keywords = ["prove", "derive", "optimize", "complex", "multi-step"]
        hard_count = sum(1 for k in hard_keywords if k in query.lower())
        
        return min(1.0, base + (hard_count * 0.15))

    async def allocate_budget(
        self, query: str, budget: int
    ) -> dict[str, int]:
        """Allocate budget across methods based on difficulty."""
        if not self._initialized:
            raise RuntimeError("SelfBudgeter must be initialized")

        difficulty = self._estimate_difficulty(query)

        if difficulty < 0.3:
            # Easy: single method
            return {MethodIdentifier.CHAIN_OF_THOUGHT: budget}
        elif difficulty < 0.7:
            # Medium: split between main and verification
            return {
                MethodIdentifier.CHAIN_OF_THOUGHT: int(budget * 0.6),
                MethodIdentifier.SELF_VERIFICATION: int(budget * 0.4),
            }
        else:
            # Hard: multi-method approach
            return {
                MethodIdentifier.MCTS: int(budget * 0.5),
                MethodIdentifier.SELF_CONSISTENCY: int(budget * 0.3),
                MethodIdentifier.SELF_VERIFICATION: int(budget * 0.2),
            }

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["SelfBudgeter", "SELF_BUDGETER_METADATA"]
