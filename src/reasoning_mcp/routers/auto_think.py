"""AutoThink router.

Adaptive CoT activation via classifier - determines if CoT is needed.

Reference: Agarwal et al. (2025) - "Auto-Think"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.models.core import RouterIdentifier, MethodIdentifier


AUTO_THINK_METADATA = RouterMetadata(
    identifier=RouterIdentifier.AUTO_THINK,
    name="AutoThink",
    description="Adaptive CoT activation - classifies if reasoning is needed.",
    tags=frozenset({"adaptive", "classifier", "cot-routing", "efficiency"}),
    complexity=3,
    supports_budget_control=True,
    supports_multi_model=False,
    best_for=("compute efficiency", "simple query detection"),
    not_recommended_for=("always-reason tasks",),
)


class AutoThink:
    """AutoThink router implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._query_complexity: float = 0.0

    @property
    def identifier(self) -> str:
        return RouterIdentifier.AUTO_THINK

    @property
    def name(self) -> str:
        return AUTO_THINK_METADATA.name

    @property
    def description(self) -> str:
        return AUTO_THINK_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._query_complexity = 0.0

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Route query to appropriate method based on complexity."""
        if not self._initialized:
            raise RuntimeError("AutoThink must be initialized before routing")

        # Simple complexity heuristic
        complexity_indicators = [
            "calculate", "compute", "solve", "analyze", "explain",
            "why", "how", "compare", "evaluate", "prove"
        ]
        
        query_lower = query.lower()
        complexity_score = sum(1 for ind in complexity_indicators if ind in query_lower)
        self._query_complexity = min(1.0, complexity_score / 5)

        # Route based on complexity
        if self._query_complexity < 0.3:
            return MethodIdentifier.ZERO_SHOT_COT  # Simple, minimal reasoning
        elif self._query_complexity < 0.6:
            return MethodIdentifier.CHAIN_OF_THOUGHT  # Standard CoT
        else:
            return MethodIdentifier.TREE_OF_THOUGHTS  # Complex, multi-path

    async def allocate_budget(
        self, query: str, budget: int
    ) -> dict[str, int]:
        """Allocate token budget based on query complexity."""
        if not self._initialized:
            raise RuntimeError("AutoThink must be initialized")

        method = await self.route(query)
        # Simple allocation - give all to selected method
        return {method: budget}

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["AutoThink", "AUTO_THINK_METADATA"]
