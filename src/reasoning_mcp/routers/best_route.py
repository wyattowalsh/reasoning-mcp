"""Best-Route router.

Optimal test-time compute allocation.

Reference: 2025 - "Best-Route"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.models.core import RouterIdentifier, MethodIdentifier


BEST_ROUTE_METADATA = RouterMetadata(
    identifier=RouterIdentifier.BEST_ROUTE,
    name="Best-Route",
    description="Optimal test-time compute allocation for reasoning.",
    tags=frozenset({"optimal", "compute", "allocation", "test-time"}),
    complexity=6,
    supports_budget_control=True,
    supports_multi_model=True,
    best_for=("compute optimization", "resource-constrained inference"),
    not_recommended_for=("unlimited compute scenarios",),
)


class BestRoute:
    """Best-Route implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._method_costs: dict[str, float] = {}
        self._method_quality: dict[str, float] = {}

    @property
    def identifier(self) -> str:
        return RouterIdentifier.BEST_ROUTE

    @property
    def name(self) -> str:
        return BEST_ROUTE_METADATA.name

    @property
    def description(self) -> str:
        return BEST_ROUTE_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        # Cost estimates (relative tokens)
        self._method_costs = {
            MethodIdentifier.ZERO_SHOT_COT: 0.1,
            MethodIdentifier.CHAIN_OF_THOUGHT: 0.3,
            MethodIdentifier.SELF_CONSISTENCY: 0.6,
            MethodIdentifier.TREE_OF_THOUGHTS: 0.8,
            MethodIdentifier.MCTS: 1.0,
        }
        # Quality estimates (0-1)
        self._method_quality = {
            MethodIdentifier.ZERO_SHOT_COT: 0.6,
            MethodIdentifier.CHAIN_OF_THOUGHT: 0.75,
            MethodIdentifier.SELF_CONSISTENCY: 0.85,
            MethodIdentifier.TREE_OF_THOUGHTS: 0.9,
            MethodIdentifier.MCTS: 0.95,
        }

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Route to method with best quality/cost ratio."""
        if not self._initialized:
            raise RuntimeError("BestRoute must be initialized")

        budget = context.get("budget", 0.5) if context else 0.5
        
        # Find best method within budget
        best_method = MethodIdentifier.CHAIN_OF_THOUGHT
        best_ratio = 0.0
        
        for method, cost in self._method_costs.items():
            if cost <= budget:
                quality = self._method_quality.get(method, 0.5)
                ratio = quality / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_method = method
        
        return best_method

    async def allocate_budget(
        self, query: str, budget: int
    ) -> dict[str, int]:
        """Optimally allocate budget across methods."""
        if not self._initialized:
            raise RuntimeError("BestRoute must be initialized")

        # Greedy allocation by quality/cost ratio
        allocation = {}
        remaining = budget
        
        # Sort methods by quality/cost ratio
        ratios = [
            (method, self._method_quality.get(method, 0.5) / cost)
            for method, cost in self._method_costs.items()
        ]
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        for method, _ in ratios:
            cost = int(self._method_costs[method] * 100)
            if cost <= remaining:
                allocation[method] = cost
                remaining -= cost
        
        return allocation

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["BestRoute", "BEST_ROUTE_METADATA"]
