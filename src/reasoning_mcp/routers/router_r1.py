"""Router-R1 router.

RL-based multi-round routing with learned policies.

Reference: 2025 - "Router-R1"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.models.core import RouterIdentifier, MethodIdentifier


ROUTER_R1_METADATA = RouterMetadata(
    identifier=RouterIdentifier.ROUTER_R1,
    name="Router-R1",
    description="RL-based multi-round routing with learned policies.",
    tags=frozenset({"rl", "learned", "policy", "multi-round"}),
    complexity=6,
    supports_budget_control=True,
    supports_multi_model=True,
    best_for=("complex routing", "adaptive strategies"),
    not_recommended_for=("simple queries",),
)


class RouterR1:
    """Router-R1 implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._policy: dict[str, float] = {}
        self._history: list[dict[str, Any]] = []

    @property
    def identifier(self) -> str:
        return RouterIdentifier.ROUTER_R1

    @property
    def name(self) -> str:
        return ROUTER_R1_METADATA.name

    @property
    def description(self) -> str:
        return ROUTER_R1_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        # Initialize policy with default preferences
        self._policy = {
            MethodIdentifier.CHAIN_OF_THOUGHT: 0.3,
            MethodIdentifier.TREE_OF_THOUGHTS: 0.25,
            MethodIdentifier.SELF_CONSISTENCY: 0.25,
            MethodIdentifier.MCTS: 0.2,
        }
        self._history = []

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Route using learned RL policy."""
        if not self._initialized:
            raise RuntimeError("RouterR1 must be initialized")

        # Simple policy-based selection
        # In practice, this would use a trained RL model
        
        query_features = self._extract_features(query)
        
        # Adjust policy based on features
        adjusted_policy = self._policy.copy()
        if query_features["complexity"] > 0.7:
            adjusted_policy[MethodIdentifier.TREE_OF_THOUGHTS] += 0.2
            adjusted_policy[MethodIdentifier.MCTS] += 0.1
        
        # Select method with highest adjusted weight
        best_method = max(adjusted_policy, key=adjusted_policy.get)
        
        # Record in history
        self._history.append({
            "query_hash": hash(query) % 10000,
            "selected": best_method,
            "policy_state": adjusted_policy,
        })
        
        return best_method

    def _extract_features(self, query: str) -> dict[str, float]:
        """Extract features from query for policy."""
        return {
            "length": min(1.0, len(query) / 500),
            "complexity": min(1.0, len(query.split()) / 100),
            "has_math": 1.0 if any(c in query for c in "+-×÷=") else 0.0,
        }

    async def allocate_budget(
        self, query: str, budget: int
    ) -> dict[str, int]:
        """Allocate budget using policy weights."""
        if not self._initialized:
            raise RuntimeError("RouterR1 must be initialized")

        # Distribute budget according to policy
        allocation = {}
        total_weight = sum(self._policy.values())
        
        for method, weight in self._policy.items():
            allocation[method] = int(budget * (weight / total_weight))
        
        return allocation

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["RouterR1", "ROUTER_R1_METADATA"]
