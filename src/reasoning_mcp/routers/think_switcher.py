"""ThinkSwitcher router.

Fast/Normal/Slow mode selection for compute efficiency.

Reference: 2025 - "ThinkSwitcher"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.base import RouterMetadata

THINK_SWITCHER_METADATA = RouterMetadata(
    identifier=RouterIdentifier.THINK_SWITCHER,
    name="ThinkSwitcher",
    description="Fast/Normal/Slow mode selection for compute efficiency.",
    tags=frozenset({"mode-selection", "efficiency", "adaptive", "speed"}),
    complexity=4,
    supports_budget_control=True,
    supports_multi_model=False,
    best_for=("latency optimization", "resource management"),
    not_recommended_for=("fixed-mode tasks",),
)


class ThinkSwitcher:
    """ThinkSwitcher router implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._current_mode: str = "normal"

    @property
    def identifier(self) -> str:
        return RouterIdentifier.THINK_SWITCHER

    @property
    def name(self) -> str:
        return THINK_SWITCHER_METADATA.name

    @property
    def description(self) -> str:
        return THINK_SWITCHER_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._current_mode = "normal"

    async def route(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Route to method based on selected mode."""
        if not self._initialized:
            raise RuntimeError("ThinkSwitcher must be initialized")

        # Determine mode from query complexity
        mode = self._select_mode(query)
        self._current_mode = mode

        mode_to_method = {
            "fast": MethodIdentifier.ZERO_SHOT_COT,
            "normal": MethodIdentifier.CHAIN_OF_THOUGHT,
            "slow": MethodIdentifier.TREE_OF_THOUGHTS,
        }
        return mode_to_method[mode]

    def _select_mode(self, query: str) -> str:
        """Select thinking mode based on query."""
        query_len = len(query)

        if query_len < 50:
            return "fast"
        elif query_len < 200:
            return "normal"
        else:
            return "slow"

    async def allocate_budget(self, query: str, budget: int) -> dict[str, int]:
        """Allocate budget based on mode."""
        if not self._initialized:
            raise RuntimeError("ThinkSwitcher must be initialized")

        mode = self._select_mode(query)
        method = await self.route(query)

        # Mode-based budget allocation
        mode_multipliers = {"fast": 0.3, "normal": 0.6, "slow": 1.0}
        allocated = int(budget * mode_multipliers[mode])

        return {method: allocated}

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ThinkSwitcher", "THINK_SWITCHER_METADATA"]
