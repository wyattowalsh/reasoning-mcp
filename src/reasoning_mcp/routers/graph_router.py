"""GraphRouter router.

Graph-based model routing using reasoning graphs.

Reference: 2025 - "GraphRouter"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.models.core import RouterIdentifier, MethodIdentifier


GRAPH_ROUTER_METADATA = RouterMetadata(
    identifier=RouterIdentifier.GRAPH_ROUTER,
    name="GraphRouter",
    description="Graph-based model routing using reasoning graphs.",
    tags=frozenset({"graph", "structured", "routing", "reasoning-graph"}),
    complexity=7,
    supports_budget_control=True,
    supports_multi_model=True,
    best_for=("complex reasoning paths", "multi-step problems"),
    not_recommended_for=("simple queries",),
)


class GraphRouter:
    """GraphRouter implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._graph: dict[str, list[str]] = {}
        self._node_methods: dict[str, str] = {}

    @property
    def identifier(self) -> str:
        return RouterIdentifier.GRAPH_ROUTER

    @property
    def name(self) -> str:
        return GRAPH_ROUTER_METADATA.name

    @property
    def description(self) -> str:
        return GRAPH_ROUTER_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        # Initialize default reasoning graph
        self._graph = {
            "start": ["analyze", "decompose"],
            "analyze": ["reason", "verify"],
            "decompose": ["reason"],
            "reason": ["synthesize"],
            "synthesize": ["end"],
            "verify": ["end"],
        }
        self._node_methods = {
            "start": MethodIdentifier.CHAIN_OF_THOUGHT,
            "analyze": MethodIdentifier.CHAIN_OF_THOUGHT,
            "decompose": MethodIdentifier.TREE_OF_THOUGHTS,
            "reason": MethodIdentifier.SELF_CONSISTENCY,
            "synthesize": MethodIdentifier.CHAIN_OF_THOUGHT,
            "verify": MethodIdentifier.SELF_CONSISTENCY,
        }

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Route using graph-based analysis."""
        if not self._initialized:
            raise RuntimeError("GraphRouter must be initialized")

        # Determine optimal path through reasoning graph
        path = self._find_optimal_path(query)
        
        # Return method for first node in path
        if path:
            return self._node_methods.get(path[0], MethodIdentifier.CHAIN_OF_THOUGHT)
        return MethodIdentifier.CHAIN_OF_THOUGHT

    def _find_optimal_path(self, query: str) -> list[str]:
        """Find optimal path through reasoning graph."""
        # Simple heuristic: longer queries need more decomposition
        if len(query) > 200:
            return ["start", "decompose", "reason", "synthesize", "end"]
        elif len(query) > 100:
            return ["start", "analyze", "reason", "synthesize", "end"]
        else:
            return ["start", "analyze", "verify", "end"]

    async def allocate_budget(
        self, query: str, budget: int
    ) -> dict[str, int]:
        """Allocate budget across graph nodes."""
        if not self._initialized:
            raise RuntimeError("GraphRouter must be initialized")

        path = self._find_optimal_path(query)
        allocation = {}
        
        # Distribute budget across path nodes
        per_node = budget // len(path) if path else budget
        for node in path:
            if node in self._node_methods:
                method = self._node_methods[node]
                allocation[method] = allocation.get(method, 0) + per_node
        
        return allocation

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["GraphRouter", "GRAPH_ROUTER_METADATA"]
