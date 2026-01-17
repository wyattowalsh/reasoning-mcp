"""RAGRouter - Retrieval-Aware Routing.

Routes queries with retrieval-augmented awareness.

Reference: 2025 - "RAG-Aware Routing"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.models.core import MethodIdentifier, RouterIdentifier
from reasoning_mcp.routers.base import RouterMetadata

RAG_ROUTER_METADATA = RouterMetadata(
    identifier=RouterIdentifier.RAG_ROUTER,
    name="RAGRouter",
    description="Retrieval-aware routing for knowledge-intensive tasks.",
    tags=frozenset({"rag", "retrieval", "knowledge", "routing"}),
    complexity=5,
    supports_budget_control=True,
    supports_multi_model=True,
    best_for=("knowledge-intensive tasks", "document QA"),
    not_recommended_for=("pure reasoning tasks",),
)


class RagRouter:
    """RAGRouter implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._retrieval_threshold: float = 0.5

    @property
    def identifier(self) -> str:
        return RouterIdentifier.RAG_ROUTER

    @property
    def name(self) -> str:
        return RAG_ROUTER_METADATA.name

    @property
    def description(self) -> str:
        return RAG_ROUTER_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._retrieval_threshold = 0.5

    async def route(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Route based on retrieval needs."""
        if not self._initialized:
            raise RuntimeError("RAGRouter must be initialized")

        # Assess retrieval need
        retrieval_score = self._assess_retrieval_need(query)

        if retrieval_score > self._retrieval_threshold:
            # High retrieval need - use methods good with context
            return MethodIdentifier.CHAIN_OF_THOUGHT
        else:
            # Pure reasoning
            return MethodIdentifier.TREE_OF_THOUGHTS

    def _assess_retrieval_need(self, query: str) -> float:
        """Assess how much retrieval the query needs."""
        query_lower = query.lower()

        retrieval_indicators = [
            "what is",
            "who is",
            "when did",
            "where is",
            "define",
            "explain",
            "describe",
            "tell me about",
            "fact",
            "history",
            "information",
        ]

        reasoning_indicators = [
            "calculate",
            "solve",
            "prove",
            "derive",
            "if",
            "then",
            "why",
            "how would",
        ]

        retrieval_score = sum(
            1 for indicator in retrieval_indicators if indicator in query_lower
        ) / len(retrieval_indicators)

        reasoning_score = sum(
            1 for indicator in reasoning_indicators if indicator in query_lower
        ) / len(reasoning_indicators)

        # Balance retrieval vs reasoning
        if retrieval_score > reasoning_score:
            return 0.7
        elif reasoning_score > retrieval_score:
            return 0.3
        return 0.5

    async def allocate_budget(self, query: str, budget: int) -> dict[str, int]:
        """Allocate budget with retrieval awareness."""
        if not self._initialized:
            raise RuntimeError("RAGRouter must be initialized")

        retrieval_score = self._assess_retrieval_need(query)
        method = await self.route(query)

        # Adjust budget based on retrieval needs
        retrieval_budget = int(budget * retrieval_score * 0.3)
        reasoning_budget = budget - retrieval_budget

        return {
            method: reasoning_budget,
            "retrieval": retrieval_budget,
        }

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["RagRouter", "RAG_ROUTER_METADATA"]
