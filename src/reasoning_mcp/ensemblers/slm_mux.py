"""SLM-MUX ensembler.

Small Language Model multiplexing and orchestration.

Reference: 2025 - "SLM-MUX"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.ensemblers.base import EnsemblerBase, EnsemblerMetadata
from reasoning_mcp.models.core import EnsemblerIdentifier


SLM_MUX_METADATA = EnsemblerMetadata(
    identifier=EnsemblerIdentifier.SLM_MUX,
    name="SLM-MUX",
    description="Small Language Model multiplexing for efficient ensemble.",
    tags=frozenset({"slm", "multiplexing", "efficient", "orchestration"}),
    complexity=5,
    min_models=3,
    max_models=10,
    supports_weighted_voting=True,
    supports_dynamic_selection=True,
    best_for=("resource-efficient ensemble", "edge deployment"),
    not_recommended_for=("single large model scenarios",),
)


class SlmMux:
    """SLM-MUX ensembler implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._active_slms: list[str] = []
        self._slm_weights: dict[str, float] = {}

    @property
    def identifier(self) -> str:
        return EnsemblerIdentifier.SLM_MUX

    @property
    def name(self) -> str:
        return SLM_MUX_METADATA.name

    @property
    def description(self) -> str:
        return SLM_MUX_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._active_slms = []
        self._slm_weights = {}

    async def ensemble(
        self,
        query: str,
        solutions: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Combine SLM outputs via multiplexing."""
        if not self._initialized:
            raise RuntimeError("SLM-MUX must be initialized")

        if not solutions:
            return ""

        # Weight and combine solutions
        weighted_solutions = []
        for i, solution in enumerate(solutions):
            weight = self._slm_weights.get(f"slm_{i}", 1.0 / len(solutions))
            weighted_solutions.append((solution, weight))
        
        # Select best solution by weight-adjusted scoring
        best_solution = self._weighted_selection(weighted_solutions)
        
        return best_solution

    def _weighted_selection(self, weighted_solutions: list[tuple[str, float]]) -> str:
        """Select best solution with weighting."""
        if not weighted_solutions:
            return ""
        
        scored = []
        for solution, weight in weighted_solutions:
            # Simple quality heuristic
            quality = 0.5
            if len(solution) > 10:
                quality += 0.2
            if any(c.isdigit() for c in solution):
                quality += 0.1
            if "=" in solution:
                quality += 0.1
            
            scored.append((solution, quality * weight))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    async def orchestrate_slms(
        self,
        query: str,
        slm_outputs: dict[str, str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Orchestrate multiple SLMs for optimal output."""
        if not self._initialized:
            raise RuntimeError("SLM-MUX must be initialized")

        # Update active SLMs
        self._active_slms = list(slm_outputs.keys())
        
        # Initialize equal weights if not set
        if not self._slm_weights:
            for slm in self._active_slms:
                self._slm_weights[slm] = 1.0 / len(self._active_slms)
        
        # Combine using multiplexing strategy
        solutions = list(slm_outputs.values())
        return await self.ensemble(query, solutions, context)

    async def select_models(
        self,
        query: str,
        available_models: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Select SLMs for the query."""
        if not self._initialized:
            raise RuntimeError("SLM-MUX must be initialized")

        # Select up to max_models SLMs
        return available_models[:SLM_MUX_METADATA.max_models]

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["SlmMux", "SLM_MUX_METADATA"]
