"""EMAFusion ensembler.

Self-optimizing LLM integration with exponential moving average.

Reference: 2025 - "EMAFusion"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.ensemblers.base import EnsemblerBase, EnsemblerMetadata
from reasoning_mcp.models.core import EnsemblerIdentifier


EMA_FUSION_METADATA = EnsemblerMetadata(
    identifier=EnsemblerIdentifier.EMA_FUSION,
    name="EMAFusion",
    description="Self-optimizing LLM integration with EMA weighting.",
    tags=frozenset({"ema", "fusion", "self-optimizing", "adaptive"}),
    complexity=5,
    min_models=2,
    max_models=6,
    supports_weighted_voting=True,
    supports_dynamic_selection=True,
    best_for=("adaptive ensemble", "online learning"),
    not_recommended_for=("static ensemble needs",),
)


class EmaFusion:
    """EMAFusion ensembler implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._model_weights: dict[str, float] = {}
        self._ema_alpha: float = 0.3
        self._performance_history: dict[str, list[float]] = {}

    @property
    def identifier(self) -> str:
        return EnsemblerIdentifier.EMA_FUSION

    @property
    def name(self) -> str:
        return EMA_FUSION_METADATA.name

    @property
    def description(self) -> str:
        return EMA_FUSION_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._model_weights = {}
        self._ema_alpha = 0.3
        self._performance_history = {}

    async def ensemble(
        self,
        query: str,
        solutions: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Combine solutions using EMA-weighted fusion."""
        if not self._initialized:
            raise RuntimeError("EMAFusion must be initialized")

        if not solutions:
            return ""

        # Initialize weights if needed
        if not self._model_weights:
            for i in range(len(solutions)):
                self._model_weights[f"model_{i}"] = 1.0 / len(solutions)

        # Score and weight solutions
        scored = []
        for i, solution in enumerate(solutions):
            model_id = f"model_{i}"
            weight = self._model_weights.get(model_id, 1.0 / len(solutions))
            quality = self._assess_quality(solution)
            scored.append((solution, weight * quality))
        
        # Select best weighted solution
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _assess_quality(self, solution: str) -> float:
        """Assess solution quality."""
        score = 0.6
        if len(solution) > 20:
            score += 0.15
        if any(c.isdigit() for c in solution):
            score += 0.1
        if "=" in solution:
            score += 0.1
        return min(1.0, score)

    async def update_weights(
        self,
        model_performances: dict[str, float],
    ) -> None:
        """Update model weights using EMA."""
        if not self._initialized:
            raise RuntimeError("EMAFusion must be initialized")

        for model_id, performance in model_performances.items():
            # EMA update
            old_weight = self._model_weights.get(model_id, 0.5)
            new_weight = (
                self._ema_alpha * performance +
                (1 - self._ema_alpha) * old_weight
            )
            self._model_weights[model_id] = new_weight
            
            # Track history
            if model_id not in self._performance_history:
                self._performance_history[model_id] = []
            self._performance_history[model_id].append(performance)

    async def fuse_adaptive(
        self,
        query: str,
        model_outputs: dict[str, str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Adaptively fuse model outputs."""
        if not self._initialized:
            raise RuntimeError("EMAFusion must be initialized")

        solutions = list(model_outputs.values())
        return await self.ensemble(query, solutions, context)

    async def select_models(
        self,
        query: str,
        available_models: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Select models based on EMA weights."""
        if not self._initialized:
            raise RuntimeError("EMAFusion must be initialized")

        # Sort by weight and select top models
        weighted = [
            (model, self._model_weights.get(model, 0.5))
            for model in available_models
        ]
        weighted.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in weighted[:EMA_FUSION_METADATA.max_models]]

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["EmaFusion", "EMA_FUSION_METADATA"]
