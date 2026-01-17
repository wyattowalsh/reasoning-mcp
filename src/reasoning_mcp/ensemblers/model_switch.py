"""ModelSwitch ensembler.

Multi-LLM repeated sampling with switching.

Reference: 2025 - "ModelSwitch"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.ensemblers.base import EnsemblerMetadata
from reasoning_mcp.models.core import EnsemblerIdentifier

MODEL_SWITCH_METADATA = EnsemblerMetadata(
    identifier=EnsemblerIdentifier.MODEL_SWITCH,
    name="ModelSwitch",
    description="Multi-LLM repeated sampling with dynamic switching.",
    tags=frozenset({"switching", "sampling", "multi-llm", "dynamic"}),
    complexity=5,
    min_models=2,
    max_models=5,
    supports_weighted_voting=True,
    supports_dynamic_selection=True,
    best_for=("diverse sampling", "exploration"),
    not_recommended_for=("single model scenarios",),
)


class ModelSwitch:
    """ModelSwitch ensembler implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._current_model_idx: int = 0
        self._samples_per_model: int = 3
        self._switch_threshold: float = 0.7

    @property
    def identifier(self) -> str:
        return EnsemblerIdentifier.MODEL_SWITCH

    @property
    def name(self) -> str:
        return MODEL_SWITCH_METADATA.name

    @property
    def description(self) -> str:
        return MODEL_SWITCH_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        self._current_model_idx = 0
        self._samples_per_model = 3
        self._switch_threshold = 0.7

    async def ensemble(
        self,
        query: str,
        solutions: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Combine solutions from switched sampling."""
        if not self._initialized:
            raise RuntimeError("ModelSwitch must be initialized")

        if not solutions:
            return ""

        # Score all solutions
        scored = [(s, self._score_solution(s)) for s in solutions]

        # Check if we should switch (no good solutions)
        best_score = max(s[1] for s in scored)
        if best_score < self._switch_threshold:
            # Would trigger model switch in real scenario
            pass

        # Return best solution
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _score_solution(self, solution: str) -> float:
        """Score a solution."""
        score = 0.5
        if len(solution) > 15:
            score += 0.2
        if any(c.isdigit() for c in solution):
            score += 0.15
        if "=" in solution:
            score += 0.1
        return min(1.0, score)

    async def sample_switch(
        self,
        query: str,
        model_outputs: dict[str, list[str]],
        n_samples: int = 3,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Sample from models with switching strategy."""
        if not self._initialized:
            raise RuntimeError("ModelSwitch must be initialized")

        all_solutions = []
        model_scores: dict[str, float] = {}

        for model_id, outputs in model_outputs.items():
            samples = outputs[:n_samples]
            scores = [self._score_solution(s) for s in samples]
            model_scores[model_id] = sum(scores) / len(scores) if scores else 0
            all_solutions.extend(samples)

        # Switch to best-performing model's outputs
        best_model = (
            max(model_scores, key=lambda k: model_scores.get(k, 0.0)) if model_scores else None
        )

        if best_model and best_model in model_outputs:
            candidates = model_outputs[best_model]
        else:
            candidates = all_solutions

        # Return best from candidates
        if candidates:
            scored = [(s, self._score_solution(s)) for s in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        return ""

    async def select_models(
        self,
        query: str,
        available_models: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Select models for switching ensemble."""
        if not self._initialized:
            raise RuntimeError("ModelSwitch must be initialized")

        return available_models[: MODEL_SWITCH_METADATA.max_models]

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ModelSwitch", "MODEL_SWITCH_METADATA"]
