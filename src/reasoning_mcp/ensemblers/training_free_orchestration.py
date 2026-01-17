"""Training-Free Orchestration ensembler.

Central controller routing without training.

Reference: 2025 - "Training-Free LLM Orchestration"
"""

from __future__ import annotations

from typing import Any

from reasoning_mcp.ensemblers.base import EnsemblerMetadata
from reasoning_mcp.models.core import EnsemblerIdentifier

TRAINING_FREE_ORCHESTRATION_METADATA = EnsemblerMetadata(
    identifier=EnsemblerIdentifier.TRAINING_FREE_ORCHESTRATION,
    name="Training-Free Orchestration",
    description="Central controller routing without additional training.",
    tags=frozenset({"training-free", "orchestration", "controller", "routing"}),
    complexity=4,
    min_models=2,
    max_models=8,
    supports_weighted_voting=False,
    supports_dynamic_selection=True,
    best_for=("quick deployment", "zero-shot orchestration"),
    not_recommended_for=("fine-tuned scenarios",),
)


class TrainingFreeOrchestration:
    """Training-Free Orchestration ensembler implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._specialist_registry: dict[str, list[str]] = {}

    @property
    def identifier(self) -> str:
        return EnsemblerIdentifier.TRAINING_FREE_ORCHESTRATION

    @property
    def name(self) -> str:
        return TRAINING_FREE_ORCHESTRATION_METADATA.name

    @property
    def description(self) -> str:
        return TRAINING_FREE_ORCHESTRATION_METADATA.description

    async def initialize(self) -> None:
        self._initialized = True
        # Default specialist capabilities
        self._specialist_registry = {
            "math": ["calculator", "solver"],
            "code": ["coder", "debugger"],
            "reasoning": ["reasoner", "analyzer"],
            "knowledge": ["retriever", "qa"],
        }

    async def ensemble(
        self,
        query: str,
        solutions: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Orchestrate solutions without training."""
        if not self._initialized:
            raise RuntimeError("Training-Free Orchestration must be initialized")

        if not solutions:
            return ""

        # Simple rule-based orchestration
        task_type = self._classify_task(query)

        # Score solutions based on task type
        scored = []
        for solution in solutions:
            score = self._score_for_task(solution, task_type)
            scored.append((solution, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _classify_task(self, query: str) -> str:
        """Classify task type from query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["calculate", "solve", "equation", "math"]):
            return "math"
        elif any(word in query_lower for word in ["code", "function", "program", "bug"]):
            return "code"
        elif any(word in query_lower for word in ["why", "how", "explain", "reason"]):
            return "reasoning"
        else:
            return "knowledge"

    def _score_for_task(self, solution: str, task_type: str) -> float:
        """Score solution for task type."""
        score = 0.5
        solution_lower = solution.lower()

        if task_type == "math":
            if any(c.isdigit() for c in solution):
                score += 0.2
            if "=" in solution:
                score += 0.2
        elif task_type == "code":
            if any(kw in solution_lower for kw in ["def ", "return", "function"]):
                score += 0.3
        elif task_type == "reasoning":
            if any(kw in solution_lower for kw in ["because", "therefore", "thus"]):
                score += 0.3
        else:
            if len(solution) > 50:
                score += 0.2

        return min(1.0, score)

    async def orchestrate(
        self,
        query: str,
        specialist_outputs: dict[str, str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Orchestrate specialist outputs."""
        if not self._initialized:
            raise RuntimeError("Training-Free Orchestration must be initialized")

        task_type = self._classify_task(query)

        # Find best specialist for task
        best_output = ""
        best_score = 0.0

        for specialist, output in specialist_outputs.items():
            # Check if specialist matches task
            relevance = 0.5
            for category, specialists in self._specialist_registry.items():
                if specialist in specialists and category == task_type:
                    relevance = 1.0
                    break

            score = self._score_for_task(output, task_type) * relevance
            if score > best_score:
                best_score = score
                best_output = output

        return best_output or list(specialist_outputs.values())[0] if specialist_outputs else ""

    async def select_models(
        self,
        query: str,
        available_models: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Select specialists for the query."""
        if not self._initialized:
            raise RuntimeError("Training-Free Orchestration must be initialized")

        task_type = self._classify_task(query)

        # Prioritize relevant specialists
        relevant = self._specialist_registry.get(task_type, [])
        selected = [m for m in available_models if m in relevant]

        # Add others up to max
        for m in available_models:
            if m not in selected:
                selected.append(m)
            if len(selected) >= TRAINING_FREE_ORCHESTRATION_METADATA.max_models:
                break

        return selected

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["TrainingFreeOrchestration", "TRAINING_FREE_ORCHESTRATION_METADATA"]
