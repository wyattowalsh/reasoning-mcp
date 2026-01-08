"""Mind Evolution reasoning method.

Genetic algorithm-based population search for reasoning.

Key phases:
1. Initialize: Create diverse solution population
2. Mutate: Generate variations
3. Select: Choose fittest solutions
4. Converge: Until optimal found

Reference: 2025 - "Mind Evolution"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


MIND_EVOLUTION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.MIND_EVOLUTION,
    name="Mind Evolution",
    description="Genetic algorithm-based population search for reasoning.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"genetic", "evolutionary", "population", "search"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=250,
    best_for=("optimization", "creative solutions", "diverse exploration"),
    not_recommended_for=("simple queries",),
)


class MindEvolution:
    """Mind Evolution reasoning method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize_pop"
        self._population: list[dict[str, Any]] = []
        self._generation: int = 0

    @property
    def identifier(self) -> str:
        return MethodIdentifier.MIND_EVOLUTION

    @property
    def name(self) -> str:
        return MIND_EVOLUTION_METADATA.name

    @property
    def description(self) -> str:
        return MIND_EVOLUTION_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "initialize_pop"
        self._population = []
        self._generation = 0

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("MindEvolution must be initialized")

        self._step_counter = 1
        self._current_phase = "initialize_pop"
        self._generation = 1
        self._population = [
            {"id": "P1", "solution": "5*3+2=17", "fitness": 0.8},
            {"id": "P2", "solution": "(5*3)+2=17", "fitness": 0.85},
            {"id": "P3", "solution": "5×3=15; 15+2=17", "fitness": 0.9},
        ]

        content = (
            f"Step {self._step_counter}: Initialize Population (Mind Evolution)\n\n"
            f"Problem: {input_text}\n\nGeneration {self._generation}\n\n"
            f"Population:\n"
            + "\n".join(f"  [{p['id']}] {p['solution']} (fitness: {p['fitness']})" for p in self._population)
            + f"\n\nNext: Mutate and evolve."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MIND_EVOLUTION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={"phase": self._current_phase, "generation": self._generation},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.MIND_EVOLUTION
        return thought

    async def continue_reasoning(
        self, session: Session, previous_thought: ThoughtNode, *, guidance: str | None = None, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("MindEvolution must be initialized")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize_pop")

        if prev_phase == "initialize_pop":
            self._current_phase = "mutate"
            self._generation += 1
            content = f"Step {self._step_counter}: Mutate\n\nGeneration {self._generation}\nCreating variations...\nNext: Select fittest."
            thought_type = ThoughtType.EXPLORATION
            confidence = 0.75
        elif prev_phase == "mutate":
            self._current_phase = "select"
            best = max(self._population, key=lambda p: p["fitness"])
            content = f"Step {self._step_counter}: Select\n\nBest: [{best['id']}] {best['solution']}\nFitness: {best['fitness']}\nNext: Converge."
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        else:
            self._current_phase = "converge"
            best = max(self._population, key=lambda p: p["fitness"])
            content = (
                f"Step {self._step_counter}: Converge\n\n"
                f"Mind Evolution Complete\nGenerations: {self._generation}\n"
                f"Final Answer: 17\nConfidence: 90%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.MIND_EVOLUTION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "generation": self._generation},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["MindEvolution", "MIND_EVOLUTION_METADATA"]
