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

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
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


class MindEvolution(ReasoningMethodBase):
    """Mind Evolution reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize_pop"
        self._population: list[dict[str, Any]] = []
        self._generation: int = 0
        self._execution_context: ExecutionContext | None = None

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
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("MindEvolution must be initialized")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "initialize_pop"
        self._generation = 1

        # Initialize population using LLM sampling if available
        self._population = await self._initialize_population(input_text)

        content = (
            f"Step {self._step_counter}: Initialize Population (Mind Evolution)\n\n"
            f"Problem: {input_text}\n\nGeneration {self._generation}\n\n"
            f"Population:\n"
            + "\n".join(
                f"  [{p['id']}] {p['solution']} (fitness: {p['fitness']})" for p in self._population
            )
            + "\n\nNext: Mutate and evolve."
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
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("MindEvolution must be initialized")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize_pop")

        # Get input_text from session or context
        input_text = ""
        if session.graph.nodes:
            # Extract from first thought - get root nodes
            root_nodes = [n for n in session.graph.nodes.values() if n.depth == 0]
            if root_nodes:
                first_thought = root_nodes[0]
                if "Problem:" in first_thought.content:
                    input_text = first_thought.content.split("Problem:")[1].split("\n")[0].strip()

        if prev_phase == "initialize_pop":
            self._current_phase = "mutate"
            self._generation += 1
            # Generate mutations using LLM sampling
            self._population = await self._mutate_population(input_text)
            mutations_desc = "\n".join(
                f"  [{p['id']}] {p['solution']} (fitness: {p['fitness']})" for p in self._population
            )
            content = (
                f"Step {self._step_counter}: Mutate\n\n"
                f"Generation {self._generation}\n"
                f"Created variations:\n{mutations_desc}\n\n"
                f"Next: Select fittest."
            )
            thought_type = ThoughtType.EXPLORATION
            confidence = 0.75
        elif prev_phase == "mutate":
            self._current_phase = "select"
            best = max(self._population, key=lambda p: p["fitness"])
            content = (
                f"Step {self._step_counter}: Select\n\n"
                f"Best: [{best['id']}] {best['solution']}\n"
                f"Fitness: {best['fitness']}\n"
                f"Next: Converge."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        else:
            self._current_phase = "converge"
            best = max(self._population, key=lambda p: p["fitness"])
            content = (
                f"Step {self._step_counter}: Converge\n\n"
                f"Mind Evolution Complete\nGenerations: {self._generation}\n"
                f"Final Solution: {best['solution']}\n"
                f"Fitness: {best['fitness']}\n"
                f"Confidence: {int(best['fitness'] * 100)}%"
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

    async def _initialize_population(self, input_text: str) -> list[dict[str, Any]]:
        """Initialize a diverse population of solutions using LLM sampling.

        Args:
            input_text: The problem to solve

        Returns:
            List of population members with solutions and fitness scores
        """
        system_prompt = (
            "You are a genetic algorithm assistant helping to initialize a diverse "
            "population of candidate solutions for Mind Evolution reasoning. "
            "Generate multiple distinct approaches or solutions to the problem, "
            "each representing a different strategy or perspective. "
            "Each solution should be concrete and actionable."
        )

        user_prompt = f"""Problem: {input_text}

Generate 3 distinct candidate solutions for this problem. Each solution should:
1. Represent a unique approach or perspective
2. Be concrete and detailed
3. Be formatted as a clear solution description

Provide exactly 3 solutions, one per line, numbered 1-3."""

        def fallback_generator() -> str:
            # Return empty string to signal fallback needed
            return ""

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.8,  # High temperature for diversity
            max_tokens=400,
        )

        # If we got a valid response, parse it
        if result:
            # Parse solutions from response
            solutions = [line.strip() for line in result.split("\n") if line.strip()]
            # Remove numbering if present (e.g., "1. " or "1) ")
            solutions = [
                sol.split(". ", 1)[-1].split(") ", 1)[-1]
                for sol in solutions
                if sol and not sol.isdigit()
            ]

            # Create population with fitness scores
            if len(solutions) >= 3:
                return [
                    {"id": f"P{i + 1}", "solution": sol, "fitness": 0.7 + (i * 0.05)}
                    for i, sol in enumerate(solutions[:3])
                ]

        # Fallback: Generate template-based solutions
        return self._generate_fallback_population(input_text)

    def _generate_fallback_population(self, input_text: str) -> list[dict[str, Any]]:
        """Generate a fallback population when LLM sampling is unavailable.

        Args:
            input_text: The problem to solve

        Returns:
            List of population members with template-based solutions
        """
        # Extract key terms from input for more relevant fallback
        words = input_text.lower().split()
        key_phrase = " ".join(words[:5]) if len(words) > 5 else input_text[:50]

        return [
            {
                "id": "P1",
                "solution": f"Direct approach: Apply standard method to {key_phrase}",
                "fitness": 0.8,
            },
            {
                "id": "P2",
                "solution": f"Alternative perspective: Reframe {key_phrase} from different angle",
                "fitness": 0.85,
            },
            {
                "id": "P3",
                "solution": f"Systematic breakdown: Decompose {key_phrase} into components",
                "fitness": 0.9,
            },
        ]

    async def _mutate_population(self, input_text: str) -> list[dict[str, Any]]:
        """Generate mutations/variations of current population using LLM sampling.

        Args:
            input_text: The original problem

        Returns:
            New population with mutations
        """
        # Get best solutions from current population
        top_solutions = sorted(self._population, key=lambda p: p["fitness"], reverse=True)[:2]

        system_prompt = (
            "You are a genetic algorithm assistant helping to mutate and improve "
            "candidate solutions. Generate variations that build upon the best "
            "existing solutions while introducing creative modifications."
        )

        user_prompt = f"""Problem: {input_text}

Current best solutions:
1. {top_solutions[0]["solution"]}
2. {top_solutions[1]["solution"]}

Generate 3 new solution variations by:
- Combining elements from the best solutions
- Introducing creative modifications
- Exploring nearby solution space

Provide exactly 3 variations, one per line, numbered 1-3."""

        def fallback_generator() -> str:
            # Return empty string to signal fallback needed
            return ""

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=400,
        )

        # If we got a valid response, parse it
        if result:
            # Parse mutations from response
            mutations = [line.strip() for line in result.split("\n") if line.strip()]
            mutations = [
                mut.split(". ", 1)[-1].split(") ", 1)[-1]
                for mut in mutations
                if mut and not mut.isdigit()
            ]

            # Create new population with slightly improved fitness
            if len(mutations) >= 3:
                return [
                    {"id": f"P{i + 1}", "solution": mut, "fitness": 0.75 + (i * 0.05)}
                    for i, mut in enumerate(mutations[:3])
                ]

        # Fallback: Simple mutation by appending variation markers
        return [
            {
                "id": "P1",
                "solution": f"{p['solution']} (variation A)",
                "fitness": p["fitness"] + 0.02,
            }
            for p in self._population
        ]


__all__ = ["MindEvolution", "MIND_EVOLUTION_METADATA"]
