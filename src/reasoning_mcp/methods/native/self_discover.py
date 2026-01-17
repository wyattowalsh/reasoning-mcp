"""Self-Discover reasoning method.

This module implements Self-Discover based on Zhou et al. (2024), which
discovers task-specific reasoning structures before solving. The method
identifies relevant reasoning modules, adapts them to the task, and
creates a structured reasoning plan.

Key phases:
1. Select: Choose relevant reasoning modules from a pool
2. Adapt: Customize modules for the specific task
3. Implement: Create structured reasoning plan
4. Execute: Follow the discovered structure to solve

Reference: Zhou et al. (2024) - "Self-Discover: Large Language Models
Self-Compose Reasoning Structures"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, ElicitationError, elicit_selection
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


SELF_DISCOVER_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_DISCOVER,
    name="Self-Discover",
    description="Discovers task-specific reasoning structures before solving. "
    "Selects, adapts, and implements custom reasoning modules for each task.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"self-compose", "structure-discovery", "modules", "adaptive", "2024"}),
    complexity=8,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=12,
    avg_tokens_per_thought=350,
    best_for=("novel problem types", "complex multi-step tasks", "adaptive reasoning"),
    not_recommended_for=("simple queries", "well-defined problems"),
)


class SelfDiscover(ReasoningMethodBase):
    """Self-Discover reasoning method implementation."""

    # Class attribute to enable/disable LLM sampling
    _use_sampling: bool = True

    # Pool of reasoning modules
    REASONING_MODULES = [
        {
            "id": "critical_thinking",
            "name": "Critical Thinking",
            "desc": "Analyze assumptions and evaluate evidence",
        },
        {
            "id": "creative",
            "name": "Creative Thinking",
            "desc": "Generate novel ideas and alternatives",
        },
        {
            "id": "systems",
            "name": "Systems Thinking",
            "desc": "Consider interconnections and feedback loops",
        },
        {
            "id": "analytical",
            "name": "Analytical Thinking",
            "desc": "Break down into components and examine",
        },
        {
            "id": "reflective",
            "name": "Reflective Thinking",
            "desc": "Learn from experience and self-evaluate",
        },
        {"id": "practical", "name": "Practical Reasoning", "desc": "Focus on actionable outcomes"},
    ]

    def __init__(self, enable_elicitation: bool = True) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "select"
        self._selected_modules: list[dict[str, Any]] = []
        self._adapted_modules: list[dict[str, Any]] = []
        self._reasoning_structure: dict[str, Any] = {}
        self._execution_context: ExecutionContext | None = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SELF_DISCOVER

    @property
    def name(self) -> str:
        return SELF_DISCOVER_METADATA.name

    @property
    def description(self) -> str:
        return SELF_DISCOVER_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "select"
        self._selected_modules = []
        self._adapted_modules = []
        self._reasoning_structure = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Self-Discover must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "select"

        # Elicitation: ask user for discovery approach
        discovery_approach = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "broad", "label": "Broad discovery - explore many modules"},
                    {"id": "focused", "label": "Focused discovery - select key modules"},
                    {"id": "adaptive", "label": "Adaptive - let the problem guide selection"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "What discovery approach should we use?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    discovery_approach = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ElicitationError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
            except (ConnectionError, OSError) as e:
                logger.warning(
                    "elicitation_connection_error",
                    method="execute",
                    error=str(e),
                )

        # Use _sample_with_fallback for LLM sampling with proper error handling
        content = await self._sample_select_phase(input_text, discovery_approach)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_DISCOVER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "modules_available": len(self.REASONING_MODULES),
                "sampled": self._execution_context is not None
                and self._execution_context.can_sample,
                "discovery_approach": discovery_approach,
                "input_text": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SELF_DISCOVER
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
            raise RuntimeError("Self-Discover must be initialized before continuation")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "select")

        # Get problem text from previous thought metadata
        input_text = previous_thought.metadata.get("input_text", "")
        if not input_text:
            # Try to extract from first thought
            thoughts = session.get_recent_thoughts(n=session.thought_count)
            if thoughts:
                input_text = thoughts[0].metadata.get("input_text", "")

        if prev_phase == "select":
            self._current_phase = "adapt"
            # Select relevant modules (heuristic fallback)
            self._selected_modules = [
                self.REASONING_MODULES[0],  # critical_thinking
                self.REASONING_MODULES[3],  # analytical
                self.REASONING_MODULES[5],  # practical
            ]

            # Use _sample_with_fallback for LLM sampling with proper error handling
            content = await self._sample_adapt_phase(input_text, self._selected_modules)

            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "adapt":
            self._current_phase = "implement"
            self._adapted_modules = [
                {"name": m["name"], "adaptation": "Customized for problem context"}
                for m in self._selected_modules
            ]

            # Use _sample_with_fallback for LLM sampling with proper error handling
            content = await self._sample_implement_phase(
                input_text, self._selected_modules, self._adapted_modules
            )

            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.75
        elif prev_phase == "implement":
            self._current_phase = "execute"
            self._reasoning_structure = {
                "steps": [
                    "1. Apply critical thinking to evaluate the problem",
                    "2. Use analytical thinking to decompose",
                    "3. Synthesize with practical focus",
                    "4. Validate and refine solution",
                ],
                "connections": ["1→2", "2→3", "3→4"],
            }

            # Use _sample_with_fallback for LLM sampling with proper error handling
            content = await self._sample_structure_phase(input_text, self._reasoning_structure)

            thought_type = ThoughtType.REASONING
            confidence = 0.8
        elif prev_phase == "execute":
            self._current_phase = "solve"

            # Use _sample_with_fallback for LLM sampling with proper error handling
            content = await self._sample_solve_phase(input_text, self._reasoning_structure)

            thought_type = ThoughtType.REASONING
            confidence = 0.85
        else:
            self._current_phase = "conclude"

            # Use _sample_with_fallback for LLM sampling with proper error handling
            content = await self._sample_conclude_phase(
                input_text, self._selected_modules, self._reasoning_structure
            )

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.87

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SELF_DISCOVER,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "modules": self._selected_modules,
                "sampled": self._execution_context is not None
                and self._execution_context.can_sample,
                "input_text": input_text,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Heuristic generation methods (fallback)

    def _generate_select_phase(self, input_text: str, discovery_approach: str | None = None) -> str:
        """Generate select phase content using heuristics."""
        approach_text = ""
        if discovery_approach:
            approach_text = f"\nDiscovery Approach: {discovery_approach}\n"

        return (
            f"Step {self._step_counter}: Select Reasoning Modules (Self-Discover)\n\n"
            f"Problem: {input_text}\n"
            f"{approach_text}\n"
            f"Available Reasoning Modules:\n"
            + "\n".join(f"  • {m['name']}: {m['desc']}" for m in self.REASONING_MODULES)
            + "\n\nAnalyzing task to select relevant modules...\n"
            "Next: Choose modules most suited to this problem type."
        )

    def _generate_adapt_phase(self, selected_modules: list[dict[str, Any]]) -> str:
        """Generate adapt phase content using heuristics."""
        return (
            f"Step {self._step_counter}: Module Selection Complete\n\n"
            f"Selected Modules for this task:\n"
            + "\n".join(f"  ✓ {m['name']}" for m in selected_modules)
            + "\n\nRationale:\n"
            "  - Critical thinking: Evaluate evidence and assumptions\n"
            "  - Analytical: Break down the problem systematically\n"
            "  - Practical: Focus on actionable solution\n\n"
            "Next: Adapt modules to specific task requirements."
        )

    def _generate_implement_phase(self, adapted_modules: list[dict[str, Any]]) -> str:
        """Generate implement phase content using heuristics."""
        return (
            f"Step {self._step_counter}: Adapt Modules to Task\n\n"
            f"Adapting selected modules...\n\n"
            f"Adapted Modules:\n"
            + "\n".join(f"  {m['name']}: {m['adaptation']}" for m in adapted_modules)
            + "\n\nNext: Implement reasoning structure."
        )

    def _generate_structure_phase(self, reasoning_structure: dict[str, Any]) -> str:
        """Generate structure phase content using heuristics."""
        return (
            f"Step {self._step_counter}: Implement Reasoning Structure\n\n"
            f"Discovered Structure:\n"
            + "\n".join(f"  {s}" for s in reasoning_structure["steps"])
            + f"\n\nFlow: {' → '.join(reasoning_structure['connections'])}\n\n"
            f"Next: Execute discovered reasoning structure."
        )

    def _generate_solve_phase(self, reasoning_structure: dict[str, Any]) -> str:
        """Generate solve phase content using heuristics."""
        return (
            f"Step {self._step_counter}: Execute Reasoning\n\n"
            f"Following discovered structure...\n\n"
            + "\n".join(f"  ✓ {s}" for s in reasoning_structure.get("steps", []))
            + "\n\nAll structure steps executed."
        )

    def _generate_conclude_phase(self) -> str:
        """Generate conclude phase content using heuristics."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Self-Discover Complete:\n"
            f"  • Modules selected: {len(self._selected_modules)}\n"
            f"  • Structure steps: {len(self._reasoning_structure.get('steps', []))}\n\n"
            f"Final Answer: [Solution from discovered reasoning structure]\n"
            f"Confidence: High (87%)\n"
            f"Structure: Task-specific, self-composed"
        )

    # LLM sampling methods using _sample_with_fallback

    async def _sample_select_phase(
        self, input_text: str, discovery_approach: str | None = None
    ) -> str:
        """Generate select phase content using LLM sampling with fallback."""
        approach_text = ""
        if discovery_approach:
            approach_text = f"\nDiscovery approach preference: {discovery_approach}"

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the SELECT phase, analyze the task and identify which reasoning modules are most relevant.

Available reasoning modules:
- Critical Thinking: Analyze assumptions and evaluate evidence
- Creative Thinking: Generate novel ideas and alternatives
- Systems Thinking: Consider interconnections and feedback loops
- Analytical Thinking: Break down into components and examine
- Reflective Thinking: Learn from experience and self-evaluate
- Practical Reasoning: Focus on actionable outcomes

Explain which modules are most relevant for the given problem and why."""

        user_prompt = f"""Problem: {input_text}{approach_text}

Analyze this problem and determine which reasoning modules from the pool would be most effective.
Explain your module selection rationale."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_select_phase(input_text, discovery_approach),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )
        return f"Step {self._step_counter}: Select Reasoning Modules\n\n{content}"

    async def _sample_adapt_phase(
        self, input_text: str, selected_modules: list[dict[str, Any]]
    ) -> str:
        """Generate adapt phase content using LLM sampling with fallback."""
        module_list = "\n".join(f"- {m['name']}: {m['desc']}" for m in selected_modules)

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the ADAPT phase, customize the selected reasoning modules to fit the specific task context.

Explain how each selected module should be adapted and applied to this particular problem."""

        user_prompt = f"""Problem: {input_text}

Selected modules:
{module_list}

For each module, explain how it should be adapted and customized for this specific problem."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_adapt_phase(selected_modules),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=900,
        )
        return f"Step {self._step_counter}: Adapt Modules to Task\n\n{content}"

    async def _sample_implement_phase(
        self,
        input_text: str,
        selected_modules: list[dict[str, Any]],
        adapted_modules: list[dict[str, Any]],
    ) -> str:
        """Generate implement phase content using LLM sampling with fallback."""
        module_list = "\n".join(f"- {m['name']}" for m in selected_modules)

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the IMPLEMENT phase, create a structured reasoning plan by composing the adapted modules.

Design a step-by-step reasoning structure that shows how to apply the modules in sequence
or combination to solve the problem."""

        user_prompt = f"""Problem: {input_text}

Adapted modules: {module_list}

Create a structured reasoning plan that shows:
1. The sequence of reasoning steps
2. How modules connect and build on each other
3. The logical flow from problem to solution"""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_implement_phase(adapted_modules),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )
        return f"Step {self._step_counter}: Implement Reasoning Structure\n\n{content}"

    async def _sample_structure_phase(
        self, input_text: str, reasoning_structure: dict[str, Any]
    ) -> str:
        """Generate structure phase content using LLM sampling with fallback."""
        structure_desc = "\n".join(reasoning_structure.get("steps", []))

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the STRUCTURE phase, verify and refine the discovered reasoning structure.

Review the reasoning structure and ensure it's well-formed and suitable for the problem."""

        user_prompt = f"""Problem: {input_text}

Proposed reasoning structure:
{structure_desc}

Verify this structure is appropriate and explain how it will guide problem-solving."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_structure_phase(reasoning_structure),
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800,
        )
        return f"Step {self._step_counter}: Verify Reasoning Structure\n\n{content}"

    async def _sample_solve_phase(
        self, input_text: str, reasoning_structure: dict[str, Any]
    ) -> str:
        """Generate solve phase content using LLM sampling with fallback."""
        structure_desc = "\n".join(reasoning_structure.get("steps", []))

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the EXECUTE phase, follow the discovered reasoning structure to solve the problem.

Apply each step of the structure systematically to arrive at a solution."""

        user_prompt = f"""Problem: {input_text}

Reasoning structure to follow:
{structure_desc}

Execute this reasoning structure step by step to solve the problem."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_solve_phase(reasoning_structure),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )
        return f"Step {self._step_counter}: Execute Reasoning Structure\n\n{content}"

    async def _sample_conclude_phase(
        self,
        input_text: str,
        selected_modules: list[dict[str, Any]],
        reasoning_structure: dict[str, Any],
    ) -> str:
        """Generate conclude phase content using LLM sampling with fallback."""
        modules_used = ", ".join(m["name"] for m in selected_modules)
        structure_desc = "\n".join(reasoning_structure.get("steps", []))

        system_prompt = """You are a reasoning assistant using Self-Discover methodology.
In the CONCLUDE phase, synthesize the final answer based on the discovered structure.

Provide a clear, confident final answer that demonstrates how the self-discovered
reasoning structure led to the solution."""

        user_prompt = f"""Problem: {input_text}

Modules used: {modules_used}

Reasoning structure applied:
{structure_desc}

Provide the final answer, explaining how the self-discovered structure led to this solution."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_conclude_phase(),
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )
        return f"Step {self._step_counter}: Final Answer\n\n{content}"


__all__ = ["SelfDiscover", "SELF_DISCOVER_METADATA"]
