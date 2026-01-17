"""Complexity-Based Prompting reasoning method.

This module implements Complexity-Based Prompting, which selects complex
examples for demonstrations. Research shows that using examples with longer
reasoning chains improves performance on complex tasks.

Key phases:
1. Measure: Assess complexity of available examples
2. Select: Choose examples with highest complexity
3. Demonstrate: Use complex examples as few-shot prompts
4. Reason: Apply CoT with complex demonstrations

Reference: Fu et al. (2023) - "Complexity-Based Prompting for Multi-Step
Reasoning"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


COMPLEXITY_BASED_METADATA = MethodMetadata(
    identifier=MethodIdentifier.COMPLEXITY_BASED,
    name="Complexity-Based Prompting",
    description="Uses complex examples with longer reasoning chains for demonstrations. "
    "More complex examples lead to better performance on multi-step reasoning.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"complexity", "few-shot", "examples", "multi-step", "selection"}),
    complexity=5,
    supports_branching=False,
    supports_revision=False,
    requires_context=True,
    min_thoughts=4,
    max_thoughts=7,
    avg_tokens_per_thought=300,
    best_for=("multi-step reasoning", "complex problems", "few-shot learning"),
    not_recommended_for=("simple tasks", "real-time responses"),
)


class ComplexityBased(ReasoningMethodBase):
    """Complexity-Based Prompting method implementation."""

    DEFAULT_EXAMPLES = 5
    DEFAULT_SELECTED = 3
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "measure"
        self._available_examples: list[dict[str, Any]] = []
        self._selected_examples: list[dict[str, Any]] = []
        self._execution_context: Any = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        return MethodIdentifier.COMPLEXITY_BASED

    @property
    def name(self) -> str:
        return COMPLEXITY_BASED_METADATA.name

    @property
    def description(self) -> str:
        return COMPLEXITY_BASED_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "measure"
        self._available_examples = []
        self._selected_examples = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Complexity-Based must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "measure"

        # Create examples with varying complexity (measured by reasoning steps)
        self._available_examples = [
            {"id": 1, "steps": 2, "complexity": "low"},
            {"id": 2, "steps": 5, "complexity": "medium"},
            {"id": 3, "steps": 8, "complexity": "high"},
            {"id": 4, "steps": 3, "complexity": "low"},
            {"id": 5, "steps": 7, "complexity": "high"},
        ]

        # Generate content with sampling or heuristic
        if self._use_sampling:
            content = await self._sample_complexity_analysis(input_text, context)
        else:
            content = self._generate_complexity_analysis_heuristic(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COMPLEXITY_BASED,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "examples": len(self._available_examples),
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.COMPLEXITY_BASED
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Complexity-Based must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "measure")

        if prev_phase == "measure":
            self._current_phase = "select"

            # Elicitation for complexity preference
            complexity_preference = "adaptive"  # default
            if (
                self.enable_elicitation
                and self._execution_context
                and hasattr(self._execution_context, "ctx")
                and self._execution_context.ctx
            ):
                try:
                    options = [
                        {"id": "high", "label": "High complexity - Most detailed examples"},
                        {"id": "medium", "label": "Medium complexity - Balanced examples"},
                        {"id": "adaptive", "label": "Adaptive - Match problem complexity"},
                    ]
                    config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                    selection = await elicit_selection(
                        self._execution_context.ctx,
                        "What complexity level for examples?",
                        options,
                        config=config,
                    )
                    if selection and selection.selected:
                        complexity_preference = selection.selected
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    # Log expected errors and fallback to default
                    logger.warning(
                        "elicitation_error",
                        method="continue_reasoning",
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    # Fallback to default
                except Exception as e:
                    # Log unexpected exceptions and re-raise to avoid masking bugs
                    logger.error(
                        "elicitation_unexpected_error",
                        method="continue_reasoning",
                        error_type=type(e).__name__,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            sorted_examples = sorted(
                self._available_examples, key=lambda x: x["steps"], reverse=True
            )

            # Filter examples based on complexity preference
            if complexity_preference == "high":
                filtered_examples = [e for e in sorted_examples if e["complexity"] == "high"]
                if len(filtered_examples) < self.DEFAULT_SELECTED:
                    filtered_examples = sorted_examples
            elif complexity_preference == "medium":
                filtered_examples = [
                    e for e in sorted_examples if e["complexity"] in ("medium", "high")
                ]
                if len(filtered_examples) < self.DEFAULT_SELECTED:
                    filtered_examples = sorted_examples
            else:  # adaptive
                filtered_examples = sorted_examples

            self._selected_examples = filtered_examples[: self.DEFAULT_SELECTED]
            if self._use_sampling:
                content = await self._sample_selection(previous_thought, context)
            else:
                content = self._generate_selection_heuristic(previous_thought, context)
            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "select":
            self._current_phase = "demonstrate"
            if self._use_sampling:
                content = await self._sample_demonstration(previous_thought, context)
            else:
                content = self._generate_demonstration_heuristic(previous_thought, context)
            thought_type = ThoughtType.REASONING
            confidence = 0.75
        elif prev_phase == "demonstrate":
            self._current_phase = "reason"
            if self._use_sampling:
                content = await self._sample_reasoning(previous_thought, context)
            else:
                content = self._generate_reasoning_heuristic(previous_thought, context)
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, context)
            else:
                content = self._generate_conclusion_heuristic(previous_thought, context)
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.COMPLEXITY_BASED,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "selected": len(self._selected_examples),
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Sampling methods (use LLM when available)
    async def _sample_complexity_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial complexity analysis using LLM sampling."""
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are an expert in complexity-based prompting for multi-step reasoning.
Analyze the given problem and identify the complexity of available examples to use as
demonstrations.

Structure your analysis with:
1. Problem statement
2. Available examples with complexity measurements (reasoning chain length)
3. Complexity ranking
4. Selection strategy for using complex examples

Complex examples with longer reasoning chains lead to better performance."""

        user_prompt = f"""Problem: {input_text}{context_info}

Available examples for analysis (sorted by complexity):
{
            chr(10).join(
                f"  Example {e['id']}: {e['steps']} reasoning steps ({e['complexity']})"
                for e in sorted(self._available_examples, key=lambda x: x["steps"], reverse=True)
            )
        }

Analyze the complexity of these examples and explain how to use the most complex ones
for demonstrations."""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_complexity_analysis_heuristic(
                input_text, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_selection(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate example selection using LLM sampling."""
        context_info = ""
        if context:
            context_info = f"\nAdditional Context: {context}"

        system_prompt = """You are an expert in complexity-based prompting.
Select the most complex examples to use as demonstrations for multi-step reasoning.

Explain why complex examples (those with longer reasoning chains) are more effective
for priming thorough analysis on complex problems."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Selected Examples:
{
            chr(10).join(
                f"  [SELECTED] Example {e['id']}: {e['steps']} steps"
                for e in self._selected_examples
            )
        }

Explain the selection of these top {self.DEFAULT_SELECTED} most complex examples and
how they will improve reasoning performance.{context_info}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_selection_heuristic(
                previous_thought, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_demonstration(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate demonstration content using LLM sampling."""
        context_info = ""
        if context:
            context_info = f"\nAdditional Context: {context}"

        system_prompt = """You are an expert in complexity-based prompting.
Demonstrate how to use complex examples as few-shot prompts to prime longer,
more thorough reasoning chains.

Show how complex demonstrations teach better reasoning patterns."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Using these selected examples as few-shot demonstrations:
{chr(10).join(f"  Example {e['id']} ({e['steps']} steps):" for e in self._selected_examples)}

Explain how these complex examples will be used as demonstrations to prime
extended reasoning on the target problem.{context_info}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_demonstration_heuristic(
                previous_thought, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_reasoning(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate extended reasoning using LLM sampling."""
        avg_steps = sum(e["steps"] for e in self._selected_examples) / len(self._selected_examples)

        context_info = ""
        if context:
            context_info = f"\nAdditional Context: {context}"

        system_prompt = """You are an expert in complexity-based prompting.
Apply extended, thorough reasoning primed by complex examples.

Generate a detailed multi-step reasoning process that demonstrates how complex
examples induce longer and more thorough reasoning chains."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Primed with examples averaging {avg_steps:.0f} reasoning steps, apply extended
reasoning to the target problem. Show how the complex demonstrations have induced
a thorough, step-by-step analysis.{context_info}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_reasoning_heuristic(
                previous_thought, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling."""
        avg_complexity = sum(e["steps"] for e in self._selected_examples) / len(
            self._selected_examples
        )

        context_info = ""
        if context:
            context_info = f"\nAdditional Context: {context}"

        system_prompt = """You are an expert in complexity-based prompting.
Synthesize the complete complexity-based reasoning process and provide a final answer.

Summarize how complex examples were selected and used to improve reasoning performance."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Complexity-Based Prompting Summary:
- Examples evaluated: {len(self._available_examples)}
- Complex examples selected: {len(self._selected_examples)}
- Average complexity: {avg_complexity:.0f} reasoning steps

Provide a final synthesis of the complexity-based prompting process,
explaining the final answer and how the method improved reasoning quality.{context_info}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_conclusion_heuristic(
                previous_thought, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    # Heuristic fallback methods (when LLM sampling unavailable)
    def _generate_complexity_analysis_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial complexity analysis using heuristic templates."""
        return (
            f"Step {self._step_counter}: Measure Example Complexity\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing available examples by reasoning chain length...\n\n"
            f"Examples (sorted by complexity):\n"
            + "\n".join(
                f"  Example {e['id']}: {e['steps']} reasoning steps ({e['complexity']})"
                for e in sorted(self._available_examples, key=lambda x: x["steps"], reverse=True)
            )
            + "\n\nComplexity metric: Number of reasoning steps.\n"
            "Next: Select highest-complexity examples."
        )

    def _generate_selection_heuristic(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate example selection using heuristic templates."""
        return (
            f"Step {self._step_counter}: Select Complex Examples\n\n"
            f"Selecting top {self.DEFAULT_SELECTED} by complexity:\n\n"
            + "\n".join(
                f"  [SELECTED] Example {e['id']}: {e['steps']} steps"
                for e in self._selected_examples
            )
            + "\n\nComplex examples teach better reasoning patterns.\n"
            "Next: Use as demonstrations."
        )

    def _generate_demonstration_heuristic(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate demonstration content using heuristic templates."""
        return (
            f"Step {self._step_counter}: Apply Complex Demonstrations\n\n"
            f"Using selected examples as few-shot prompts:\n\n"
            + "\n".join(
                f"  Example {e['id']} ({e['steps']} steps):\n"
                f"    Q: [Complex question]\n"
                f"    A: [Detailed {e['steps']}-step reasoning]\n"
                for e in self._selected_examples
            )
            + "\n\nComplex demonstrations prime longer reasoning chains.\n"
            "Next: Apply reasoning to target problem."
        )

    def _generate_reasoning_heuristic(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate extended reasoning using heuristic templates."""
        avg_steps = sum(e["steps"] for e in self._selected_examples) / len(self._selected_examples)
        return (
            f"Step {self._step_counter}: Apply Extended Reasoning\n\n"
            f"Primed with avg {avg_steps:.0f} steps per example:\n\n"
            f"Reasoning on target:\n"
            f"  Step 1: [First reasoning step]\n"
            f"  Step 2: [Second reasoning step]\n"
            f"  ...\n"
            f"  Step N: [Final reasoning step]\n\n"
            f"Complex examples induced thorough reasoning."
        )

    def _generate_conclusion_heuristic(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using heuristic templates."""
        avg_complexity = sum(e["steps"] for e in self._selected_examples) / len(
            self._selected_examples
        )
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Complexity-Based Prompting Complete:\n"
            f"  Examples evaluated: {len(self._available_examples)}\n"
            f"  Complex examples selected: {len(self._selected_examples)}\n"
            f"  Avg complexity: {avg_complexity:.0f} steps\n\n"
            f"Final Answer: [Answer from extended reasoning]\n"
            f"Confidence: High (88%)\n\n"
            f"Method: Complexity-based example selection\n"
            f"  - Selected examples with longest reasoning chains\n"
            f"  - Complex demonstrations prime thorough analysis"
        )


__all__ = ["ComplexityBased", "COMPLEXITY_BASED_METADATA"]
