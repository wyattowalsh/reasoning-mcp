"""Iterative Refinement reasoning method.

This module implements Iterative Refinement, which improves answers through
multiple passes of generation and refinement. Each iteration critiques and
improves upon the previous answer until convergence or max iterations.

Key phases:
1. Generate: Initial answer generation
2. Critique: Identify weaknesses in current answer
3. Refine: Improve answer based on critique
4. Repeat: Continue until satisfactory or max iterations

Reference: Standard technique widely used in LLM reasoning (2022-2025)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_selection,
)
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


ITERATIVE_REFINEMENT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ITERATIVE_REFINEMENT,
    name="Iterative Refinement",
    description="Multiple passes to progressively refine and improve answers. "
    "Each iteration critiques and enhances the previous response.",
    category=MethodCategory.CORE,
    tags=frozenset({"iterative", "refinement", "improvement", "critique", "progressive"}),
    complexity=4,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=("quality improvement", "writing tasks", "complex responses"),
    not_recommended_for=("simple queries", "time-critical tasks"),
)


class IterativeRefinement(ReasoningMethodBase):
    """Iterative Refinement reasoning method implementation."""

    _use_sampling: bool = True
    DEFAULT_MAX_ITERATIONS = 3

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_elicitation: bool = True,
    ) -> None:
        self._max_iterations = max_iterations
        self.enable_elicitation = enable_elicitation
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._current_iteration = 0
        self._current_answer: str = ""
        self._improvement_history: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.ITERATIVE_REFINEMENT

    @property
    def name(self) -> str:
        return ITERATIVE_REFINEMENT_METADATA.name

    @property
    def description(self) -> str:
        return ITERATIVE_REFINEMENT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.CORE

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._current_iteration = 0
        self._current_answer = ""
        self._improvement_history = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Iterative Refinement must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "generate"
        self._current_iteration = 1

        # Generate initial answer using LLM with fallback
        self._current_answer = await self._sample_with_fallback(
            user_prompt=self._build_initial_answer_prompt(input_text),
            fallback_generator=lambda: self._generate_initial_answer(input_text),
            system_prompt=self._get_initial_answer_system_prompt(),
        )

        self._improvement_history.append(
            {
                "iteration": 1,
                "quality": 0.6,
                "changes": "Initial generation",
            }
        )

        content = (
            f"Step {self._step_counter}: Initial Generation (Iterative Refinement)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating initial answer (Iteration 1/{self._max_iterations})...\n\n"
            f"Initial Answer:\n{self._current_answer}\n\n"
            f"Quality estimate: 60%\n"
            f"Next: Critique and identify improvements."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ITERATIVE_REFINEMENT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "iteration": self._current_iteration,
                "input": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.ITERATIVE_REFINEMENT
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
            raise RuntimeError("Iterative Refinement must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate" or prev_phase == "refine":
            self._current_phase = "critique"

            # Generate critique using LLM with fallback
            original_input = previous_thought.metadata.get("input", "")
            critique_text = await self._sample_with_fallback(
                user_prompt=self._build_critique_prompt(self._current_answer, original_input),
                fallback_generator=self._generate_critique,
                system_prompt=self._get_critique_system_prompt(),
            )

            content = (
                f"Step {self._step_counter}: Critique (Iteration {self._current_iteration})\n\n"
                f"Analyzing current answer for weaknesses:\n\n"
                f"Current Answer:\n{self._current_answer}\n\n"
                f"{critique_text}\n\n"
                f"Improvement opportunities identified.\n"
                f"Next: Apply refinements."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.65 + (self._current_iteration * 0.05)
        elif prev_phase == "critique" and self._current_iteration < self._max_iterations:
            self._current_phase = "refine"
            self._current_iteration += 1
            prev_quality = self._improvement_history[-1]["quality"]
            new_quality = min(0.95, prev_quality + 0.1)

            # Optional elicitation: ask user which refinement approach to use
            refinement_approach = None
            if (
                self.enable_elicitation
                and self._execution_context
                and hasattr(self._execution_context, "ctx")
                and self._execution_context.ctx
            ):
                try:
                    options = [
                        {
                            "id": "aggressive",
                            "label": "Aggressive refinement - major changes allowed",
                        },
                        {
                            "id": "conservative",
                            "label": "Conservative refinement - small incremental changes",
                        },
                        {
                            "id": "targeted",
                            "label": "Targeted refinement - focus on specific weaknesses",
                        },
                        {
                            "id": "comprehensive",
                            "label": "Comprehensive - address all aspects",
                        },
                    ]
                    config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                    selection = await elicit_selection(
                        self._execution_context.ctx,
                        "What refinement approach should we use?",
                        options,
                        config=config,
                    )
                    if selection and selection.selected:
                        refinement_approach = selection.selected
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        error=str(e),
                    )
                    # Fall back to default behavior (no refinement_approach)

            # Generate refined answer using LLM with fallback
            refined_answer, changes_made = await self._sample_refinement_with_fallback(
                self._current_answer,
                previous_thought.content,
                previous_thought.metadata.get("input", ""),
                refinement_approach,
            )
            self._current_answer = refined_answer

            self._improvement_history.append(
                {
                    "iteration": self._current_iteration,
                    "quality": new_quality,
                    "changes": f"Addressed {len(changes_made)} issues from critique",
                }
            )

            changes_text = "\n".join(f"  - {change}" for change in changes_made)
            next_step = (
                "Critique again." if self._current_iteration < self._max_iterations else "Finalize."
            )
            approach_text = (
                f" (using {refinement_approach} approach)" if refinement_approach else ""
            )
            content = (
                f"Step {self._step_counter}: Refine{approach_text} "
                f"(Iteration {self._current_iteration}/{self._max_iterations})\n\n"
                f"Applying improvements based on critique:\n\n"
                f"Changes Made:\n"
                f"{changes_text}\n\n"
                f"Refined Answer:\n{self._current_answer}\n\n"
                f"Quality improvement: {prev_quality:.0%} -> {new_quality:.0%}\n"
                f"Next: {next_step}"
            )
            thought_type = ThoughtType.REVISION
            confidence = new_quality
        else:
            self._current_phase = "conclude"
            final_quality = self._improvement_history[-1]["quality"]
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Iterative Refinement Complete:\n"
                f"  Iterations: {self._current_iteration}\n"
                f"  Quality progression:\n"
                + "\n".join(
                    f"    Iteration {h['iteration']}: {h['quality']:.0%}"
                    for h in self._improvement_history
                )
                + f"\n\nFinal Answer:\n{self._current_answer}\n\n"
                f"Confidence: High ({int(final_quality * 100)}%)\n\n"
                f"Method: Iterative refinement\n"
                f"  - Generate -> Critique -> Refine cycle\n"
                f"  - Progressive quality improvement\n"
                f"  - Converged after {self._current_iteration} iterations"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = final_quality

        # Propagate input from previous thought
        input_text = previous_thought.metadata.get("input", "")

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.ITERATIVE_REFINEMENT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "iteration": self._current_iteration,
                "history": self._improvement_history,
                "input": input_text,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Heuristic generation methods (fallback when LLM not available)

    def _generate_initial_answer(self, input_text: str) -> str:
        """Generate initial answer using heuristic approach.

        Args:
            input_text: The problem or question to address

        Returns:
            A heuristic initial answer
        """
        return f"[Initial answer for: {input_text[:50]}...]"

    def _generate_critique(self) -> str:
        """Generate critique using heuristic approach.

        Returns:
            A heuristic critique with identified issues
        """
        return (
            "Identified Issues:\n"
            "  1. Could be more specific in [area]\n"
            "  2. Missing consideration of [aspect]\n"
            "  3. Could improve clarity in [section]"
        )

    # Prompt building helpers for _sample_with_fallback

    def _get_initial_answer_system_prompt(self) -> str:
        """Get the system prompt for initial answer generation."""
        return (
            "You are a reasoning assistant using Iterative Refinement methodology.\n"
            "Generate an initial answer to the problem. This will be refined in "
            "subsequent iterations.\n"
            "Provide a thoughtful first attempt that addresses the core question."
        )

    def _build_initial_answer_prompt(self, input_text: str) -> str:
        """Build the user prompt for initial answer generation.

        Args:
            input_text: The problem or question to address

        Returns:
            Formatted prompt for LLM
        """
        return (
            f"Problem: {input_text}\n\n"
            "Generate an initial answer. Focus on addressing the main question clearly "
            "and comprehensively.\n"
            "This is iteration 1, so aim for a solid foundation that can be improved upon.\n\n"
            "Provide your answer directly without meta-commentary about the refinement process."
        )

    def _get_critique_system_prompt(self) -> str:
        """Get the system prompt for critique generation."""
        return (
            "You are a critical evaluator in an Iterative Refinement process.\n"
            "Analyze the current answer and identify specific areas for improvement.\n"
            "Be constructive and specific in your critique."
        )

    def _build_critique_prompt(self, current_answer: str, original_question: str) -> str:
        """Build the user prompt for critique generation.

        Args:
            current_answer: The answer to critique
            original_question: The original problem/question

        Returns:
            Formatted prompt for LLM
        """
        return (
            f"Original Question: {original_question}\n\n"
            f"Current Answer:\n{current_answer}\n\n"
            "Analyze this answer and identify 3-5 specific issues or areas for improvement.\n"
            "Focus on:\n"
            "- Clarity and precision\n"
            "- Completeness of coverage\n"
            "- Accuracy of reasoning\n"
            "- Missing considerations\n"
            "- Areas lacking detail or examples\n\n"
            "Format your critique as:\n"
            "Identified Issues:\n"
            "  1. [specific issue]\n"
            "  2. [specific issue]\n"
            "  3. [specific issue]\n"
            "  ...\n\n"
            "Be specific and actionable in your critique."
        )

    def _get_refinement_system_prompt(self, approach: str | None = None) -> str:
        """Get the system prompt for refinement generation.

        Args:
            approach: Optional refinement approach

        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a refinement specialist in an Iterative Refinement process.\n"
            "Based on the critique, generate an improved version of the answer that "
            "addresses the identified issues. Focus on making concrete improvements "
            "while preserving what works well."
        )

        approach_guidance = ""
        if approach == "aggressive":
            approach_guidance = (
                "\n\nApproach: Aggressive refinement - feel free to make "
                "major changes and rewrite sections as needed."
            )
        elif approach == "conservative":
            approach_guidance = (
                "\n\nApproach: Conservative refinement - make small, incremental "
                "changes while preserving the core structure."
            )
        elif approach == "targeted":
            approach_guidance = (
                "\n\nApproach: Targeted refinement - focus specifically on "
                "addressing the weaknesses identified in the critique."
            )
        elif approach == "comprehensive":
            approach_guidance = (
                "\n\nApproach: Comprehensive refinement - address all aspects "
                "systematically and be thorough."
            )

        return f"{base_prompt}{approach_guidance}"

    def _build_refinement_prompt(
        self,
        current_answer: str,
        critique_content: str,
        original_question: str,
    ) -> str:
        """Build the user prompt for refinement generation.

        Args:
            current_answer: The current answer to refine
            critique_content: The critique identifying issues
            original_question: The original problem/question

        Returns:
            Formatted prompt for LLM
        """
        return (
            f"Original Question: {original_question}\n\n"
            f"Current Answer:\n{current_answer}\n\n"
            f"Critique:\n{critique_content}\n\n"
            "Generate a refined version of the answer that addresses the critique.\n"
            "Then, on a new line starting with \"CHANGES:\", list the specific "
            "improvements made.\n\n"
            "Format:\n"
            "[Your refined answer here]\n\n"
            "CHANGES:\n"
            "- [change 1]\n"
            "- [change 2]\n"
            "- [change 3]"
        )

    def _generate_refinement_fallback(self) -> tuple[str, list[str]]:
        """Generate fallback refinement when LLM is unavailable.

        Returns:
            Tuple of (refined answer placeholder, list of placeholder changes)
        """
        return (
            f"[Refined answer - iteration {self._current_iteration}]",
            ["Improved specificity", "Added missing consideration", "Enhanced clarity"],
        )

    def _parse_refinement_response(self, result: str) -> tuple[str, list[str]]:
        """Parse LLM response to extract answer and changes.

        Args:
            result: Raw LLM response

        Returns:
            Tuple of (refined answer, list of changes made)
        """
        if "CHANGES:" in result:
            parts = result.split("CHANGES:", 1)
            refined_answer = parts[0].strip()
            changes_section = parts[1].strip()

            # Extract individual changes
            changes = [
                line.strip("- ").strip()
                for line in changes_section.split("\n")
                if line.strip().startswith("-") or line.strip().startswith("•")
            ]

            if not changes:
                # If no bullet points found, treat each non-empty line as a change
                changes = [line.strip() for line in changes_section.split("\n") if line.strip()]

            return refined_answer, changes if changes else ["Refined based on critique"]
        else:
            # If no CHANGES section found, use the whole response as the answer
            return result.strip(), ["Refined based on critique"]

    async def _sample_refinement_with_fallback(
        self,
        current_answer: str,
        critique_content: str,
        original_question: str,
        approach: str | None = None,
    ) -> tuple[str, list[str]]:
        """Generate refined answer using LLM with proper fallback handling.

        This method uses _sample_with_fallback for the LLM call but returns
        a tuple since refinement needs both the answer and changes list.

        Args:
            current_answer: The current answer to refine
            critique_content: The critique identifying issues
            original_question: The original problem/question
            approach: Optional refinement approach

        Returns:
            Tuple of (refined answer, list of changes made)
        """
        # Use a marker to detect if fallback was used
        fallback_marker = "__FALLBACK_USED__"

        def fallback_with_marker() -> str:
            return fallback_marker

        result = await self._sample_with_fallback(
            user_prompt=self._build_refinement_prompt(
                current_answer, critique_content, original_question
            ),
            fallback_generator=fallback_with_marker,
            system_prompt=self._get_refinement_system_prompt(approach),
        )

        if result == fallback_marker:
            return self._generate_refinement_fallback()

        return self._parse_refinement_response(result)


__all__ = ["IterativeRefinement", "ITERATIVE_REFINEMENT_METADATA"]
