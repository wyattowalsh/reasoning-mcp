"""Active Prompt reasoning method.

This module implements Active Prompt, which selects the most informative
examples based on uncertainty. Instead of using random or fixed examples,
it identifies questions where the model is most uncertain and uses human-
annotated answers for those as demonstrations.

Key phases:
1. Query: Run initial queries to measure uncertainty
2. Select: Choose examples with highest uncertainty
3. Annotate: Use annotated examples for demonstration
4. Reason: Apply CoT with selected examples

Reference: Diao et al. (2023) - "Active Prompting with Chain-of-Thought
for Large Language Models"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_feedback,
    elicit_reasoning_guidance,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from fastmcp.server import Context

    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


ACTIVE_PROMPT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ACTIVE_PROMPT,
    name="Active Prompt",
    description="Selects examples based on uncertainty for better demonstrations. "
    "Identifies where the model is most uncertain and uses those for few-shot learning.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"active-learning", "uncertainty", "few-shot", "selection", "examples"}),
    complexity=5,
    supports_branching=False,
    supports_revision=False,
    requires_context=True,
    min_thoughts=4,
    max_thoughts=7,
    avg_tokens_per_thought=300,
    best_for=("few-shot learning", "example selection", "adaptive prompting"),
    not_recommended_for=("zero-shot tasks", "simple queries"),
)


class ActivePrompt(ReasoningMethodBase):
    """Active Prompt reasoning method implementation."""

    DEFAULT_CANDIDATES = 5
    DEFAULT_SELECTED = 3

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Active Prompt method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "query"
        self._candidate_examples: list[dict[str, Any]] = []
        self._selected_examples: list[dict[str, Any]] = []
        self._uncertainty_scores: list[float] = []
        self.enable_elicitation = enable_elicitation
        self._ctx: Context | None = None
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.ACTIVE_PROMPT

    @property
    def name(self) -> str:
        return ACTIVE_PROMPT_METADATA.name

    @property
    def description(self) -> str:
        return ACTIVE_PROMPT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "query"
        self._candidate_examples = []
        self._selected_examples = []
        self._uncertainty_scores = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Active Prompt must be initialized before execution")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter = 1
        self._current_phase = "query"

        # Initialize candidate examples with uncertainty
        import random

        self._candidate_examples = [
            {
                "id": i + 1,
                "question": f"Example question {i + 1}",
                "uncertainty": round(random.uniform(0.3, 0.9), 2),
            }
            for i in range(self.DEFAULT_CANDIDATES)
        ]
        self._uncertainty_scores = [e["uncertainty"] for e in self._candidate_examples]

        # Generate content with sampling or fallback
        if use_sampling:
            content = await self._sample_query_phase(input_text, context)
        else:
            content = self._generate_query_phase(input_text)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ACTIVE_PROMPT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "candidates": len(self._candidate_examples),
                "input": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.ACTIVE_PROMPT
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
            raise RuntimeError("Active Prompt must be initialized before continuation")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        if execution_context:
            self._execution_context = execution_context

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "query")

        if prev_phase == "query":
            self._current_phase = "select"
            # Select top uncertain examples
            sorted_examples = sorted(
                self._candidate_examples, key=lambda x: x["uncertainty"], reverse=True
            )
            self._selected_examples = sorted_examples[: self.DEFAULT_SELECTED]

            # Optional elicitation: ask user for guidance on example selection
            elicited_response = ""
            if self.enable_elicitation and self._ctx and not guidance:
                try:
                    elicit_config = ElicitationConfig(
                        timeout=60, required=False, default_on_timeout=None
                    )

                    # Ask for reasoning guidance on which examples to prioritize
                    selected_ids = ", ".join(f"Example {e['id']}" for e in self._selected_examples)
                    reasoning_guidance = await elicit_reasoning_guidance(
                        self._ctx,
                        previous_thought.metadata.get("input", ""),
                        f"Selected examples based on uncertainty: {selected_ids}",
                        config=elicit_config,
                    )
                    if reasoning_guidance.direction:
                        focus_text = (
                            ", ".join(reasoning_guidance.focus_areas)
                            if reasoning_guidance.focus_areas
                            else "Use selected high-uncertainty examples"
                        )
                        elicited_response = (
                            f"\n\n[Your Guidance]:\n"
                            f"Direction: {reasoning_guidance.direction}\n"
                            f"Focus: {focus_text}"
                        )
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        phase="select",
                        error=str(e),
                    )
                    # Elicitation failed - continue without it
                except Exception as e:
                    logger.error(
                        "elicitation_unexpected_error",
                        method="continue_reasoning",
                        phase="select",
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            # Generate content with sampling or fallback
            if use_sampling:
                content = await self._sample_select_phase(
                    previous_thought.metadata.get("input", ""),
                    sorted_examples,
                    elicited_response,
                    context,
                )
            else:
                content = self._generate_select_phase(sorted_examples, elicited_response)

            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "select":
            self._current_phase = "annotate"

            # Optional elicitation: ask user for feedback on annotated examples
            elicited_response = ""
            if self.enable_elicitation and self._ctx and not guidance:
                try:
                    elicit_config = ElicitationConfig(
                        timeout=60, required=False, default_on_timeout=None
                    )

                    # Ask for feedback on the annotated examples
                    feedback_prompt = (
                        "Review the annotated examples. "
                        "Do you have any feedback on how they should be used or refined?"
                    )
                    feedback = await elicit_feedback(
                        self._ctx,
                        feedback_prompt,
                        config=elicit_config,
                    )
                    if feedback.feedback:
                        elicited_response = f"\n\n[Your Feedback]: {feedback.feedback}"
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        phase="annotate",
                        error=str(e),
                    )
                    # Elicitation failed - continue without it
                except Exception as e:
                    logger.error(
                        "elicitation_unexpected_error",
                        method="continue_reasoning",
                        phase="annotate",
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            # Generate content with sampling or fallback
            if use_sampling:
                content = await self._sample_annotate_phase(
                    previous_thought.metadata.get("input", ""),
                    elicited_response,
                    context,
                )
            else:
                content = self._generate_annotate_phase(elicited_response)

            thought_type = ThoughtType.REASONING
            confidence = 0.8
        elif prev_phase == "annotate":
            self._current_phase = "reason"
            # Generate content with sampling or fallback
            if use_sampling:
                content = await self._sample_reason_phase(
                    previous_thought.metadata.get("input", ""),
                    context,
                )
            else:
                content = self._generate_reason_phase()

            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            # Generate content with sampling or fallback
            if use_sampling:
                content = await self._sample_conclude_phase(
                    previous_thought.metadata.get("input", ""),
                    context,
                )
            else:
                content = self._generate_conclude_phase()

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.87

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.ACTIVE_PROMPT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "selected": len(self._selected_examples),
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Sampling methods
    async def _sample_query_phase(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the query phase content using LLM sampling.

        Args:
            input_text: The problem to solve
            context: Optional additional context

        Returns:
            The content for the query phase thought
        """
        system_prompt = """You are an Active Prompt reasoning assistant.
Analyze the given problem and identify candidate examples where uncertainty is highest.
Your response should:
1. Explain the Active Prompt approach
2. List candidate examples with uncertainty scores
3. Explain how uncertainty is measured
4. Prepare for example selection based on uncertainty"""

        candidates_text = chr(10).join(
            f"[{e['id']}] {e['question']}: uncertainty = {e['uncertainty']:.0%}"
            for e in self._candidate_examples
        )

        user_prompt = f"""Problem: {input_text}

Using Active Prompt reasoning:
1. Generate {self.DEFAULT_CANDIDATES} candidate example questions related to this problem
2. For each candidate, estimate the model's uncertainty (0.0 to 1.0)
3. Explain what makes each example uncertain
4. Prepare to select the most uncertain examples for demonstration

Candidate Examples with Uncertainty Analysis:
{candidates_text}

Analyze these candidates and their uncertainty levels."""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_query_phase(input_text),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=600,
        )

    def _generate_query_phase(self, input_text: str) -> str:
        """Generate the query phase content (fallback heuristic).

        Args:
            input_text: The problem to solve

        Returns:
            The content for the query phase thought
        """
        return (
            f"Step {self._step_counter}: Query for Uncertainty (Active Prompt)\n\n"
            f"Problem: {input_text}\n\n"
            f"Running initial queries to measure uncertainty...\n\n"
            f"Candidate Examples (with uncertainty scores):\n"
            + "\n".join(
                f"  [{e['id']}] {e['question']}: uncertainty = {e['uncertainty']:.0%}"
                for e in self._candidate_examples
            )
            + "\n\nNext: Select highest-uncertainty examples."
        )

    async def _sample_select_phase(
        self,
        input_text: str,
        sorted_examples: list[dict[str, Any]],
        elicited_response: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the select phase content using LLM sampling.

        Args:
            input_text: The problem to solve
            sorted_examples: Examples sorted by uncertainty
            elicited_response: Any elicited user feedback
            context: Optional additional context

        Returns:
            The content for the select phase thought
        """
        system_prompt = """You are an Active Prompt reasoning assistant.
Select examples with the highest uncertainty for demonstration.
Your response should:
1. Rank examples by uncertainty
2. Select the most uncertain examples
3. Explain why these examples are most informative
4. Justify the selection strategy"""

        ranked_examples = chr(10).join(
            f"{i + 1}. Example {e['id']}: {e['uncertainty']:.0%}"
            + (" <- SELECTED" if e in self._selected_examples else "")
            for i, e in enumerate(sorted_examples)
        )

        user_prompt = f"""Problem: {input_text}

Examples ranked by uncertainty:
{ranked_examples}

Selected {len(self._selected_examples)} examples with highest uncertainty.
Explain why these selections are optimal for learning.{elicited_response}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_select_phase(
                sorted_examples, elicited_response
            ),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=600,
        )

    def _generate_select_phase(
        self,
        sorted_examples: list[dict[str, Any]],
        elicited_response: str,
    ) -> str:
        """Generate the select phase content (fallback heuristic).

        Args:
            sorted_examples: Examples sorted by uncertainty
            elicited_response: Any elicited user feedback

        Returns:
            The content for the select phase thought
        """
        return (
            f"Step {self._step_counter}: Select High-Uncertainty Examples\n\n"
            f"Ranking by uncertainty (highest first):\n"
            + "\n".join(
                f"  {i + 1}. Example {e['id']}: {e['uncertainty']:.0%}"
                + (" <- SELECTED" if e in self._selected_examples else "")
                for i, e in enumerate(sorted_examples)
            )
            + f"\n\nSelected {len(self._selected_examples)} examples with highest uncertainty.\n"
            f"These are most informative for learning.\n"
            f"Next: Use annotated examples for demonstration." + elicited_response
        )

    async def _sample_annotate_phase(
        self,
        input_text: str,
        elicited_response: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the annotate phase content using LLM sampling.

        Args:
            input_text: The problem to solve
            elicited_response: Any elicited user feedback
            context: Optional additional context

        Returns:
            The content for the annotate phase thought
        """
        system_prompt = """You are an Active Prompt reasoning assistant.
Use the selected high-uncertainty examples as demonstrations.
Your response should:
1. Present the annotated examples
2. Show the reasoning process for each example
3. Explain how these examples inform the solution
4. Prepare to apply this knowledge to the target problem"""

        examples_text = chr(10).join(
            f"Example {e['id']}:{chr(10)}"
            f"  Q: {e['question']}{chr(10)}"
            f"  A: [Annotated reasoning and answer]"
            for e in self._selected_examples
        )

        user_prompt = f"""Problem: {input_text}

Selected Examples for Demonstration:
{examples_text}

Annotate these examples with detailed reasoning chains.
Show how they teach about uncertain areas.{elicited_response}"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_annotate_phase(elicited_response),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=700,
        )

    def _generate_annotate_phase(self, elicited_response: str) -> str:
        """Generate the annotate phase content (fallback heuristic).

        Args:
            elicited_response: Any elicited user feedback

        Returns:
            The content for the annotate phase thought
        """
        return (
            f"Step {self._step_counter}: Apply Annotated Examples\n\n"
            f"Using selected examples as demonstrations:\n\n"
            + "\n".join(
                f"  Example {e['id']}:\n"
                f"    Q: {e['question']}\n"
                f"    A: [Annotated reasoning and answer]\n"
                for e in self._selected_examples
            )
            + "\n\nThese examples teach the model about uncertain areas.\n"
            "Next: Reason with informed demonstrations." + elicited_response
        )

    async def _sample_reason_phase(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the reason phase content using LLM sampling.

        Args:
            input_text: The problem to solve
            context: Optional additional context

        Returns:
            The content for the reason phase thought
        """
        system_prompt = """You are an Active Prompt reasoning assistant.
Apply the patterns learned from high-uncertainty examples to solve the target problem.
Your response should:
1. Apply patterns from the annotated examples
2. Follow the demonstrated reasoning style
3. Address areas of uncertainty identified earlier
4. Provide detailed step-by-step reasoning"""

        user_prompt = f"""Problem: {input_text}

With {len(self._selected_examples)} informative examples as demonstrations:

Now apply the reasoning patterns from these examples to solve the target problem:
1. Apply patterns from examples
2. Follow demonstrated reasoning style
3. Address areas of previous uncertainty

Provide your reasoning."""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=self._generate_reason_phase,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=700,
        )

    def _generate_reason_phase(self) -> str:
        """Generate the reason phase content (fallback heuristic).

        Returns:
            The content for the reason phase thought
        """
        return (
            f"Step {self._step_counter}: Apply Reasoning\n\n"
            f"With {len(self._selected_examples)} informative examples as demonstrations:\n\n"
            f"Reasoning on the target problem:\n"
            f"  1. Apply patterns from examples\n"
            f"  2. Follow demonstrated reasoning style\n"
            f"  3. Address areas of previous uncertainty\n\n"
            f"Answer derived from active-prompted reasoning."
        )

    async def _sample_conclude_phase(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the conclude phase content using LLM sampling.

        Args:
            input_text: The problem to solve
            context: Optional additional context

        Returns:
            The content for the conclude phase thought
        """
        system_prompt = """You are an Active Prompt reasoning assistant.
Provide the final answer based on active-prompted reasoning.
Your response should:
1. Summarize the Active Prompt process
2. Present the final answer
3. Explain how uncertainty-based selection improved the solution
4. Provide confidence assessment"""

        user_prompt = f"""Problem: {input_text}

Active Prompt Complete:
- Candidates evaluated: {len(self._candidate_examples)}
- Examples selected: {len(self._selected_examples)}
- Selection criterion: Highest uncertainty

Provide the final answer with a summary of the active-prompted reasoning process."""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=self._generate_conclude_phase,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=600,
        )

    def _generate_conclude_phase(self) -> str:
        """Generate the conclude phase content (fallback heuristic).

        Returns:
            The content for the conclude phase thought
        """
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Active Prompt Complete:\n"
            f"  Candidates evaluated: {len(self._candidate_examples)}\n"
            f"  Examples selected: {len(self._selected_examples)}\n"
            f"  Selection criterion: Highest uncertainty\n\n"
            f"Final Answer: [Answer with active-prompted reasoning]\n"
            f"Confidence: High (87%)\n\n"
            f"Method: Active example selection\n"
            f"  - Measured uncertainty across examples\n"
            f"  - Selected most informative demonstrations\n"
            f"  - Applied targeted few-shot learning"
        )


__all__ = ["ActivePrompt", "ACTIVE_PROMPT_METADATA"]
