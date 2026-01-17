"""Zero-Shot Chain-of-Thought reasoning method.

This module implements Zero-Shot CoT, the simplest form of chain-of-thought
prompting. It uses the magic phrase "Let's think step by step" to trigger
reasoning without requiring few-shot examples.

Key phases:
1. Trigger: Apply "Let's think step by step"
2. Reason: Generate step-by-step reasoning
3. Extract: Derive answer from reasoning
4. Conclude: Present final answer

Reference: Kojima et al. (2022) - "Large Language Models are Zero-Shot Reasoners"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
from reasoning_mcp.methods.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SAMPLING_TEMPERATURE,
    MethodMetadata,
    ReasoningMethodBase,
)
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


ZERO_SHOT_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ZERO_SHOT_COT,
    name="Zero-Shot Chain-of-Thought",
    description="Simple 'Let's think step by step' trigger for reasoning. "
    "No examples needed - just the magic phrase to elicit step-by-step thinking.",
    category=MethodCategory.CORE,
    tags=frozenset({"zero-shot", "simple", "trigger", "step-by-step", "foundational"}),
    complexity=2,
    supports_branching=False,
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=5,
    avg_tokens_per_thought=200,
    best_for=("quick reasoning", "simple problems", "general tasks"),
    not_recommended_for=("complex multi-step", "tasks needing examples"),
)


class ZeroShotCoT(ReasoningMethodBase):
    """Zero-Shot Chain-of-Thought reasoning method implementation."""

    TRIGGER_PHRASE = "Let's think step by step."
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "trigger"
        self._reasoning_steps: list[str] = []
        self._execution_context: ExecutionContext | None = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        return MethodIdentifier.ZERO_SHOT_COT

    @property
    def name(self) -> str:
        return ZERO_SHOT_COT_METADATA.name

    @property
    def description(self) -> str:
        return ZERO_SHOT_COT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.CORE

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "trigger"
        self._reasoning_steps = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Zero-Shot CoT must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "trigger"

        # Elicit user guidance on reasoning approach if enabled
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "step_by_step", "label": "Step-by-step - Detailed reasoning"},
                    {"id": "concise", "label": "Concise - Brief reasoning"},
                    {"id": "thorough", "label": "Thorough - Exhaustive analysis"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "How should I approach this reasoning?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    # Use selection to guide reasoning
                    pass
            except TimeoutError as e:
                logger.warning(
                    "elicitation_timeout",
                    method="execute",
                    error=str(e),
                )
            except (ConnectionError, OSError) as e:
                logger.warning(
                    "elicitation_connection_error",
                    method="execute",
                    error=str(e),
                )
            except ValueError as e:
                logger.warning(
                    "elicitation_value_error",
                    method="execute",
                    error=str(e),
                )

        # Generate reasoning using _sample_with_fallback
        reasoning_content = await self._sample_reasoning(input_text)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ZERO_SHOT_COT,
            content=reasoning_content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "trigger": self.TRIGGER_PHRASE,
                "sampled": execution_context is not None and execution_context.can_sample,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.ZERO_SHOT_COT
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
            raise RuntimeError("Zero-Shot CoT must be initialized before continuation")

        self._execution_context = execution_context
        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "trigger")

        # Generate continuation using _sample_with_fallback
        content = await self._sample_continuation(
            previous_thought.content,
            prev_phase,
            guidance,
        )
        thought_type, confidence = self._get_phase_metadata(prev_phase)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.ZERO_SHOT_COT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "steps": self._reasoning_steps,
                "sampled": execution_context is not None and execution_context.can_sample,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    async def _sample_reasoning(self, input_text: str) -> str:
        """Generate reasoning using LLM sampling with fallback.

        Uses the execution context's sampling capability to generate
        actual zero-shot CoT reasoning with the magic trigger phrase.
        Falls back to heuristic generation if sampling is unavailable or fails.

        Args:
            input_text: The problem or question to reason about

        Returns:
            A formatted string containing the sampled or heuristic reasoning
        """
        system_prompt = (
            f"You are a reasoning assistant using Zero-Shot Chain-of-Thought methodology.\n"
            f'Apply the magic phrase "{self.TRIGGER_PHRASE}" to trigger step-by-step reasoning.\n\n'
            f"Generate clear, step-by-step reasoning WITHOUT requiring examples.\n"
            f"Simply think through the problem systematically and show your work."
        )

        user_prompt = f"""Problem: {input_text}

{self.TRIGGER_PHRASE}

Now solve the problem step by step, showing your reasoning clearly."""

        def fallback_generator() -> str:
            return self._generate_reasoning_heuristic(input_text)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # If we got a sampled result (not the heuristic fallback), format it
        if result != fallback_generator():
            return (
                f"Step {self._step_counter}: Apply Zero-Shot Trigger\n\n"
                f"Problem: {input_text}\n\n"
                f'Applying: "{self.TRIGGER_PHRASE}"\n\n{result}'
            )
        return result

    def _generate_reasoning_heuristic(self, input_text: str) -> str:
        """Generate reasoning using heuristic fallback.

        Args:
            input_text: The problem or question to reason about

        Returns:
            A formatted string containing heuristic reasoning
        """
        content = (
            f"Step {self._step_counter}: Apply Zero-Shot Trigger\n\n"
            f"Problem: {input_text}\n\n"
            f"Applying the magic phrase...\n\n"
            f'"{self.TRIGGER_PHRASE}"\n\n'
            f"This simple trigger elicits step-by-step reasoning\n"
            f"without requiring any examples.\n\n"
            f"Next: Generate reasoning steps."
        )
        return content

    async def _sample_continuation(
        self,
        previous_content: str,
        prev_phase: str,
        guidance: str | None,
    ) -> str:
        """Generate continuation using LLM sampling with fallback.

        Args:
            previous_content: Content from the previous thought
            prev_phase: The previous reasoning phase
            guidance: Optional guidance for the next steps

        Returns:
            A formatted string continuing the reasoning
        """
        phase_prompts = {
            "trigger": "Now generate the actual step-by-step reasoning to solve the problem.",
            "reason": "Now extract and formulate the final answer from the reasoning steps.",
            "extract": "Now conclude the analysis and summarize the complete solution.",
        }

        system_prompt = """You are continuing Zero-Shot Chain-of-Thought reasoning.
Build on the previous work and advance to the next phase of reasoning."""

        user_prompt = f"""Previous reasoning:
{previous_content}

Next phase: {phase_prompts.get(prev_phase, "Continue reasoning")}
{f"Guidance: {guidance}" if guidance else ""}

Continue the reasoning process."""

        def fallback_generator() -> str:
            content, _, _ = self._generate_continuation_heuristic(prev_phase, guidance)
            return content

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=1000,
        )

        # If we got a sampled result (not the heuristic fallback), format it
        if result != fallback_generator():
            return f"Step {self._step_counter}: {self._get_phase_name(prev_phase)}\n\n{result}"
        return result

    def _generate_continuation_heuristic(
        self,
        prev_phase: str,
        guidance: str | None,
    ) -> tuple[str, ThoughtType, float]:
        """Generate continuation using heuristic fallback.

        Args:
            prev_phase: The previous reasoning phase
            guidance: Optional guidance for the next steps

        Returns:
            Tuple of (content, thought_type, confidence)
        """
        if prev_phase == "trigger":
            self._current_phase = "reason"
            self._reasoning_steps = [
                "First, I need to understand what is being asked.",
                "Next, I identify the key information.",
                "Then, I apply the relevant knowledge or formula.",
                "Finally, I compute or derive the answer.",
            ]
            content = (
                f"Step {self._step_counter}: Generate Reasoning\n\n"
                f"Thinking step by step:\n\n"
                + "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self._reasoning_steps))
                + "\n\nReasoning chain generated.\n"
                "Next: Extract the final answer."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.75
        elif prev_phase == "reason":
            self._current_phase = "extract"
            content = (
                f"Step {self._step_counter}: Extract Answer\n\n"
                f"From the reasoning steps:\n"
                + "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self._reasoning_steps))
                + "\n\nDerived Answer: [Answer extracted from reasoning]\n"
                "The answer follows directly from the step-by-step analysis."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.8
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Zero-Shot CoT Complete:\n"
                f'  Trigger: "{self.TRIGGER_PHRASE}"\n'
                f"  Steps generated: {len(self._reasoning_steps)}\n\n"
                f"Final Answer: [Answer]\n"
                f"Confidence: Good (80%)\n\n"
                f"Method: Zero-shot chain-of-thought\n"
                f"  - No examples required\n"
                f"  - Simple trigger phrase\n"
                f"  - Step-by-step reasoning elicited"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.8

        return content, thought_type, confidence

    def _get_phase_metadata(self, prev_phase: str) -> tuple[ThoughtType, float]:
        """Get thought type and confidence for a phase transition.

        Args:
            prev_phase: The previous reasoning phase

        Returns:
            Tuple of (thought_type, confidence)
        """
        if prev_phase == "trigger":
            self._current_phase = "reason"
            return ThoughtType.REASONING, 0.75
        elif prev_phase == "reason":
            self._current_phase = "extract"
            return ThoughtType.SYNTHESIS, 0.8
        else:
            self._current_phase = "conclude"
            return ThoughtType.CONCLUSION, 0.8

    def _get_phase_name(self, prev_phase: str) -> str:
        """Get human-readable name for the next phase.

        Args:
            prev_phase: The previous reasoning phase

        Returns:
            Human-readable phase name
        """
        phase_names = {
            "trigger": "Generate Reasoning",
            "reason": "Extract Answer",
            "extract": "Final Answer",
        }
        return phase_names.get(prev_phase, "Continue")


__all__ = ["ZeroShotCoT", "ZERO_SHOT_COT_METADATA"]
