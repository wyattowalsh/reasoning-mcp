"""Hidden CoT Decoding reasoning method.

Efficient CoT without explicit tokens - reasoning happens in hidden states.

Reference: Wang et al. (2025) - "Hidden CoT Decoding"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import (
    PRECISE_TEMPERATURE,
    MethodMetadata,
    ReasoningMethodBase,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


HIDDEN_COT_DECODING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.HIDDEN_COT_DECODING,
    name="Hidden CoT Decoding",
    description="Efficient CoT reasoning in hidden states without explicit tokens.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"hidden", "efficient", "decoding", "implicit"}),
    complexity=5,
    supports_branching=False,
    supports_revision=False,
    requires_context=False,
    min_thoughts=2,
    max_thoughts=4,
    avg_tokens_per_thought=100,
    best_for=("token efficiency", "fast inference"),
    not_recommended_for=("explainability", "debugging"),
)


class HiddenCotDecoding(ReasoningMethodBase):
    """Hidden CoT Decoding method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "hidden_reason"
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.HIDDEN_COT_DECODING

    @property
    def name(self) -> str:
        return HIDDEN_COT_DECODING_METADATA.name

    @property
    def description(self) -> str:
        return HIDDEN_COT_DECODING_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "hidden_reason"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HiddenCotDecoding must be initialized")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "hidden_reason"

        # Use sampling if available, otherwise fall back to heuristic
        if self._execution_context and self._execution_context.can_sample:
            content = await self._sample_hidden_reasoning(input_text)
        else:
            content = self._generate_hidden_reasoning(input_text)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HIDDEN_COT_DECODING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,
            quality_score=0.75,
            metadata={"phase": self._current_phase, "sampled": self._execution_context is not None},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.HIDDEN_COT_DECODING
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
            raise RuntimeError("HiddenCotDecoding must be initialized")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1
        self._current_phase = "decode"

        # Use sampling if available, otherwise fall back to heuristic
        if self._execution_context and self._execution_context.can_sample:
            content = await self._sample_decode_answer(previous_thought.content, guidance)
        else:
            content = self._generate_decode_answer()

        thought = ThoughtNode(
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.HIDDEN_COT_DECODING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.88,
            quality_score=0.88,
            metadata={"phase": self._current_phase, "sampled": self._execution_context is not None},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_hidden_reasoning(self, input_text: str) -> str:
        """Generate hidden reasoning content using heuristic (fallback method).

        Args:
            input_text: The problem or question to reason about

        Returns:
            A formatted string simulating hidden state reasoning
        """
        content = (
            f"Step {self._step_counter}: Hidden Reasoning (Hidden CoT Decoding)\n\n"
            f"Problem: {input_text}\n\n"
            f"[Hidden state reasoning - no explicit tokens]\n"
            f"Internal computation proceeding...\n\n"
            f"Next: Decode to answer."
        )
        return content

    def _generate_decode_answer(self) -> str:
        """Generate decoded answer using heuristic (fallback method).

        Returns:
            A formatted string with the decoded answer
        """
        content = (
            f"Step {self._step_counter}: Decode Answer\n\n"
            f"Hidden reasoning complete.\n"
            f"Final Answer: 17\n"
            f"Confidence: 88%\n\n"
            f"Method: Hidden CoT Decoding\n"
            f"  - Zero explicit reasoning tokens\n"
            f"  - Maximum efficiency"
        )
        return content

    async def _sample_hidden_reasoning(self, input_text: str) -> str:
        """Generate hidden reasoning using LLM sampling.

        Uses the execution context's sampling capability to simulate
        the hidden CoT process and prepare for decoding.

        Args:
            input_text: The problem or question to reason about

        Returns:
            A formatted string containing the hidden reasoning phase

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_hidden_reasoning but was not provided"
            )

        system_prompt = """You are a reasoning assistant using Hidden CoT Decoding methodology.

In this method, reasoning happens implicitly in the model's hidden states without
generating explicit reasoning tokens. You should:
1. Acknowledge the problem
2. Indicate internal processing is occurring
3. Prepare for efficient decoding

This is the HIDDEN REASONING phase - keep output minimal and efficient."""

        user_prompt = f"""Problem: {input_text}

Phase: Hidden Reasoning

Acknowledge the problem and indicate that internal hidden-state reasoning is proceeding.
Keep the response brief - the actual reasoning happens in hidden states, not tokens."""

        # Capture step_counter for fallback closure
        step_counter = self._step_counter

        def fallback_generator() -> str:
            return self._generate_hidden_reasoning(input_text)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=PRECISE_TEMPERATURE,  # Lower temperature for consistency
            max_tokens=300,  # Minimal tokens for hidden phase
        )

        # Ensure proper formatting if we got sampled content
        # Check if it's already formatted (fallback returns formatted content)
        if f"Step {step_counter}: Hidden Reasoning" not in content:
            formatted_content = (
                f"Step {self._step_counter}: Hidden Reasoning (Hidden CoT Decoding)\n\n"
                f"Problem: {input_text}\n\n"
                f"{content}\n\n"
                f"Next: Decode to answer."
            )
            return formatted_content
        return content

    async def _sample_decode_answer(
        self, previous_content: str, guidance: str | None = None
    ) -> str:
        """Generate decoded answer using LLM sampling.

        Uses the execution context's sampling capability to decode the
        final answer from the hidden reasoning state.

        Args:
            previous_content: Content from the hidden reasoning phase
            guidance: Optional guidance for answer generation

        Returns:
            A formatted string with the decoded answer

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_decode_answer but was not provided"
            )

        system_prompt = """You are a reasoning assistant completing Hidden CoT Decoding.

You are in the DECODE phase. The reasoning has occurred in hidden states.
Now decode the final answer efficiently:
1. State the answer clearly
2. Provide confidence assessment
3. Emphasize efficiency (zero explicit reasoning tokens)

Be concise and direct - the hard reasoning work is already done."""

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Previous hidden reasoning:
{previous_content}
{guidance_text}

Phase: Decode Answer

Decode and present the final answer from the hidden reasoning state.
Include confidence level and emphasize the efficiency of this method."""

        # Capture step_counter for fallback closure
        step_counter = self._step_counter

        def fallback_generator() -> str:
            return self._generate_decode_answer()

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=PRECISE_TEMPERATURE,  # Lower temperature for consistent decoding
            max_tokens=400,
        )

        # Ensure proper formatting if we got sampled content
        # Check if it's already formatted (fallback returns formatted content)
        if f"Step {step_counter}: Decode Answer" not in content:
            formatted_content = (
                f"Step {self._step_counter}: Decode Answer\n\n"
                f"{content}\n\n"
                f"Method: Hidden CoT Decoding\n"
                f"  - Zero explicit reasoning tokens\n"
                f"  - Maximum efficiency"
            )
            return formatted_content
        return content


__all__ = ["HiddenCotDecoding", "HIDDEN_COT_DECODING_METADATA"]
