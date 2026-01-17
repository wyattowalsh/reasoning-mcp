"""Sequential Thinking reasoning method.

This module implements the simplest reasoning method: linear, step-by-step thinking
where each thought builds directly on the previous one. This is the foundational
method for straightforward problems that don't require branching or complex exploration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for Sequential Thinking method
SEQUENTIAL_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SEQUENTIAL_THINKING,
    name="Sequential Thinking",
    description="Basic step-by-step reasoning with explicit thought progression. "
    "Each thought builds linearly on the previous one without branching.",
    category=MethodCategory.CORE,
    tags=frozenset(
        {
            "sequential",
            "linear",
            "simple",
            "foundational",
            "step-by-step",
        }
    ),
    complexity=1,  # Simplest method
    supports_branching=False,  # Pure linear progression
    supports_revision=True,  # Can revise previous thoughts
    requires_context=False,  # No special context needed
    min_thoughts=1,
    max_thoughts=0,  # No limit
    avg_tokens_per_thought=300,  # Relatively concise
    best_for=(
        "straightforward problems",
        "clear step-by-step tasks",
        "linear analysis",
        "simple reasoning chains",
        "tutorials and explanations",
    ),
    not_recommended_for=(
        "complex multi-faceted problems",
        "problems requiring parallel exploration",
        "uncertainty and multiple hypotheses",
        "creative brainstorming",
    ),
)


class SequentialThinking(ReasoningMethodBase):
    """Sequential Thinking reasoning method implementation.

    This class implements the simplest reasoning pattern: linear, step-by-step
    thinking where each thought follows directly from the previous one. It's ideal
    for straightforward problems with clear progression.

    Key characteristics:
    - Linear progression (step 1 -> step 2 -> step 3)
    - No branching or parallel paths
    - Each thought builds on the previous
    - Simple and predictable
    - Low computational overhead

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SequentialThinking()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain how to make coffee"
        ... )
        >>> print(result.content)  # First step of reasoning

        Continue reasoning:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Continue with the next step"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    def __init__(self) -> None:
        """Initialize the Sequential Thinking method."""
        self._initialized = False
        self._step_counter = 0
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SEQUENTIAL_THINKING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SEQUENTIAL_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SEQUENTIAL_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.CORE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Sequential Thinking method for execution.
        For this simple method, initialization is minimal.

        Examples:
            >>> method = SequentialThinking()
            >>> await method.initialize()
            >>> assert method._initialized is True
        """
        self._initialized = True
        self._step_counter = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Sequential Thinking method.

        This method creates the first thought in a sequential reasoning chain.
        It analyzes the input and generates an initial step in the reasoning process.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (unused for sequential thinking)

        Returns:
            A ThoughtNode representing the first reasoning step

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SequentialThinking()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How do I solve 2x + 5 = 15?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SEQUENTIAL_THINKING
        """
        if not self._initialized:
            raise RuntimeError("Sequential Thinking method must be initialized before execution")

        # Reset step counter for new execution
        self._step_counter = 1

        # Store execution context and check if sampling is available
        self._execution_context = execution_context
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Create the initial thought
        if use_sampling:
            content = await self._sample_initial_thought(input_text, context)
        else:
            content = self._generate_initial_thought(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Moderate initial confidence
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "sequential",
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SEQUENTIAL_THINKING

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
        """Continue reasoning from a previous thought.

        This method generates the next sequential step, building directly on
        the previous thought. Each continuation increments the step number.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the sequential reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SequentialThinking()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Analyze problem")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="What's the next step?"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
            >>> assert second.depth == 1
        """
        if not self._initialized:
            raise RuntimeError("Sequential Thinking method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Store execution context and check if sampling is available
        self._execution_context = execution_context
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Generate continuation content
        if use_sampling:
            content = await self._sample_continuation(previous_thought, guidance, context)
        else:
            content = self._generate_continuation(
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )

        # Calculate confidence based on depth (may decrease slightly with more steps)
        confidence = max(0.5, previous_thought.confidence - (0.02 * self._step_counter))

        thought = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "previous_step": previous_thought.step_number,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "sequential",
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Sequential Thinking, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SequentialThinking()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_thought(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial thought content.

        This is a helper method that would typically call an LLM or reasoning engine.
        For now, it returns a template that can be filled by the actual implementation.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: Initial analysis\n\n"
            f"Problem: {input_text}\n\n"
            f"Let me begin by understanding what we need to accomplish. "
            f"I'll approach this step-by-step in a clear, sequential manner."
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation thought content.

        This is a helper method that would typically call an LLM or reasoning engine
        to generate the next step based on the previous thought.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The content for the continuation thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Continuing from previous step\n\n"
            f"Building on step {previous_thought.step_number}, "
            f"let me proceed to the next logical step in our reasoning.{guidance_text}"
        )

    async def _sample_initial_thought(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample initial thought content using LLM.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled content for the initial thought
        """
        self._require_execution_context()

        system_prompt = """You are a reasoning assistant using Sequential Thinking methodology.
Analyze problems step-by-step in a clear, linear manner where each thought builds directly
on the previous one.

Your analysis should:
1. Start by understanding the problem clearly
2. Break it down into logical sequential steps
3. Use clear, straightforward reasoning
4. Build each step on the previous one
5. Keep the progression simple and easy to follow"""

        user_prompt = f"""Problem: {input_text}

This is step {self._step_counter} of the sequential reasoning process.

Provide the initial analysis and first reasoning step."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_initial_thought(input_text, context),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=500,
        )

    async def _sample_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample continuation thought content using LLM.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The sampled content for the continuation thought
        """
        self._require_execution_context()

        system_prompt = """You are a reasoning assistant using Sequential Thinking methodology.
Continue the step-by-step reasoning process, building directly on the previous step.

Your continuation should:
1. Reference and build upon the previous step
2. Maintain the logical progression
3. Keep the reasoning clear and sequential
4. Move forward to the next logical step
5. Stay focused on the linear chain of reasoning"""

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Previous step (Step {previous_thought.step_number}):
{previous_thought.content}

This is step {self._step_counter} of the sequential reasoning process.{guidance_text}

Provide the next reasoning step, building directly on the previous step."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_continuation(
                previous_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=500,
        )
