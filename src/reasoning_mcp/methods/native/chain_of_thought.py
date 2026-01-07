"""Chain of Thought reasoning method implementation.

Chain of Thought (CoT) is a classic reasoning method that generates explicit,
step-by-step reasoning chains showing the "work" of the thought process. Each
step explicitly shows reasoning with connective phrases like "Let me think about
this...", "First, I need to...", "This means that...", and "Therefore...".

This method is particularly effective for:
- Mathematical and logical problems
- Multi-step analysis requiring explicit reasoning
- Problems where showing intermediate steps is valuable
- Teaching and explaining complex reasoning processes

Example reasoning chain:
    1. "Let me think about this problem step by step..."
    2. "First, I need to identify the key variables: X, Y, and Z."
    3. "Given X=5, this means that Y must be greater than 5."
    4. "Therefore, if Y>5 and Z=Y+2, then Z must be at least 8."
    5. "In conclusion, the minimum value of Z is 8."
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)

# Define metadata for the Chain of Thought method
CHAIN_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
    name="Chain of Thought",
    description="Classic step-by-step reasoning with explicit intermediate steps and logical connections",
    category=MethodCategory.CORE,
    tags=frozenset({
        "sequential",
        "explicit",
        "logical",
        "step-by-step",
        "foundational",
        "teaching",
    }),
    complexity=3,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=10,
    avg_tokens_per_thought=200,
    best_for=(
        "Mathematical problems",
        "Logical reasoning",
        "Multi-step analysis",
        "Explicit problem solving",
        "Educational explanations",
        "Sequential decision making",
    ),
    not_recommended_for=(
        "Creative brainstorming",
        "Parallel exploration",
        "Open-ended ideation",
        "Problems requiring multiple perspectives",
    ),
)


class ChainOfThought:
    """Chain of Thought reasoning method implementation.

    This class implements explicit, step-by-step reasoning where each thought
    shows the reasoning process with clear logical connectives. The method
    generates a sequential chain of thoughts that build upon each other,
    making the reasoning process transparent and traceable.

    Attributes:
        identifier: Unique identifier matching MethodIdentifier.CHAIN_OF_THOUGHT
        name: Human-readable name "Chain of Thought"
        description: Brief description of the method
        category: Category as MethodCategory.CORE

    Examples:
        >>> method = ChainOfThought()
        >>> session = Session().start()
        >>> await method.initialize()
        >>> thought = await method.execute(
        ...     session,
        ...     "What is 15% of 240?"
        ... )
        >>> print(thought.content)
        Let me think about this step by step...

        First, I need to convert 15% to a decimal: 15% = 0.15

        Next, I multiply 240 by 0.15: 240 × 0.15

        Breaking this down: 240 × 0.15 = 240 × (0.1 + 0.05)
                                         = 24 + 12
                                         = 36

        Therefore, 15% of 240 equals 36.
    """

    def __init__(self) -> None:
        """Initialize the Chain of Thought reasoning method."""
        self._step_count = 0
        self._is_initialized = False

    @property
    def identifier(self) -> str:
        """Return the unique identifier for this method."""
        return str(MethodIdentifier.CHAIN_OF_THOUGHT)

    @property
    def name(self) -> str:
        """Return the human-readable name of this method."""
        return "Chain of Thought"

    @property
    def description(self) -> str:
        """Return a brief description of this method."""
        return "Classic step-by-step reasoning with explicit intermediate steps and logical connections"

    @property
    def category(self) -> str:
        """Return the category this method belongs to."""
        return str(MethodCategory.CORE)

    async def initialize(self) -> None:
        """Initialize the method.

        For Chain of Thought, initialization is minimal as it doesn't require
        external resources or complex setup.
        """
        self._step_count = 0
        self._is_initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute Chain of Thought reasoning on the input.

        This method generates a reasoning chain with explicit intermediate steps,
        using connective phrases to show the logical flow. Each step builds upon
        the previous ones, creating a coherent reasoning narrative.

        The generated chain includes:
        - Opening statement: "Let me think about this step by step..."
        - First step: "First, I need to..."
        - Intermediate steps: "Next,", "This means that...", "Given that..."
        - Conclusion: "Therefore,", "In conclusion,"

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (max_steps, target_steps, etc.)

        Returns:
            A ThoughtNode containing the complete reasoning chain

        Examples:
            >>> session = Session().start()
            >>> method = ChainOfThought()
            >>> await method.initialize()
            >>> thought = await method.execute(session, "If x+5=12, what is x?")
            >>> assert "Let me think" in thought.content
            >>> assert "First," in thought.content or "first" in thought.content.lower()
            >>> assert "Therefore" in thought.content or "therefore" in thought.content.lower()
        """
        if not self._is_initialized:
            await self.initialize()

        # Extract context parameters
        context = context or {}
        max_steps = context.get("max_steps", 5)
        target_steps = context.get("target_steps", 4)

        # Reset step count for this execution
        self._step_count = 0

        # Generate the chain of thought reasoning
        reasoning_chain = self._generate_reasoning_chain(
            input_text,
            max_steps=max_steps,
            target_steps=target_steps,
        )

        # Determine thought type based on session state
        thought_type = ThoughtType.INITIAL
        parent_id = None
        depth = 0

        # If session has thoughts, this is a continuation
        if session.thought_count > 0:
            thought_type = ThoughtType.CONTINUATION
            # Get the most recent thought as parent
            recent_thoughts = session.get_recent_thoughts(n=1)
            if recent_thoughts:
                parent = recent_thoughts[0]
                parent_id = parent.id
                depth = parent.depth + 1

        # Create the thought node
        thought = ThoughtNode(
            id=str(uuid4()),
            type=thought_type,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content=reasoning_chain,
            parent_id=parent_id,
            depth=depth,
            confidence=0.85,  # High confidence for structured reasoning
            step_number=session.thought_count + 1,
            metadata={
                "reasoning_steps": self._step_count,
                "input_text": input_text,
                "method": "chain_of_thought",
                "explicit_steps": True,
            },
        )

        # Add thought to session
        session.add_thought(thought)

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        Extends the reasoning chain by adding new steps that build upon
        the previous thought's conclusions.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next steps
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the reasoning chain

        Examples:
            >>> # After initial execution
            >>> continuation = await method.continue_reasoning(
            ...     session,
            ...     previous_thought,
            ...     guidance="Now consider the edge cases"
            ... )
            >>> assert continuation.parent_id == previous_thought.id
        """
        if not self._is_initialized:
            await self.initialize()

        # Build continuation text
        continuation_input = guidance or "Continue the reasoning process"

        # Generate continued reasoning
        reasoning_chain = self._generate_reasoning_continuation(
            previous_thought.content,
            continuation_input,
        )

        # Create continuation thought
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content=reasoning_chain,
            parent_id=previous_thought.id,
            depth=previous_thought.depth + 1,
            confidence=0.80,
            step_number=session.thought_count + 1,
            metadata={
                "continued_from": previous_thought.id,
                "guidance": guidance,
                "method": "chain_of_thought",
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and ready, False otherwise

        Examples:
            >>> method = ChainOfThought()
            >>> assert await method.health_check() is False  # Not initialized
            >>> await method.initialize()
            >>> assert await method.health_check() is True  # Now ready
        """
        return self._is_initialized

    def _generate_reasoning_chain(
        self,
        input_text: str,
        max_steps: int = 5,
        target_steps: int = 4,
    ) -> str:
        """Generate a chain of thought reasoning response.

        This is a placeholder implementation that demonstrates the structure
        of a chain of thought response. In a real implementation, this would
        use an LLM to generate the actual reasoning steps.

        Args:
            input_text: The input problem or question
            max_steps: Maximum number of reasoning steps
            target_steps: Target number of reasoning steps to aim for

        Returns:
            A formatted string containing the reasoning chain
        """
        # In a real implementation, this would call an LLM with appropriate prompting
        # For now, we return a structured template showing the expected format

        steps = []
        actual_steps = min(target_steps, max_steps)

        # Opening statement (always present if we have at least 1 step)
        if self._step_count < actual_steps:
            steps.append("Let me think about this step by step...")
            self._step_count += 1

        # First step - problem analysis
        if self._step_count < actual_steps:
            steps.append(f"\nFirst, I need to carefully analyze the problem: {input_text}")
            self._step_count += 1

        # Intermediate steps (would be generated by LLM in real implementation)
        if self._step_count < actual_steps:
            steps.append("\nNext, I'll identify the key components and relationships.")
            self._step_count += 1

        if self._step_count < actual_steps:
            steps.append("\nThis means that I should consider each element systematically.")
            self._step_count += 1

        # Conclusion (always include if within max_steps)
        if self._step_count < max_steps:
            steps.append("\nTherefore, by following this logical progression, I can arrive at a well-reasoned conclusion.")
            self._step_count += 1

        return "\n".join(steps)

    def _generate_reasoning_continuation(
        self,
        previous_content: str,
        continuation_input: str,
    ) -> str:
        """Generate a continuation of the reasoning chain.

        Args:
            previous_content: Content from the previous thought
            continuation_input: The continuation guidance or input

        Returns:
            A formatted string continuing the reasoning chain
        """
        # In a real implementation, this would use an LLM to generate the continuation
        # For now, we return a structured continuation template

        steps = []

        steps.append("Building on the previous reasoning...")
        self._step_count += 1

        steps.append(f"\nGiven the guidance: {continuation_input}")
        self._step_count += 1

        steps.append("\nI'll extend the analysis to address this new aspect.")
        self._step_count += 1

        steps.append("\nTherefore, this additional consideration leads to further insights.")
        self._step_count += 1

        return "\n".join(steps)
