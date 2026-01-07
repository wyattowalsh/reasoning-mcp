"""Self-Reflection reasoning method.

This module implements a metacognitive reasoning method that performs iterative
self-evaluation and refinement. The method generates an initial response, then
critiques it, identifies weaknesses, and improves through multiple reflection cycles
until a quality threshold is met.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for Self-Reflection method
SELF_REFLECTION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_REFLECTION,
    name="Self-Reflection",
    description="Metacognitive reasoning with iterative self-critique and improvement. "
    "Generates initial response, then evaluates and refines through reflection cycles "
    "until quality threshold is met.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset({
        "metacognitive",
        "self-critique",
        "iterative",
        "refinement",
        "quality-driven",
        "self-evaluation",
        "improvement",
    }),
    complexity=4,  # Medium complexity - iterative refinement requires depth
    supports_branching=False,  # Linear refinement path
    supports_revision=True,  # Core feature - revising through reflection
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: initial + critique + improvement
    max_thoughts=20,  # Multiple reflection cycles possible
    avg_tokens_per_thought=400,  # Moderate - includes analysis
    best_for=(
        "complex problem solving",
        "answer quality optimization",
        "critical thinking tasks",
        "self-improving systems",
        "quality-sensitive outputs",
        "iterative refinement",
        "deep analysis",
    ),
    not_recommended_for=(
        "simple factual queries",
        "time-critical decisions",
        "tasks requiring external validation",
        "problems with clear right answers",
    ),
)


class SelfReflection:
    """Self-Reflection reasoning method implementation.

    This class implements a metacognitive reasoning pattern where the system
    generates an initial response, then iteratively critiques and improves it
    through self-reflection cycles. Each cycle involves:
    1. Generating/reviewing current answer
    2. Critiquing weaknesses and gaps
    3. Scoring quality
    4. Improving based on critique
    5. Repeating until quality threshold met

    Key characteristics:
    - Metacognitive awareness
    - Self-critique capability
    - Iterative refinement
    - Quality-driven termination
    - Revision-based improvement
    - Medium complexity (4-5)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SelfReflection()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain the concept of entropy"
        ... )
        >>> print(result.content)  # Initial response

        Continue with critique:
        >>> critique = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Evaluate and improve"
        ... )
        >>> print(critique.type)  # ThoughtType.VERIFICATION (critique phase)

        Continue with improvement:
        >>> improved = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=critique,
        ...     guidance="Apply improvements"
        ... )
        >>> print(improved.type)  # ThoughtType.REVISION (improvement phase)
    """

    # Quality threshold for completion
    QUALITY_THRESHOLD = 0.8
    # Maximum reflection cycles to prevent infinite loops
    MAX_REFLECTION_CYCLES = 5

    def __init__(self) -> None:
        """Initialize the Self-Reflection method."""
        self._initialized = False
        self._step_counter = 0
        self._reflection_cycle = 0
        self._current_phase: str = "initial"  # initial, critique, improve

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SELF_REFLECTION

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SELF_REFLECTION_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SELF_REFLECTION_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HIGH_VALUE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Self-Reflection method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = SelfReflection()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._reflection_cycle == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._reflection_cycle = 0
        self._current_phase = "initial"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the Self-Reflection method.

        This method creates the initial response that will be iteratively
        refined through self-reflection cycles. It generates a first attempt
        at answering the question or solving the problem.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include quality_threshold)

        Returns:
            A ThoughtNode representing the initial response

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfReflection()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="What is the best approach to learning?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SELF_REFLECTION
            >>> assert "reflection_cycle" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError(
                "Self-Reflection method must be initialized before execution"
            )

        # Reset for new execution
        self._step_counter = 1
        self._reflection_cycle = 0
        self._current_phase = "initial"

        # Extract quality threshold from context if provided
        quality_threshold = self.QUALITY_THRESHOLD
        if context and "quality_threshold" in context:
            quality_threshold = min(max(context["quality_threshold"], 0.0), 1.0)

        # Generate initial response
        content = self._generate_initial_response(input_text, context)

        # Initial quality score (moderate - room for improvement)
        initial_quality = 0.6

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial confidence - will improve through reflection
            quality_score=initial_quality,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "self_reflection",
                "phase": self._current_phase,
                "reflection_cycle": self._reflection_cycle,
                "quality_threshold": quality_threshold,
                "needs_improvement": initial_quality < quality_threshold,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SELF_REFLECTION

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

        This method implements the reflection cycle logic:
        - If previous was initial/improvement: generate critique
        - If previous was critique: generate improvement
        - Continues until quality threshold met or max cycles reached

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the self-reflection process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfReflection()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Explain recursion")
            >>> critique = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert critique.type == ThoughtType.VERIFICATION
            >>> assert critique.metadata["phase"] == "critique"
            >>>
            >>> improvement = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=critique
            ... )
            >>> assert improvement.type == ThoughtType.REVISION
            >>> assert improvement.metadata["phase"] == "improve"
        """
        if not self._initialized:
            raise RuntimeError(
                "Self-Reflection method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Get quality threshold from previous thought's metadata
        quality_threshold = previous_thought.metadata.get(
            "quality_threshold", self.QUALITY_THRESHOLD
        )

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "initial")

        if prev_phase in ("initial", "improve"):
            # Next: critique
            self._current_phase = "critique"
            thought_type = ThoughtType.VERIFICATION
            content = self._generate_critique(previous_thought, guidance, context)

            # Critique identifies weaknesses
            quality_score = previous_thought.quality_score or 0.6
            confidence = 0.7  # Moderate confidence in critique

        elif prev_phase == "critique":
            # Next: improvement
            self._current_phase = "improve"
            self._reflection_cycle += 1
            thought_type = ThoughtType.REVISION
            content = self._generate_improvement(previous_thought, guidance, context)

            # Quality improves with each cycle
            prev_quality = previous_thought.parent_id and session.graph.get_node(
                previous_thought.parent_id
            )
            base_quality = (
                prev_quality.quality_score if prev_quality and prev_quality.quality_score
                else 0.6
            )
            quality_score = min(base_quality + (0.1 * self._reflection_cycle), 1.0)
            confidence = min(0.6 + (0.1 * self._reflection_cycle), 0.95)

        else:
            # Fallback to critique
            self._current_phase = "critique"
            thought_type = ThoughtType.VERIFICATION
            content = self._generate_critique(previous_thought, guidance, context)
            quality_score = previous_thought.quality_score or 0.6
            confidence = 0.7

        # Check if we should continue or conclude
        should_continue = (
            quality_score < quality_threshold
            and self._reflection_cycle < self.MAX_REFLECTION_CYCLES
        )

        # If this is an improvement that meets threshold, mark as conclusion
        if self._current_phase == "improve" and not should_continue:
            thought_type = ThoughtType.CONCLUSION

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SELF_REFLECTION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "reflection_cycle": self._reflection_cycle,
                "quality_threshold": quality_threshold,
                "needs_improvement": should_continue,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "self_reflection",
                "previous_quality": previous_thought.quality_score,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Self-Reflection, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SelfReflection()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_response(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial response to the input.

        This is a helper method that would typically call an LLM to generate
        the initial attempt at solving the problem.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial response

        Note:
            In a full implementation, this would use an LLM to generate
            the actual response. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: Initial Response (Reflection Cycle 0)\n\n"
            f"Problem: {input_text}\n\n"
            f"Let me provide an initial analysis and response to this question. "
            f"I will then reflect on this response to identify areas for improvement "
            f"and iteratively refine my answer.\n\n"
            f"Initial thoughts:\n"
            f"[This would contain the LLM-generated initial response to the question]\n\n"
            f"Note: This is my first attempt. I will now critique and improve this response."
        )

    def _generate_critique(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a self-critique of the previous response.

        This is a helper method that would typically call an LLM to analyze
        the previous response and identify weaknesses, gaps, and areas for improvement.

        Args:
            previous_thought: The response to critique
            guidance: Optional guidance for the critique
            context: Optional additional context

        Returns:
            The content for the critique

        Note:
            In a full implementation, this would use an LLM to generate
            the actual critique. This is a placeholder that provides
            the structure.
        """
        cycle = self._reflection_cycle
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Self-Critique (Reflection Cycle {cycle})\n\n"
            f"Evaluating the previous response (Step {previous_thought.step_number})...\n\n"
            f"Strengths:\n"
            f"[LLM would identify what was done well]\n\n"
            f"Weaknesses:\n"
            f"[LLM would identify gaps, errors, unclear explanations]\n\n"
            f"Missing elements:\n"
            f"[LLM would identify what should be added]\n\n"
            f"Suggested improvements:\n"
            f"[LLM would provide specific recommendations]\n\n"
            f"Quality assessment: {previous_thought.quality_score:.2f}/1.00{guidance_text}"
        )

    def _generate_improvement(
        self,
        critique_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an improved response based on the critique.

        This is a helper method that would typically call an LLM to generate
        an improved version incorporating the critique's feedback.

        Args:
            critique_thought: The critique to address
            guidance: Optional guidance for the improvement
            context: Optional additional context

        Returns:
            The content for the improved response

        Note:
            In a full implementation, this would use an LLM to generate
            the actual improved response. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Improved Response (Reflection Cycle {self._reflection_cycle})\n\n"
            f"Based on the critique in Step {critique_thought.step_number}, "
            f"I will now provide an improved response addressing the identified weaknesses.\n\n"
            f"Improvements applied:\n"
            f"[LLM would address each critique point]\n\n"
            f"Revised response:\n"
            f"[LLM would provide improved, more complete response]\n\n"
            f"Quality: {critique_thought.quality_score:.2f}/1.00 → "
            f"Target: {critique_thought.metadata.get('quality_threshold', 0.8):.2f}/1.00{guidance_text}"
        )
