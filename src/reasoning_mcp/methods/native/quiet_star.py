"""Quiet-STaR reasoning method.

This module implements the Quiet-STaR (Self-Taught Reasoner) method, which
generates internal rationales before producing outputs. The key idea is "thinking
before speaking" - generating inner thoughts that guide and improve the final
response through rationale generation, integration, and output phases.

Reference: Zelikman et al. (2024) - "Quiet-STaR: Language Models Can Teach
Themselves to Think Before Speaking"
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

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for Quiet-STaR method
QUIET_STAR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.QUIET_STAR,
    name="Quiet-STaR",
    description="Internal rationale generation before producing outputs. "
    "Generates 'inner thoughts' that guide final response through "
    "rationale → integrate → output phases for improved reasoning quality.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "internal-reasoning",
            "rationale",
            "self-taught",
            "inner-thoughts",
            "staged-reasoning",
            "quality-improvement",
            "think-before-speaking",
        }
    ),
    complexity=7,  # Advanced complexity - internal rationale generation
    supports_branching=False,  # Linear rationale → integrate → output
    supports_revision=False,  # No revision - forward flow
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: rationale + integrate + output
    max_thoughts=5,  # rationale + integrate + output (+ optional elaboration)
    avg_tokens_per_thought=450,  # Moderate - internal rationales
    best_for=(
        "complex reasoning tasks",
        "quality-critical outputs",
        "self-improving systems",
        "careful analysis",
        "thoughtful responses",
        "internal deliberation",
        "rationale-driven reasoning",
    ),
    not_recommended_for=(
        "simple factual queries",
        "time-critical decisions",
        "tasks requiring speed over quality",
        "direct responses without reasoning",
    ),
)

logger = structlog.get_logger(__name__)


class QuietStar(ReasoningMethodBase):
    """Quiet-STaR reasoning method implementation.

    This class implements the Quiet-STaR pattern where the system generates
    internal rationales before producing outputs. The method proceeds through:
    1. Rationale generation: Create internal reasoning thoughts
    2. Integration: Integrate rationales with the problem context
    3. Output: Produce final response informed by internal thoughts

    Key characteristics:
    - Internal rationale generation
    - "Think before speaking" approach
    - Staged reasoning process
    - Integration of inner thoughts
    - Quality-focused outputs
    - Advanced complexity (7)

    The method tracks rationale tokens, integration weights, and inner thoughts
    to ensure high-quality reasoning before final output.

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = QuietStar()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain quantum entanglement"
        ... )
        >>> print(result.type)  # ThoughtType.REASONING (rationale phase)

        Continue with integration:
        >>> integrate = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Integrate rationale with context"
        ... )
        >>> print(integrate.type)  # ThoughtType.SYNTHESIS (integrate phase)

        Continue to final output:
        >>> output = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=integrate,
        ...     guidance="Generate final response"
        ... )
        >>> print(output.type)  # ThoughtType.CONCLUSION (output phase)
    """

    # Maximum tokens for internal rationale generation
    MAX_RATIONALE_TOKENS = 512
    # Threshold for integration weight (how well rationale fits)
    INTEGRATION_THRESHOLD = 0.7
    # Maximum inner thoughts to generate
    MAX_INNER_THOUGHTS = 3
    # Enable LLM sampling for this method
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Quiet-STaR method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "rationale"  # rationale, integrate, output
        self._rationale_tokens = 0
        self._integration_weight = 0.0
        self._inner_thoughts: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.QUIET_STAR

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return QUIET_STAR_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return QUIET_STAR_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Quiet-STaR method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = QuietStar()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "rationale"
            >>> assert method._rationale_tokens == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "rationale"
        self._rationale_tokens = 0
        self._integration_weight = 0.0
        self._inner_thoughts = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Quiet-STaR method.

        This method creates the initial internal rationale generation phase.
        It generates inner thoughts that will guide the final response.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include max_rationale_tokens)

        Returns:
            A ThoughtNode representing the initial rationale generation

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = QuietStar()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="What causes lightning?"
            ... )
            >>> assert thought.type == ThoughtType.REASONING
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.QUIET_STAR
            >>> assert "phase" in thought.metadata
            >>> assert thought.metadata["phase"] == "rationale"
        """
        if not self._initialized:
            raise RuntimeError("Quiet-STaR method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "rationale"
        self._rationale_tokens = 0
        self._integration_weight = 0.0
        self._inner_thoughts = []

        # Extract max_rationale_tokens from context if provided
        max_rationale_tokens = self.MAX_RATIONALE_TOKENS
        if context and "max_rationale_tokens" in context:
            max_rationale_tokens = max(min(context["max_rationale_tokens"], 2048), 128)

        # Generate internal rationale
        content, inner_thought = await self._generate_rationale(
            input_text, context, max_rationale_tokens
        )

        # Store inner thought
        self._inner_thoughts.append(inner_thought)

        # Estimate rationale tokens (rough approximation)
        self._rationale_tokens = int(len(content.split()) * 1.3)  # ~1.3 tokens per word

        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.QUIET_STAR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.65,  # Moderate initial confidence
            quality_score=0.7,  # Good starting quality for rationale
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "quiet_star",
                "phase": self._current_phase,
                "rationale_tokens": self._rationale_tokens,
                "max_rationale_tokens": max_rationale_tokens,
                "inner_thoughts_count": len(self._inner_thoughts),
                "inner_thought": inner_thought,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.QUIET_STAR

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

        This method implements the Quiet-STaR phase progression:
        - If previous was rationale: generate integration
        - If previous was integration: generate final output
        - If previous was output: can optionally elaborate further

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the Quiet-STaR process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = QuietStar()
            >>> await method.initialize()
            >>> rationale = await method.execute(session, "Explain DNA")
            >>> integrate = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=rationale
            ... )
            >>> assert integrate.type == ThoughtType.SYNTHESIS
            >>> assert integrate.metadata["phase"] == "integrate"
            >>>
            >>> output = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=integrate
            ... )
            >>> assert output.type == ThoughtType.CONCLUSION
            >>> assert output.metadata["phase"] == "output"
        """
        if not self._initialized:
            raise RuntimeError("Quiet-STaR method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "rationale")

        if prev_phase == "rationale":
            # Next: integrate rationale with context
            self._current_phase = "integrate"
            thought_type = ThoughtType.SYNTHESIS
            content, integration_weight = await self._generate_integration(
                previous_thought, guidance, context
            )
            self._integration_weight = integration_weight

            confidence = min(0.7 + (integration_weight * 0.2), 0.95)  # Higher if well-integrated
            quality_score = min(0.75 + (integration_weight * 0.15), 0.95)

        elif prev_phase == "integrate":
            # Next: generate final output
            self._current_phase = "output"
            thought_type = ThoughtType.CONCLUSION
            content = await self._generate_output(previous_thought, guidance, context)

            # High confidence and quality for final output
            confidence = min(0.8 + (self._integration_weight * 0.15), 0.98)
            quality_score = min(0.85 + (self._integration_weight * 0.1), 0.98)

        elif prev_phase == "output":
            # Optional: elaborate on output
            self._current_phase = "elaborate"
            thought_type = ThoughtType.CONTINUATION
            content = await self._generate_elaboration(previous_thought, guidance, context)

            confidence = 0.85
            quality_score = 0.9

        else:
            # Fallback to integration
            self._current_phase = "integrate"
            thought_type = ThoughtType.SYNTHESIS
            content, integration_weight = await self._generate_integration(
                previous_thought, guidance, context
            )
            self._integration_weight = integration_weight
            confidence = 0.75
            quality_score = 0.8

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.QUIET_STAR,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "quiet_star",
                "rationale_tokens": self._rationale_tokens,
                "integration_weight": self._integration_weight,
                "inner_thoughts_count": len(self._inner_thoughts),
                "previous_phase": prev_phase,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Quiet-STaR, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = QuietStar()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _generate_rationale(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Generate internal rationale for the input.

        This is a helper method that would typically call an LLM to generate
        internal reasoning thoughts before producing output.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context
            max_tokens: Maximum tokens for rationale

        Returns:
            Tuple of (content for thought, inner thought summary)

        Note:
            In a full implementation, this would use an LLM to generate
            the actual internal rationale. This is a placeholder that
            provides the structure.
        """

        # Define fallback content generator
        def generate_fallback() -> str:
            return (
                f"Before generating a response, I will first think through this problem "
                f"internally. This rationale generation phase allows me to organize my "
                f"thoughts and ensure quality reasoning before producing output.\n\n"
                f"Internal Rationale:\n"
                f"- Analyzing the core question and identifying key components\n"
                f"- Activating relevant knowledge and mental models\n"
                f"- Considering potential approaches and their implications\n"
                f"- Identifying uncertainties and areas needing careful thought\n"
                f"- Planning the structure of my response\n\n"
                f"Rationale tokens: ~{max_tokens // 2} (budget: {max_tokens})"
            )

        prompt = (
            f"Generate an internal rationale for this problem before producing output.\n\n"
            f"Problem: {input_text}\n\n"
            f"Provide your internal thinking process:\n"
            f"- Analyze the core question and identify key components\n"
            f"- Activate relevant knowledge and mental models\n"
            f"- Consider potential approaches and their implications\n"
            f"- Identify uncertainties and areas needing careful thought\n"
            f"- Plan the structure of your response\n\n"
            f"Keep your rationale within approximately {max_tokens} tokens."
        )

        system_prompt = (
            "You are generating internal rationales for the Quiet-STaR method. "
            "Think deeply before speaking. Generate thoughtful, internal reasoning "
            "that will guide high-quality output. Focus on analyzing the problem, "
            "activating relevant knowledge, and planning your approach."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=system_prompt,
        )

        # Check if we got LLM result or fallback
        is_llm_result = "Rationale tokens:" not in result

        if is_llm_result:
            inner_thought = (
                f"LLM-generated rationale for: {input_text[:50]}... "
                f"[Analyzed problem structure and planned response approach]"
            )
        else:
            inner_thought = (
                f"Internal rationale for: {input_text[:50]}... "
                f"Analyzing problem structure, identifying key concepts, "
                f"considering relevant knowledge."
            )

        content = (
            f"Step {self._step_counter}: Internal Rationale Generation\n\n"
            f"Problem: {input_text}\n\n"
            f"[Internal Thinking - Before Speaking]\n\n"
            f"{result}\n\n"
            f"Rationale tokens budget: {max_tokens}\n\n"
            f"Inner thought: {inner_thought}\n\n"
            f"Note: This internal reasoning will guide the quality of my final response."
        )

        return content, inner_thought

    async def _generate_integration(
        self,
        rationale_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """Generate integration of rationale with problem context.

        This is a helper method that would typically call an LLM to integrate
        the internal rationale with the problem context and prepare for output.

        Args:
            rationale_thought: The rationale thought to integrate
            guidance: Optional guidance for integration
            context: Optional additional context

        Returns:
            Tuple of (content for thought, integration weight)

        Note:
            In a full implementation, this would use an LLM to perform
            actual integration. This is a placeholder that provides
            the structure.
        """
        inner_thought = rationale_thought.metadata.get("inner_thought", "")
        base_weight = rationale_thought.quality_score or 0.7

        # Define fallback content generator
        def generate_fallback() -> str:
            return (
                "Integration Process:\n"
                "- Connecting internal rationale to explicit problem requirements\n"
                "- Validating reasoning chain for logical consistency\n"
                "- Weighting different aspects of the rationale by relevance\n"
                "- Preparing key insights for final output\n"
                "- Ensuring coherent flow from thought to expression"
            )

        rationale_content = rationale_thought.content

        prompt = (
            f"Integrate the following internal rationale with the problem context.\n\n"
            f"Previous rationale:\n{rationale_content}\n\n"
            f"Inner thought: {inner_thought}\n\n"
        )

        if guidance:
            prompt += f"Guidance: {guidance}\n\n"

        prompt += (
            "Perform the integration by:\n"
            "- Connecting internal rationale to explicit problem requirements\n"
            "- Validating reasoning chain for logical consistency\n"
            "- Weighting different aspects of the rationale by relevance\n"
            "- Preparing key insights for final output\n"
            "- Ensuring coherent flow from thought to expression\n\n"
            "Provide your integrated analysis."
        )

        system_prompt = (
            "You are integrating internal rationale with problem context "
            "for the Quiet-STaR method. Ensure logical consistency and "
            "prepare the reasoning for final output. Focus on connecting "
            "thoughts to requirements and validating the reasoning chain."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=system_prompt,
        )

        # Check if we got LLM result or fallback (fallback contains "Integration Process:")
        is_llm_result = "Integration Process:" not in result

        # Calculate integration weight based on rationale quality
        if is_llm_result:
            integration_weight = min(base_weight + 0.15, 1.0)  # Boost for LLM integration
        else:
            integration_weight = min(base_weight + 0.1, 1.0)

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        status = (
            "Well-integrated"
            if integration_weight >= self.INTEGRATION_THRESHOLD
            else "Partial integration"
        )
        content = (
            f"Step {self._step_counter}: Integration Phase\n\n"
            f"Integrating internal rationale with problem context...\n\n"
            f"Previous inner thought: {inner_thought}\n\n"
            f"{result}\n\n"
            f"Integration Weight: {integration_weight:.3f} "
            f"(threshold: {self.INTEGRATION_THRESHOLD})\n\n"
            f"Status: {status}\n\n"
            f"The rationale is now integrated and ready to inform "
            f"the final output.{guidance_text}"
        )

        return content, integration_weight

    async def _generate_output(
        self,
        integration_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final output based on integrated rationale.

        This is a helper method that would typically call an LLM to generate
        the final output informed by the internal rationale and integration.

        Args:
            integration_thought: The integration thought
            guidance: Optional guidance for output
            context: Optional additional context

        Returns:
            The content for the final output

        Note:
            In a full implementation, this would use an LLM to generate
            the actual output. This is a placeholder that provides
            the structure.
        """
        integration_weight = integration_thought.metadata.get("integration_weight", 0.7)

        # Define fallback content generator
        def generate_fallback() -> str:
            return (
                "[This would contain the LLM-generated final answer, informed by "
                "the internal rationale and integration process. The response benefits "
                "from the 'thinking before speaking' approach, resulting in higher "
                "quality, more thoughtful output.]\n\n"
                "Key insights from internal rationale:\n"
                "- Point 1: [Derived from internal thinking]\n"
                "- Point 2: [Derived from internal thinking]\n"
                "- Point 3: [Derived from internal thinking]"
            )

        integration_content = integration_thought.content

        prompt = (
            f"Generate the final output based on the integrated rationale.\n\n"
            f"Integration:\n{integration_content}\n\n"
            f"Integration quality: {integration_weight:.3f}\n"
            f"Inner thoughts processed: {len(self._inner_thoughts)}\n\n"
        )

        if guidance:
            prompt += f"Guidance: {guidance}\n\n"

        prompt += (
            "Provide your final response, informed by the internal rationale "
            "and integration process. The response should benefit from the "
            "'thinking before speaking' approach, resulting in high quality output."
        )

        system_prompt = (
            "You are generating final output for the Quiet-STaR method. "
            "Your response should be informed by the internal rationale and integration "
            "that came before. Speak with confidence based on your internal deliberation. "
            "Provide a clear, thoughtful, and high-quality answer."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=system_prompt,
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Final Output\n\n"
            f"Based on the internal rationale and integration process, "
            f"here is the final response.\n\n"
            f"[Output - Informed by Internal Thinking]\n\n"
            f"Integration quality: {integration_weight:.3f}\n"
            f"Inner thoughts processed: {len(self._inner_thoughts)}\n"
            f"Rationale tokens used: {int(self._rationale_tokens)}\n\n"
            f"Final Response:\n{result}\n\n"
            f"This response has been carefully constructed through internal "
            f"deliberation before expression.{guidance_text}"
        )

        return content

    async def _generate_elaboration(
        self,
        output_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate optional elaboration on the output.

        This is a helper method that would typically call an LLM to elaborate
        on specific aspects of the output if requested.

        Args:
            output_thought: The output thought to elaborate on
            guidance: Optional guidance for elaboration
            context: Optional additional context

        Returns:
            The content for the elaboration

        Note:
            In a full implementation, this would use an LLM to generate
            actual elaboration. This is a placeholder that provides
            the structure.
        """

        # Define fallback content generator
        def generate_fallback() -> str:
            return (
                "[LLM would expand on specific aspects of the output]\n\n"
                "Additional Insights:\n"
                "- [Derived from the internal reasoning]\n"
                "- [Further considerations]\n"
                "- [Additional implications]\n\n"
                "Further Considerations:\n"
                "[LLM would address any remaining questions or nuances]"
            )

        output_content = output_thought.content

        prompt = (
            f"Elaborate on the previous output with additional detail "
            f"and clarification.\n\n"
            f"Previous output (Step {output_thought.step_number}):\n"
            f"{output_content}\n\n"
            "Provide additional insights and further considerations. "
            "Expand on specific aspects of the output while maintaining "
            "consistency with the internal rationale process."
        )

        if guidance:
            prompt += f"\n\nGuidance: {guidance}"

        system_prompt = (
            "You are elaborating on output for the Quiet-STaR method. "
            "Build on the previous response with additional detail, insights, "
            "and clarifications. Maintain consistency with the internal rationale "
            "that guided the original output."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=system_prompt,
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Elaboration\n\n"
            f"Building on the previous output (Step {output_thought.step_number}), "
            f"here is additional detail and clarification.\n\n"
            f"{result}\n\n"
            f"This elaboration draws on the same internal rationale process "
            f"to maintain consistency and quality.{guidance_text}"
        )

        return content
