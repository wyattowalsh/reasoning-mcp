"""Dialectic reasoning method.

This module implements the dialectic reasoning method using the thesis-antithesis-synthesis
progression. This method is ideal for exploring balanced perspectives on complex questions,
examining opposing viewpoints, and producing nuanced conclusions.
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

# Metadata for Dialectic method
DIALECTIC_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DIALECTIC,
    name="Dialectic",
    description="Thesis-antithesis-synthesis reasoning for balanced analysis. "
    "Explores opposing viewpoints to produce nuanced, higher-order conclusions.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset(
        {
            "dialectic",
            "thesis",
            "antithesis",
            "synthesis",
            "balanced",
            "opposing-views",
            "philosophical",
        }
    ),
    complexity=6,  # Medium-high complexity
    supports_branching=True,  # Supports multiple antitheses
    supports_revision=True,  # Can revise positions
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least thesis, antithesis, synthesis
    max_thoughts=0,  # No limit (can have multiple antitheses and syntheses)
    avg_tokens_per_thought=400,  # Moderate token usage
    best_for=(
        "complex philosophical questions",
        "ethical dilemmas",
        "balanced analysis of opposing views",
        "policy debates",
        "theoretical discussions",
        "nuanced conclusions",
        "critical thinking",
    ),
    not_recommended_for=(
        "simple factual queries",
        "urgent decisions",
        "purely technical problems",
        "when only one perspective exists",
    ),
)


class Dialectic(ReasoningMethodBase):
    """Dialectic reasoning method implementation.

    This class implements the dialectic reasoning pattern: thesis-antithesis-synthesis.
    It explores a position (thesis), then examines opposing viewpoints (antithesis),
    and finally produces a balanced, higher-order conclusion (synthesis).

    Key characteristics:
    - Three-phase dialectical process
    - Considers opposing viewpoints
    - Produces balanced synthesis
    - Supports multiple antitheses (branching)
    - Medium-high complexity (5-6)

    Phases:
    1. Thesis: Establish an initial position or claim
    2. Antithesis: Develop opposing views or counterarguments
    3. Synthesis: Integrate perspectives into a nuanced conclusion

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = Dialectic()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Is artificial intelligence beneficial to humanity?"
        ... )
        >>> print(result.content)  # Thesis

        Continue to antithesis:
        >>> antithesis = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Develop the opposing viewpoint"
        ... )
        >>> print(antithesis.content)  # Antithesis

        Create synthesis:
        >>> synthesis = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=antithesis,
        ...     guidance="Synthesize the perspectives"
        ... )
        >>> print(synthesis.content)  # Synthesis
    """

    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Dialectic method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._phase = "thesis"  # Current phase: thesis, antithesis, synthesis
        self._thesis_id: str | None = None  # ID of the thesis thought
        self._antithesis_count = 0  # Track number of antitheses generated
        self.enable_elicitation = enable_elicitation
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.DIALECTIC

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return DIALECTIC_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return DIALECTIC_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HIGH_VALUE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Dialectic method for execution.
        Resets internal state for dialectical reasoning.

        Examples:
            >>> method = Dialectic()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._phase == "thesis"
        """
        self._initialized = True
        self._step_counter = 0
        self._phase = "thesis"
        self._thesis_id = None
        self._antithesis_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Dialectic method.

        This method creates the thesis - the first phase of dialectical reasoning.
        It establishes an initial position or claim about the input question.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the thesis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Dialectic()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Should we prioritize economic growth over environmental protection?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.DIALECTIC
            >>> assert thought.metadata["phase"] == "thesis"
        """
        if not self._initialized:
            raise RuntimeError("Dialectic method must be initialized before execution")

        # Store execution context for elicitation
        self._execution_context = execution_context

        # Reset state for new execution
        self._step_counter = 1
        self._phase = "thesis"
        self._antithesis_count = 0

        # Determine if we should use sampling
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Generate thesis content
        if use_sampling:
            if execution_context is None:
                raise RuntimeError("execution_context cannot be None when use_sampling is True")
            content = await self._sample_thesis(input_text, context)
        else:
            content = self._generate_thesis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIALECTIC,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Moderate initial confidence
            metadata={
                "input": input_text,
                "context": context or {},
                "phase": "thesis",
                "reasoning_type": "dialectic",
            },
        )

        # Store thesis ID for reference
        self._thesis_id = thought.id

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DIALECTIC

        # Move to antithesis phase
        self._phase = "antithesis"

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

        This method generates the next phase in the dialectical process.
        Based on the current phase and guidance, it produces either an
        antithesis (opposing view) or synthesis (integrated conclusion).

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the dialectical reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Dialectic()
            >>> await method.initialize()
            >>> thesis = await method.execute(session, "Question")
            >>> antithesis = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=thesis,
            ...     guidance="Develop the opposing viewpoint"
            ... )
            >>> assert antithesis.type == ThoughtType.BRANCH
            >>> assert antithesis.metadata["phase"] == "antithesis"
            >>>
            >>> synthesis = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=antithesis,
            ...     guidance="Synthesize the perspectives"
            ... )
            >>> assert synthesis.type == ThoughtType.SYNTHESIS
            >>> assert synthesis.metadata["phase"] == "synthesis"
        """
        if not self._initialized:
            raise RuntimeError("Dialectic method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Determine phase from guidance or current state
        phase = self._determine_phase(guidance, previous_thought)

        # Optional elicitation: ask user which dialectical direction to explore
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "thesis", "label": "Develop thesis argument further"},
                    {"id": "antithesis", "label": "Develop antithesis argument further"},
                    {"id": "synthesis", "label": "Attempt synthesis of both positions"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "Which dialectical direction should we explore?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    # Override phase based on user selection
                    phase = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="continue_reasoning",
                    error=str(e),
                )
                # Fall back to default behavior

        # Determine if we should use sampling
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Generate content based on phase
        if phase == "antithesis":
            if use_sampling:
                if execution_context is None:
                    raise RuntimeError("execution_context cannot be None when use_sampling is True")
                content = await self._sample_antithesis(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            else:
                content = self._generate_antithesis(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            thought_type = ThoughtType.BRANCH
            self._antithesis_count += 1
            # Antitheses branch from the thesis
            parent_id = self._thesis_id or previous_thought.id
            depth = 1  # Same depth as thesis
            confidence = 0.7  # Equal confidence to thesis
            branch_id = f"antithesis_{self._antithesis_count}"

        elif phase == "synthesis":
            if use_sampling:
                if execution_context is None:
                    raise RuntimeError("execution_context cannot be None when use_sampling is True")
                content = await self._sample_synthesis(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            else:
                content = self._generate_synthesis(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            thought_type = ThoughtType.SYNTHESIS
            parent_id = previous_thought.id
            depth = previous_thought.depth + 1
            # Higher confidence for synthesis (integrates multiple views)
            confidence = min(0.9, previous_thought.confidence + 0.15)
            branch_id = None

        else:  # Further refinement or continuation
            if use_sampling:
                if execution_context is None:
                    raise RuntimeError("execution_context cannot be None when use_sampling is True")
                content = await self._sample_continuation(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            else:
                content = self._generate_continuation(
                    previous_thought=previous_thought,
                    guidance=guidance,
                    context=context,
                )
            thought_type = ThoughtType.CONTINUATION
            parent_id = previous_thought.id
            depth = previous_thought.depth + 1
            confidence = previous_thought.confidence
            branch_id = previous_thought.branch_id

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.DIALECTIC,
            content=content,
            parent_id=parent_id,
            step_number=self._step_counter,
            depth=depth,
            confidence=confidence,
            branch_id=branch_id,
            metadata={
                "phase": phase,
                "previous_phase": previous_thought.metadata.get("phase", "unknown"),
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "dialectic",
            },
        )

        # Add to session
        session.add_thought(thought)

        # Update phase state
        self._phase = phase

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Dialectic reasoning, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Dialectic()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _determine_phase(
        self,
        guidance: str | None,
        previous_thought: ThoughtNode,
    ) -> str:
        """Determine the next phase based on guidance and previous thought.

        Args:
            guidance: Optional guidance for the next step
            previous_thought: The previous thought in the chain

        Returns:
            The next phase: "antithesis", "synthesis", or "continuation"
        """
        # Check guidance for explicit phase keywords
        if guidance:
            guidance_lower = guidance.lower()
            if any(
                word in guidance_lower
                for word in ["antithesis", "opposing", "counter", "alternative"]
            ):
                return "antithesis"
            if any(
                word in guidance_lower
                for word in ["synthesis", "synthesize", "integrate", "combine", "conclude"]
            ):
                return "synthesis"

        # Infer from previous phase
        previous_phase = previous_thought.metadata.get("phase", "")

        # Phase transition mapping: thesis -> antithesis -> synthesis -> continuation
        phase_map = {
            "thesis": "antithesis",  # After thesis, default to antithesis
            "antithesis": "synthesis",  # After antithesis, default to synthesis
            "synthesis": "continuation",  # After synthesis, continue refinement
        }

        # Default to continuation for any other phase
        return phase_map.get(previous_phase, "continuation")

    def _generate_thesis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the thesis content.

        This is a helper method that would typically call an LLM or reasoning engine.
        For now, it returns a template that can be filled by the actual implementation.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the thesis

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: THESIS - Establishing Initial Position\n\n"
            f"Question: {input_text}\n\n"
            f"THESIS:\n"
            f"Let me begin by establishing a clear position on this question. "
            f"I will present the strongest arguments for one perspective, "
            f"acknowledging that this is one side of a multifaceted issue that "
            f"warrants dialectical examination."
        )

    def _generate_antithesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate antithesis thought content.

        This is a helper method that would typically call an LLM or reasoning engine
        to generate opposing viewpoints to the thesis.

        Args:
            previous_thought: The thought to build upon (typically the thesis)
            guidance: Optional guidance for the antithesis
            context: Optional additional context

        Returns:
            The content for the antithesis thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        antithesis_num = self._antithesis_count + 1

        return (
            f"Step {self._step_counter}: ANTITHESIS {antithesis_num} - Examining the Opposition\n\n"
            f"Now I will critically examine the opposing perspective. "
            f"This antithesis challenges the thesis by:\n"
            f"1. Identifying weaknesses or limitations in the thesis\n"
            f"2. Presenting alternative evidence or reasoning\n"
            f"3. Offering a fundamentally different viewpoint\n\n"
            f"This opposition is not merely contrarian, but seeks to illuminate "
            f"aspects of the question that the thesis may have overlooked.{guidance_text}"
        )

    def _generate_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate synthesis thought content.

        This is a helper method that would typically call an LLM or reasoning engine
        to generate a higher-order synthesis integrating thesis and antithesis.

        Args:
            previous_thought: The thought to build upon (typically an antithesis)
            guidance: Optional guidance for the synthesis
            context: Optional additional context

        Returns:
            The content for the synthesis thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: SYNTHESIS - Integrating Perspectives\n\n"
            f"Having examined both the thesis and its antithesis, I now synthesize "
            f"these perspectives into a higher-order understanding:\n\n"
            f"The synthesis:\n"
            f"1. Acknowledges the partial truths in both thesis and antithesis\n"
            f"2. Resolves contradictions where possible\n"
            f"3. Identifies a more nuanced position that transcends the original binary\n"
            f"4. Recognizes complexities and contextual dependencies\n\n"
            f"This synthesis represents not a mere compromise, but an evolved "
            f"understanding that incorporates insights from both perspectives.{guidance_text}"
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation thought content.

        This is a helper method that would typically call an LLM or reasoning engine
        to generate further refinement or elaboration.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the continuation
            context: Optional additional context

        Returns:
            The content for the continuation thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning content. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        previous_phase = previous_thought.metadata.get("phase", "unknown")

        return (
            f"Step {self._step_counter}: Continuing Dialectical Analysis\n\n"
            f"Building on the {previous_phase} from step {previous_thought.step_number}, "
            f"let me further develop this dialectical reasoning. "
            f"This continuation deepens our understanding by exploring implications, "
            f"examining edge cases, or considering additional perspectives.{guidance_text}"
        )

    async def _sample_thesis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate thesis content using LLM sampling.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled content for the thesis
        """
        system_prompt = """You are a reasoning assistant using dialectic methodology.
Generate a THESIS - an initial position on the question.

Structure your thesis with:
1. Clear statement of the position
2. Strongest arguments supporting this perspective
3. Evidence and reasoning
4. Acknowledgment that this is one side of a multifaceted issue

Be thorough but balanced, presenting the strongest case for this perspective."""

        user_prompt = f"""Question: {input_text}

Generate a thesis that establishes a clear, well-reasoned position on this question.
Present the strongest arguments for one perspective."""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_thesis(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Check if we got a fallback response (which already includes the step header)
        if content.startswith(f"Step {step_counter}:"):
            return content
        return f"Step {step_counter}: THESIS - Establishing Initial Position\n\n{content}"

    async def _sample_antithesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate antithesis content using LLM sampling.

        Args:
            previous_thought: The thought to build upon (typically the thesis)
            guidance: Optional guidance for the antithesis
            context: Optional additional context

        Returns:
            The sampled content for the antithesis
        """
        antithesis_num = self._antithesis_count + 1
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using dialectic methodology.
Generate an ANTITHESIS - an opposing viewpoint to the thesis.

Structure your antithesis with:
1. Identification of weaknesses or limitations in the thesis
2. Alternative evidence or reasoning
3. A fundamentally different perspective
4. Critical examination of assumptions

Be rigorous and substantive, not merely contrarian. Illuminate aspects the thesis may
have overlooked."""

        user_prompt = f"""Previous thesis:
{previous_thought.content}

Generate an antithesis that critically examines and opposes the thesis.
Present a well-reasoned alternative perspective.{guidance_text}"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_antithesis(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Check if we got a fallback response (which already includes the step header)
        if content.startswith(f"Step {step_counter}:"):
            return content
        return (
            f"Step {step_counter}: ANTITHESIS {antithesis_num} - "
            f"Examining the Opposition\n\n{content}"
        )

    async def _sample_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate synthesis content using LLM sampling.

        Args:
            previous_thought: The thought to build upon (typically an antithesis)
            guidance: Optional guidance for the synthesis
            context: Optional additional context

        Returns:
            The sampled content for the synthesis
        """
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using dialectic methodology.
Generate a SYNTHESIS - a higher-order integration of thesis and antithesis.

Structure your synthesis with:
1. Acknowledgment of partial truths in both perspectives
2. Resolution of contradictions where possible
3. A more nuanced position that transcends the original binary
4. Recognition of complexities and contextual dependencies

Create not a mere compromise, but an evolved understanding that incorporates insights
from both perspectives."""

        user_prompt = f"""Previous dialectical reasoning:
{previous_thought.content}

Generate a synthesis that integrates the thesis and antithesis into a higher-order understanding.
Create a nuanced position that transcends the binary opposition.{guidance_text}"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_synthesis(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        # Check if we got a fallback response (which already includes the step header)
        if content.startswith(f"Step {step_counter}:"):
            return content
        return f"Step {step_counter}: SYNTHESIS - Integrating Perspectives\n\n{content}"

    async def _sample_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation content using LLM sampling.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the continuation
            context: Optional additional context

        Returns:
            The sampled content for the continuation
        """
        previous_phase = previous_thought.metadata.get("phase", "unknown")
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant continuing dialectic reasoning.
Build upon the previous dialectical analysis with deeper insights.

Continue by:
1. Exploring implications of the previous phase
2. Examining edge cases or nuances
3. Considering additional perspectives
4. Deepening the dialectical analysis"""

        user_prompt = f"""Previous {previous_phase} (step {previous_thought.step_number}):
{previous_thought.content}

Continue the dialectical reasoning, deepening our understanding of the question.{guidance_text}"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_continuation(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Check if we got a fallback response (which already includes the step header)
        if content.startswith(f"Step {step_counter}:"):
            return content
        return f"Step {step_counter}: Continuing Dialectical Analysis\n\n{content}"
