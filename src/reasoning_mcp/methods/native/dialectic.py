"""Dialectic reasoning method.

This module implements the dialectic reasoning method using the thesis-antithesis-synthesis
progression. This method is ideal for exploring balanced perspectives on complex questions,
examining opposing viewpoints, and producing nuanced conclusions.
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


# Metadata for Dialectic method
DIALECTIC_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DIALECTIC,
    name="Dialectic",
    description="Thesis-antithesis-synthesis reasoning for balanced analysis. "
    "Explores opposing viewpoints to produce nuanced, higher-order conclusions.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset({
        "dialectic",
        "thesis",
        "antithesis",
        "synthesis",
        "balanced",
        "opposing-views",
        "philosophical",
    }),
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


class Dialectic:
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

    def __init__(self) -> None:
        """Initialize the Dialectic method."""
        self._initialized = False
        self._step_counter = 0
        self._phase = "thesis"  # Current phase: thesis, antithesis, synthesis
        self._thesis_id: str | None = None  # ID of the thesis thought
        self._antithesis_count = 0  # Track number of antitheses generated

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
    ) -> ThoughtNode:
        """Execute the Dialectic method.

        This method creates the thesis - the first phase of dialectical reasoning.
        It establishes an initial position or claim about the input question.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context

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
            raise RuntimeError(
                "Dialectic method must be initialized before execution"
            )

        # Reset state for new execution
        self._step_counter = 1
        self._phase = "thesis"
        self._antithesis_count = 0

        # Generate thesis content
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
            raise RuntimeError(
                "Dialectic method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Determine phase from guidance or current state
        phase = self._determine_phase(guidance, previous_thought)

        # Generate content based on phase
        if phase == "antithesis":
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

        if previous_phase == "thesis":
            # After thesis, default to antithesis
            return "antithesis"
        elif previous_phase == "antithesis":
            # After antithesis, default to synthesis
            return "synthesis"
        elif previous_phase == "synthesis":
            # After synthesis, continue refinement
            return "continuation"
        else:
            # Default to continuation
            return "continuation"

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
