"""Socratic Reasoning method.

This module implements the Socratic method: question-driven reasoning that
uncovers assumptions, exposes contradictions, and guides toward deeper understanding
through systematic inquiry. This is a teaching and critical thinking method that
leads to self-discovered insights.
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


# Metadata for Socratic Reasoning method
SOCRATIC_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SOCRATIC,
    name="Socratic Reasoning",
    description="Question-driven reasoning that uncovers assumptions, exposes "
    "contradictions, and guides toward deeper understanding through systematic inquiry. "
    "Uses probing questions to lead to self-discovered insights.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({
        "socratic",
        "questioning",
        "critical-thinking",
        "assumptions",
        "contradictions",
        "discovery",
        "dialogue",
        "teaching",
        "inquiry",
    }),
    complexity=4,  # Medium-high complexity
    supports_branching=False,  # Linear dialogue progression
    supports_revision=True,  # Can revise understanding
    requires_context=False,  # No special context needed
    min_thoughts=3,  # Need multiple questions for dialogue
    max_thoughts=0,  # No limit - dialogue can continue
    avg_tokens_per_thought=400,  # Questions and responses
    best_for=(
        "uncovering assumptions",
        "critical thinking development",
        "teaching and learning",
        "exposing contradictions",
        "guided self-discovery",
        "examining beliefs",
        "conceptual clarification",
        "philosophical inquiry",
    ),
    not_recommended_for=(
        "quick factual answers",
        "routine problem solving",
        "time-sensitive decisions",
        "simple calculations",
        "when direct answers are needed",
    ),
)


class SocraticReasoning:
    """Socratic Reasoning method implementation.

    This class implements the Socratic method of reasoning through systematic
    questioning. It probes assumptions, challenges beliefs, exposes contradictions,
    and guides toward deeper understanding through a dialogue of inquiry.

    Key characteristics:
    - Question-driven discovery
    - Assumption challenging
    - Contradiction exposure
    - Guided insight development
    - Self-discovery focus
    - Progressive questioning

    The method progresses through several questioning phases:
    1. Initial probing questions
    2. Assumption identification
    3. Contradiction examination
    4. Guided exploration
    5. Synthesis and insight

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SocraticReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Democracy is the best form of government"
        ... )
        >>> print(result.content)  # Initial probing questions

        Continue reasoning:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="What do you mean by 'best'?"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    # Questioning phases for Socratic dialogue
    PHASE_INITIAL = "initial_probing"
    PHASE_ASSUMPTIONS = "assumption_identification"
    PHASE_CONTRADICTIONS = "contradiction_examination"
    PHASE_EXPLORATION = "guided_exploration"
    PHASE_SYNTHESIS = "insight_synthesis"

    def __init__(self) -> None:
        """Initialize the Socratic Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase = self.PHASE_INITIAL
        self._identified_assumptions: list[str] = []
        self._identified_contradictions: list[str] = []

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SOCRATIC

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SOCRATIC_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SOCRATIC_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Socratic Reasoning method for execution.
        Resets all internal state for a fresh dialogue.

        Examples:
            >>> method = SocraticReasoning()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = self.PHASE_INITIAL
        self._identified_assumptions = []
        self._identified_contradictions = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the Socratic Reasoning method.

        This method creates the first thought in a Socratic dialogue,
        starting with probing questions to explore the claim or problem.

        Args:
            session: The current reasoning session
            input_text: The claim, belief, or problem to examine
            context: Optional additional context

        Returns:
            A ThoughtNode representing the initial probing questions

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SocraticReasoning()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="All people want freedom"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SOCRATIC
        """
        if not self._initialized:
            raise RuntimeError(
                "Socratic Reasoning method must be initialized before execution"
            )

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = self.PHASE_INITIAL
        self._identified_assumptions = []
        self._identified_contradictions = []

        # Create initial probing questions
        content = self._generate_initial_probing(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SOCRATIC,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Lower initial confidence - questioning process
            metadata={
                "input": input_text,
                "context": context or {},
                "phase": self._current_phase,
                "reasoning_type": "socratic",
                "questioning_focus": "initial_probing",
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SOCRATIC

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue Socratic reasoning from a previous thought.

        This method generates the next question or exploration in the dialogue,
        progressing through the Socratic questioning phases based on what has
        been uncovered so far.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance or response to previous questions
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the Socratic dialogue

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SocraticReasoning()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Knowledge is power")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Power to do what?"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError(
                "Socratic Reasoning method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Progress through questioning phases
        self._advance_phase()

        # Generate next question or exploration
        content = self._generate_continuation(
            previous_thought=previous_thought,
            guidance=guidance,
            context=context,
        )

        # Confidence may increase as we uncover deeper understanding
        confidence = min(0.9, 0.6 + (0.05 * self._step_counter))

        # Determine thought type based on phase
        thought_type = self._get_thought_type_for_phase()

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SOCRATIC,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "previous_step": previous_thought.step_number,
                "guidance": guidance or "",
                "context": context or {},
                "phase": self._current_phase,
                "reasoning_type": "socratic",
                "assumptions_identified": len(self._identified_assumptions),
                "contradictions_found": len(self._identified_contradictions),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Socratic Reasoning, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SocraticReasoning()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _advance_phase(self) -> None:
        """Advance to the next questioning phase based on step count.

        The Socratic method progresses through distinct phases:
        - Steps 1-2: Initial probing
        - Steps 3-4: Assumption identification
        - Steps 5-6: Contradiction examination
        - Steps 7-8: Guided exploration
        - Steps 9+: Insight synthesis
        """
        if self._step_counter <= 2:
            self._current_phase = self.PHASE_INITIAL
        elif self._step_counter <= 4:
            self._current_phase = self.PHASE_ASSUMPTIONS
        elif self._step_counter <= 6:
            self._current_phase = self.PHASE_CONTRADICTIONS
        elif self._step_counter <= 8:
            self._current_phase = self.PHASE_EXPLORATION
        else:
            self._current_phase = self.PHASE_SYNTHESIS

    def _get_thought_type_for_phase(self) -> ThoughtType:
        """Get the appropriate thought type for the current phase.

        Returns:
            The ThoughtType that best represents the current phase
        """
        phase_to_type = {
            self.PHASE_INITIAL: ThoughtType.CONTINUATION,
            self.PHASE_ASSUMPTIONS: ThoughtType.HYPOTHESIS,
            self.PHASE_CONTRADICTIONS: ThoughtType.VERIFICATION,
            self.PHASE_EXPLORATION: ThoughtType.CONTINUATION,
            self.PHASE_SYNTHESIS: ThoughtType.SYNTHESIS,
        }
        return phase_to_type.get(self._current_phase, ThoughtType.CONTINUATION)

    def _generate_initial_probing(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial probing questions.

        This is a helper method that creates the opening Socratic questions
        to begin the dialogue and exploration.

        Args:
            input_text: The claim or problem to examine
            context: Optional additional context

        Returns:
            The content for the initial probing questions

        Note:
            In a full implementation, this would use an LLM to generate
            contextually appropriate questions. This is a structured template.
        """
        return (
            f"Step {self._step_counter}: Initial Probing Questions\n\n"
            f"Examining the claim: '{input_text}'\n\n"
            f"Let me begin by asking some fundamental questions:\n\n"
            f"1. What exactly do you mean by the key terms in this statement?\n"
            f"2. What assumptions underlie this claim?\n"
            f"3. Is this always true, or are there exceptions?\n"
            f"4. What evidence supports this claim?\n"
            f"5. Have you considered alternative perspectives?\n\n"
            f"These questions help us explore the foundations of the claim "
            f"and uncover any hidden assumptions we should examine."
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation content based on the current phase.

        This is a helper method that creates phase-appropriate questions
        and explorations for the Socratic dialogue.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance or response to previous questions
            context: Optional additional context

        Returns:
            The content for the continuation thought

        Note:
            In a full implementation, this would use an LLM to generate
            contextually appropriate questions and insights based on the
            dialogue history and current phase.
        """
        guidance_text = f"\n\nBased on your response: {guidance}\n" if guidance else ""

        if self._current_phase == self.PHASE_INITIAL:
            return (
                f"Step {self._step_counter}: Continuing Initial Probing{guidance_text}\n"
                f"Let me probe deeper into the meaning:\n\n"
                f"- Can you clarify what you mean by the central concept?\n"
                f"- How would you define this in your own words?\n"
                f"- Are we using the same definitions, or might we be talking about different things?"
            )

        elif self._current_phase == self.PHASE_ASSUMPTIONS:
            # Track an identified assumption
            if guidance:
                self._identified_assumptions.append(guidance[:100])  # Store summary

            return (
                f"Step {self._step_counter}: Identifying Assumptions{guidance_text}\n"
                f"Let's examine the assumptions we're making:\n\n"
                f"- What are we taking for granted without evidence?\n"
                f"- Why do we believe this assumption is valid?\n"
                f"- What happens if this assumption is false?\n"
                f"- Are there cultural or personal biases influencing this assumption?\n\n"
                f"Assumptions identified so far: {len(self._identified_assumptions)}"
            )

        elif self._current_phase == self.PHASE_CONTRADICTIONS:
            return (
                f"Step {self._step_counter}: Examining Contradictions{guidance_text}\n"
                f"Let's look for potential contradictions:\n\n"
                f"- Does this claim conflict with other things we know to be true?\n"
                f"- Are there internal inconsistencies in the reasoning?\n"
                f"- Can we think of counterexamples that challenge this claim?\n"
                f"- What would someone who disagrees say, and how would we respond?\n\n"
                f"The goal is not to win an argument, but to test the strength "
                f"of the reasoning through rigorous examination."
            )

        elif self._current_phase == self.PHASE_EXPLORATION:
            return (
                f"Step {self._step_counter}: Guided Exploration{guidance_text}\n"
                f"Let's explore the implications and connections:\n\n"
                f"- If this is true, what else must be true?\n"
                f"- What are the practical consequences of accepting this claim?\n"
                f"- How does this relate to other knowledge we have?\n"
                f"- Can we find analogies that help clarify or test this reasoning?\n\n"
                f"Through exploration, we often discover insights we hadn't considered."
            )

        else:  # PHASE_SYNTHESIS
            return (
                f"Step {self._step_counter}: Synthesizing Insights{guidance_text}\n"
                f"Through our dialogue, what have we discovered?\n\n"
                f"Key insights:\n"
                f"- Assumptions examined: {len(self._identified_assumptions)}\n"
                f"- Contradictions explored: {len(self._identified_contradictions)}\n"
                f"- Deeper understanding: What new perspective has emerged?\n\n"
                f"The Socratic method shows that true wisdom often lies not in "
                f"having all the answers, but in knowing which questions to ask. "
                f"What is the most important insight you've gained from this inquiry?"
            )
