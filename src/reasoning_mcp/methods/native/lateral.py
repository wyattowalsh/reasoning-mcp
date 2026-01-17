"""Lateral Thinking reasoning method.

This module implements the lateral thinking reasoning method using creative,
divergent problem-solving techniques that challenge assumptions and find
non-obvious solutions. This method is ideal for innovation, breaking mental
blocks, and discovering unconventional approaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models.session import Session


# Metadata for Lateral Thinking method
LATERAL_THINKING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LATERAL_THINKING,
    name="Lateral Thinking",
    description="Creative divergent thinking that challenges assumptions and finds non-obvious solutions",
    category=MethodCategory.HOLISTIC,
    tags=frozenset(
        {
            "creative",
            "divergent",
            "innovation",
            "assumptions",
            "lateral",
            "unconventional",
            "brainstorming",
        }
    ),
    complexity=6,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=5,  # At least: framing, assumption challenge, exploration, connection, synthesis
    max_thoughts=0,  # No limit
    avg_tokens_per_thought=450,
    best_for=(
        "creative problem solving",
        "innovation",
        "breaking mental blocks",
        "finding unconventional solutions",
        "design thinking",
        "ideation",
        "overcoming constraints",
    ),
    not_recommended_for=(
        "precise calculations",
        "well-defined procedural tasks",
        "safety-critical decisions",
        "regulatory compliance",
        "strict logical proofs",
    ),
)


class LateralThinkingMethod(ReasoningMethodBase):
    """Lateral Thinking reasoning method implementation.

    This class implements lateral thinking - a creative, divergent approach to
    problem-solving that challenges conventional assumptions and finds non-obvious
    solutions through techniques like provocation, random entry, reversal, and
    analogy.

    Key characteristics:
    - Challenges hidden assumptions
    - Uses random stimuli to spark ideas
    - Explores reversals and opposites
    - Draws analogies from unrelated domains
    - Uses provocations to break mental patterns
    - Considers alternative perspectives

    Phases:
    1. Problem Framing: Identify constraints and assumptions
    2. Assumption Challenging: Question everything
    3. Divergent Exploration: Generate wild ideas using lateral techniques
    4. Connection Making: Link disparate concepts
    5. Solution Synthesis: Combine insights into solutions

    Lateral Techniques:
    - ASSUMPTION_CHALLENGE: Identify and question hidden assumptions
    - RANDOM_ENTRY: Use random stimuli to spark new ideas
    - REVERSAL: Think about the opposite or reverse
    - ANALOGY: Draw parallels from unrelated domains
    - PROVOCATION: Use "what if" provocations to break patterns
    - ALTERNATIVE_PERSPECTIVE: View from different stakeholders/roles

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = LateralThinkingMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can we reduce traffic congestion?"
        ... )
        >>> print(result.content)  # Problem framing

        Continue with assumption challenge:
        >>> challenge = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Challenge the core assumptions"
        ... )
        >>> print(challenge.content)  # Assumption challenges
    """

    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Lateral Thinking method."""
        self._initialized = False
        self._step_counter = 0
        self._phase = "framing"  # Current phase
        self._root_id: str | None = None  # ID of the root thought
        self._techniques_used: set[str] = set()  # Track which techniques we've used
        self._ideas_generated: list[str] = []  # Track generated ideas
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.LATERAL_THINKING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return LATERAL_THINKING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return LATERAL_THINKING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Lateral Thinking method for execution.
        Resets internal state for creative reasoning.

        Examples:
            >>> method = LateralThinkingMethod()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._phase == "framing"
        """
        self._initialized = True
        self._step_counter = 0
        self._phase = "framing"
        self._root_id = None
        self._techniques_used = set()
        self._ideas_generated = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Lateral Thinking method.

        This method creates the initial problem framing - identifying the core
        problem, constraints, and hidden assumptions to challenge.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the problem framing

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LateralThinkingMethod()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How to increase product sales?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.LATERAL_THINKING
            >>> assert thought.metadata["phase"] == "framing"
        """
        if not self._initialized:
            raise RuntimeError("Lateral Thinking method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._execution_context = execution_context

        # Reset state for new execution
        self._step_counter = 1
        self._phase = "framing"
        self._techniques_used = set()
        self._ideas_generated = []

        # Generate problem framing content
        if execution_context and execution_context.can_sample:
            content = await self._sample_framing(input_text, context)
        else:
            content = self._generate_framing(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LATERAL_THINKING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial moderate confidence
            metadata={
                "input": input_text,
                "context": context or {},
                "phase": "framing",
                "reasoning_type": "lateral_thinking",
                "techniques_available": [
                    "assumption_challenge",
                    "random_entry",
                    "reversal",
                    "analogy",
                    "provocation",
                    "alternative_perspective",
                ],
            },
        )

        # Store root ID for reference
        self._root_id = thought.id

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LATERAL_THINKING

        # Move to next phase
        self._phase = "assumption_challenge"

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

        This method generates the next phase in the lateral thinking process.
        It applies various lateral thinking techniques to generate creative
        insights and unconventional solutions.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the lateral thinking

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LateralThinkingMethod()
            >>> await method.initialize()
            >>> framing = await method.execute(session, "Problem")
            >>> challenge = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=framing,
            ...     guidance="Challenge assumptions"
            ... )
            >>> assert challenge.metadata["phase"] == "assumption_challenge"
        """
        if not self._initialized:
            raise RuntimeError("Lateral Thinking method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Determine phase and technique
        phase, technique = self._determine_phase_and_technique(guidance, previous_thought)

        # Generate content based on phase and technique
        content, thought_type, parent_id, depth, confidence, branch_id = (
            self._generate_thought_details(phase, technique, previous_thought, guidance, context)
        )

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LATERAL_THINKING,
            content=content,
            parent_id=parent_id,
            step_number=self._step_counter,
            depth=depth,
            confidence=confidence,
            branch_id=branch_id,
            metadata={
                "phase": phase,
                "technique": technique,
                "previous_phase": previous_thought.metadata.get("phase", "unknown"),
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "lateral_thinking",
                "techniques_used": list(self._techniques_used),
            },
        )

        # Add to session
        session.add_thought(thought)

        # Update phase state
        self._phase = phase
        if technique:
            self._techniques_used.add(technique)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Lateral Thinking reasoning, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = LateralThinkingMethod()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _determine_phase_and_technique(
        self,
        guidance: str | None,
        previous_thought: ThoughtNode,
    ) -> tuple[str, str]:
        """Determine the next phase and technique based on guidance and previous thought.

        Args:
            guidance: Optional guidance for the next step
            previous_thought: The previous thought in the chain

        Returns:
            Tuple of (phase, technique)
        """
        # Check guidance for explicit technique keywords
        if guidance:
            guidance_lower = guidance.lower()

            # Check for specific techniques
            if any(word in guidance_lower for word in ["assumption", "challenge", "question"]):
                return "assumption_challenge", "assumption_challenge"
            if any(word in guidance_lower for word in ["random", "unrelated", "stimulus"]):
                return "divergent_exploration", "random_entry"
            if any(word in guidance_lower for word in ["reversal", "reverse", "opposite", "flip"]):
                return "divergent_exploration", "reversal"
            if any(word in guidance_lower for word in ["analog", "similar", "like", "parallel"]):
                return "connection_making", "analogy"
            if any(word in guidance_lower for word in ["provocation", "what if", "suppose"]):
                return "divergent_exploration", "provocation"
            if any(word in guidance_lower for word in ["perspective", "viewpoint", "stakeholder"]):
                return "divergent_exploration", "alternative_perspective"
            if any(
                word in guidance_lower for word in ["synthesis", "combine", "integrate", "solution"]
            ):
                return "synthesis", "synthesis"

        # Infer from previous phase
        previous_phase = previous_thought.metadata.get("phase", "")

        if previous_phase == "framing":
            return "assumption_challenge", "assumption_challenge"
        elif previous_phase == "assumption_challenge":
            # Start divergent exploration with first unused technique
            if "random_entry" not in self._techniques_used:
                return "divergent_exploration", "random_entry"
            elif "reversal" not in self._techniques_used:
                return "divergent_exploration", "reversal"
            elif "provocation" not in self._techniques_used:
                return "divergent_exploration", "provocation"
            else:
                return "divergent_exploration", "alternative_perspective"
        elif previous_phase == "divergent_exploration":
            # Continue exploration or move to connection making
            if len(self._techniques_used) < 4:
                # Continue with more exploration techniques
                if "reversal" not in self._techniques_used:
                    return "divergent_exploration", "reversal"
                elif "provocation" not in self._techniques_used:
                    return "divergent_exploration", "provocation"
                elif "alternative_perspective" not in self._techniques_used:
                    return "divergent_exploration", "alternative_perspective"
                else:
                    return "connection_making", "analogy"
            else:
                return "connection_making", "analogy"
        elif previous_phase == "connection_making":
            return "synthesis", "synthesis"
        else:
            # Default to continuation
            return "continuation", "refinement"

    def _generate_thought_details(
        self,
        phase: str,
        technique: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType, str, int, float, str | None]:
        """Generate thought details based on phase and technique.

        Args:
            phase: The current phase
            technique: The technique to use
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (content, thought_type, parent_id, depth, confidence, branch_id)
        """
        if phase == "assumption_challenge":
            content = self._generate_assumption_challenge(previous_thought, guidance, context)
            return (
                content,
                ThoughtType.EXPLORATION,
                previous_thought.id,
                previous_thought.depth + 1,
                0.65,
                None,
            )

        elif phase == "divergent_exploration":
            if technique == "random_entry":
                content = self._generate_random_entry(previous_thought, guidance, context)
            elif technique == "reversal":
                content = self._generate_reversal(previous_thought, guidance, context)
            elif technique == "provocation":
                content = self._generate_provocation(previous_thought, guidance, context)
            elif technique == "alternative_perspective":
                content = self._generate_alternative_perspective(
                    previous_thought, guidance, context
                )
            else:
                content = self._generate_exploration(previous_thought, guidance, context)

            # Explorations can branch
            branch_id = f"exploration_{technique}_{self._step_counter}"
            return (
                content,
                ThoughtType.BRANCH,
                self._root_id or previous_thought.id,
                1,  # Branch from root level
                0.7,
                branch_id,
            )

        elif phase == "connection_making":
            content = self._generate_analogy(previous_thought, guidance, context)
            return (
                content,
                ThoughtType.HYPOTHESIS,
                previous_thought.id,
                previous_thought.depth + 1,
                0.75,
                previous_thought.branch_id,
            )

        elif phase == "synthesis":
            content = self._generate_synthesis(previous_thought, guidance, context)
            return (
                content,
                ThoughtType.SYNTHESIS,
                previous_thought.id,
                previous_thought.depth + 1,
                0.85,
                None,
            )

        else:  # continuation/refinement
            content = self._generate_continuation(previous_thought, guidance, context)
            return (
                content,
                ThoughtType.CONTINUATION,
                previous_thought.id,
                previous_thought.depth + 1,
                previous_thought.confidence,
                previous_thought.branch_id,
            )

    def _generate_framing(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the problem framing content.

        This is a helper method that would typically call an LLM or reasoning engine.
        For now, it returns a template that can be filled by the actual implementation.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the problem framing
        """
        return (
            f"Step {self._step_counter}: PROBLEM FRAMING - Setting the Creative Stage\n\n"
            f"Problem: {input_text}\n\n"
            f"INITIAL ANALYSIS:\n"
            f"Before we dive into lateral thinking, let's frame this problem clearly:\n"
            f"- What are we really trying to achieve?\n"
            f"- What constraints or limitations exist?\n"
            f"- What assumptions might we be making unconsciously?\n"
            f"- What is the conventional approach (that we'll challenge)?\n\n"
            f"This framing sets the stage for creative, divergent thinking. "
            f"We'll challenge every assumption and explore unconventional paths."
        )

    def _generate_assumption_challenge(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate assumption challenge content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for assumption challenge
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: CHALLENGING ASSUMPTIONS\n\n"
            f"Let's question the fundamental assumptions embedded in this problem:\n\n"
            f"ASSUMPTION CHALLENGES:\n"
            f"1. What if our definition of the problem itself is wrong?\n"
            f"2. What constraints are we accepting that might not be real?\n"
            f"3. What 'rules' are we following that could be broken?\n"
            f"4. What are we assuming about the context that might be false?\n"
            f"5. What if the 'obvious' solution is actually preventing us from seeing better options?\n\n"
            f"By challenging these assumptions, we open up new solution spaces "
            f"that conventional thinking would miss.{guidance_text}"
        )

    def _generate_random_entry(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate random entry technique content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for random entry exploration
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Generate a random concept to use as stimulus
        random_concepts = [
            "ocean waves",
            "bicycle gears",
            "tree roots",
            "jazz improvisation",
            "ant colonies",
            "puzzle pieces",
            "river deltas",
            "magnetic fields",
        ]
        random_concept = random_concepts[self._step_counter % len(random_concepts)]

        return (
            f"Step {self._step_counter}: RANDOM ENTRY - Spark from the Unexpected\n\n"
            f"RANDOM STIMULUS: {random_concept}\n\n"
            f"Now, let's use this completely unrelated concept to generate new ideas:\n\n"
            f"EXPLORATION:\n"
            f"- What characteristics does '{random_concept}' have?\n"
            f"- How might these characteristics apply to our problem?\n"
            f"- What unexpected connections can we draw?\n"
            f"- What new perspectives does this stimulus reveal?\n\n"
            f"Random entry breaks our mental patterns by introducing elements from "
            f"outside our problem domain, sparking novel connections.{guidance_text}"
        )

    def _generate_reversal(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate reversal technique content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for reversal exploration
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: REVERSAL - Flipping the Problem\n\n"
            f"What if we completely reversed the problem?\n\n"
            f"REVERSAL EXPLORATION:\n"
            f"1. Instead of solving the problem, how could we make it WORSE?\n"
            f"2. What if we did the exact opposite of the conventional approach?\n"
            f"3. What if we swapped the problem and solution?\n"
            f"4. What if we inverted all the constraints?\n"
            f"5. What would happen if we reversed the order, direction, or sequence?\n\n"
            f"Reversals help us see hidden aspects by examining the negative space "
            f"around our problem. Sometimes the path forward is revealed by looking backward.{guidance_text}"
        )

    def _generate_provocation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate provocation technique content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for provocation exploration
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: PROVOCATION - Breaking Mental Patterns\n\n"
            f"Let's use provocative 'what if' statements to escape conventional thinking:\n\n"
            f"PROVOCATIONS:\n"
            f"- What if we had unlimited resources?\n"
            f"- What if the problem solved itself?\n"
            f"- What if we had to solve this in 5 minutes instead of 5 months?\n"
            f"- What if we eliminated the most 'essential' component?\n"
            f"- What if we made it deliberately imperfect?\n"
            f"- What if we combined it with something completely incompatible?\n\n"
            f"These provocations are deliberately unrealistic to jolt us out of "
            f"habitual thinking patterns and reveal hidden possibilities.{guidance_text}"
        )

    def _generate_alternative_perspective(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate alternative perspective technique content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for alternative perspective exploration
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: ALTERNATIVE PERSPECTIVES - Seeing Through Different Eyes\n\n"
            f"How would different stakeholders or personas view this problem?\n\n"
            f"PERSPECTIVE SHIFTS:\n"
            f"1. A child's perspective: What would a 5-year-old suggest?\n"
            f"2. An artist's perspective: How would this be approached creatively?\n"
            f"3. A scientist's perspective: What would the data and experiments show?\n"
            f"4. A comedian's perspective: What's absurd or ironic about this?\n"
            f"5. A future historian's perspective: How will this look in 100 years?\n"
            f"6. Nature's perspective: How do natural systems solve similar problems?\n\n"
            f"Each perspective reveals blind spots and assumptions we carry "
            f"from our own limited viewpoint.{guidance_text}"
        )

    def _generate_analogy(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate analogy content for connection making.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for analogy connections
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: ANALOGIES - Drawing Unexpected Connections\n\n"
            f"Let's draw parallels from completely different domains:\n\n"
            f"ANALOGICAL THINKING:\n"
            f"- How is this problem like a biological system?\n"
            f"- What can we learn from how cities handle similar challenges?\n"
            f"- How do ecosystems solve resource allocation?\n"
            f"- What patterns from music, art, or games apply here?\n"
            f"- How do social networks address comparable issues?\n\n"
            f"CONNECTIONS EMERGING:\n"
            f"By mapping our problem onto these diverse domains, we discover "
            f"proven patterns and principles that can be adapted to our context. "
            f"The best solutions often come from unexpected analogies.{guidance_text}"
        )

    def _generate_exploration(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate general exploration content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for exploration
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: CREATIVE EXPLORATION\n\n"
            f"Continuing our lateral thinking journey with free exploration:\n\n"
            f"DIVERGENT IDEAS:\n"
            f"- What wild ideas emerge when we suspend judgment?\n"
            f"- What would a breakthrough solution look like?\n"
            f"- What combinations haven't been tried?\n"
            f"- What if we ignored all best practices?\n\n"
            f"This is a space for unrestricted creative thinking, where quantity "
            f"of ideas matters more than immediate quality.{guidance_text}"
        )

    def _generate_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate synthesis content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for synthesis
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        techniques = ", ".join(self._techniques_used)

        return (
            f"Step {self._step_counter}: SYNTHESIS - Harvesting Creative Insights\n\n"
            f"We've explored this problem using lateral thinking techniques: {techniques}\n\n"
            f"SYNTHESIZING INSIGHTS:\n"
            f"Now let's distill our divergent exploration into actionable solutions:\n\n"
            f"1. NOVEL APPROACHES: What unconventional methods emerged?\n"
            f"2. CHALLENGED ASSUMPTIONS: Which assumptions did we successfully break?\n"
            f"3. UNEXPECTED CONNECTIONS: What surprising insights arose?\n"
            f"4. CREATIVE SOLUTIONS: What innovative paths forward can we pursue?\n"
            f"5. NEXT EXPERIMENTS: What should we test or prototype?\n\n"
            f"This synthesis transforms creative chaos into structured innovation, "
            f"preserving the best ideas while making them practical.{guidance_text}"
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for continuation
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        previous_phase = previous_thought.metadata.get("phase", "unknown")

        return (
            f"Step {self._step_counter}: Continuing Lateral Thinking\n\n"
            f"Building on the {previous_phase} from step {previous_thought.step_number}, "
            f"let's further develop our creative reasoning. "
            f"This continuation explores additional facets, refines ideas, "
            f"or applies our lateral insights in new directions.{guidance_text}"
        )

    async def _sample_framing(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate problem framing content using LLM sampling.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the problem framing

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling but was not provided")

        system_prompt = """You are a lateral thinking expert using creative, divergent problem-solving techniques.

Your task is to frame the problem clearly before applying lateral thinking techniques.

Structure your response:
1. Restate the problem clearly
2. Identify what we're really trying to achieve
3. List constraints and limitations
4. Identify assumptions that might be unconsciously made
5. Note the conventional approach (that we'll challenge later)

Be clear and analytical. This framing sets the stage for creative exploration."""

        user_prompt = f"""Problem: {input_text}

Please frame this problem for lateral thinking exploration."""

        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        def fallback() -> str:
            return self._generate_framing(input_text, context)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        return (
            f"Step {self._step_counter}: PROBLEM FRAMING - Setting the Creative Stage\n\n{result}"
        )


# Export metadata and class
__all__ = [
    "LateralThinkingMethod",
    "LATERAL_THINKING_METADATA",
]
