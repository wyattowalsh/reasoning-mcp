"""Step Back reasoning method.

This module implements the Step Back reasoning method: an abstraction-first approach
where we step back from the specific problem to identify abstract principles and
general cases, then apply those high-level insights to solve the specific problem.
This method is particularly effective for problems requiring conceptual understanding
and principle-based reasoning.
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

# Metadata for Step Back method
STEP_BACK_METADATA = MethodMetadata(
    identifier=MethodIdentifier.STEP_BACK,
    name="Step Back",
    description="Step back to consider higher-level concepts and abstract principles "
    "before solving specific problem details. Abstraction-first reasoning approach.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "abstraction",
            "principles",
            "conceptual",
            "general-to-specific",
            "high-level",
            "strategic",
        }
    ),
    complexity=4,  # Medium complexity - requires abstraction skills
    supports_branching=False,  # Linear progression through abstraction levels
    supports_revision=True,  # Can revise both abstract and specific thoughts
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: step back, principle, application
    max_thoughts=0,  # No limit - can iterate through multiple principles
    avg_tokens_per_thought=400,  # Moderate - abstractions need explanation
    best_for=(
        "conceptual problems",
        "principle-based reasoning",
        "problems requiring understanding of fundamentals",
        "generalizable solutions",
        "theory application",
        "strategic planning",
        "educational explanations",
        "pattern recognition across domains",
    ),
    not_recommended_for=(
        "simple procedural tasks",
        "problems requiring concrete details only",
        "time-critical immediate actions",
        "highly specific technical implementation",
        "rote memorization tasks",
    ),
)


class StepBack(ReasoningMethodBase):
    """Step Back reasoning method implementation.

    This class implements an abstraction-first reasoning pattern where we:
    1. Step back from the specific problem
    2. Identify abstract principles and general cases
    3. Understand the conceptual framework
    4. Apply high-level insights to the specific problem

    The method progresses through distinct phases:
    - ABSTRACTION: Step back to identify higher-level concepts
    - PRINCIPLES: Extract general principles and patterns
    - FRAMEWORK: Build conceptual understanding
    - APPLICATION: Apply abstractions to specific problem
    - REFINEMENT: Iterate and refine the solution

    Key characteristics:
    - Abstraction before specifics
    - Principle identification
    - General-to-specific reasoning
    - Conceptual grounding
    - Medium computational overhead

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = StepBack()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can I improve my team's productivity?"
        ... )
        >>> print(result.content)  # Abstraction step

        Continue reasoning:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Identify the underlying principles"
        ... )
        >>> print(next_thought.metadata["phase"])  # "PRINCIPLES"
    """

    # Reasoning phases
    PHASE_ABSTRACTION = "ABSTRACTION"
    PHASE_PRINCIPLES = "PRINCIPLES"
    PHASE_FRAMEWORK = "FRAMEWORK"
    PHASE_APPLICATION = "APPLICATION"
    PHASE_REFINEMENT = "REFINEMENT"

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Step Back method.

        Args:
            enable_elicitation: Whether to enable user interaction for guiding
                abstraction level selection (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase = self.PHASE_ABSTRACTION
        self._abstraction_levels: list[str] = []
        self._identified_principles: list[str] = []
        self._use_sampling = False
        self._execution_context: ExecutionContext | None = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.STEP_BACK

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return STEP_BACK_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return STEP_BACK_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Step Back method for execution.
        Initializes phase tracking and abstraction storage.

        Examples:
            >>> method = StepBack()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._current_phase == StepBack.PHASE_ABSTRACTION
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = self.PHASE_ABSTRACTION
        self._abstraction_levels = []
        self._identified_principles = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Step Back method.

        This method creates the first thought in a step-back reasoning chain.
        It begins by stepping back from the specific problem to identify
        higher-level concepts and abstract patterns.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the initial abstraction step

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = StepBack()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How do I optimize this database query?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.metadata["phase"] == StepBack.PHASE_ABSTRACTION
            >>> assert thought.method_id == MethodIdentifier.STEP_BACK
        """
        if not self._initialized:
            raise RuntimeError("Step Back method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = self.PHASE_ABSTRACTION
        self._abstraction_levels = []
        self._identified_principles = []

        # Elicit user guidance on abstraction level if enabled
        selected_abstraction = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
        ):
            try:
                options = [
                    {
                        "id": "conceptual",
                        "label": (
                            "Conceptual/Theoretical - "
                            "Step back to fundamental theories and concepts"
                        ),
                    },
                    {
                        "id": "methodological",
                        "label": "Methodological - Step back to examine approaches and methods",
                    },
                    {
                        "id": "domain",
                        "label": "Domain Principles - Step back to core principles in the domain",
                    },
                    {
                        "id": "system",
                        "label": "System Thinking - Step back to see the broader system context",
                    },
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                if self._execution_context.ctx is None:
                    raise RuntimeError("Execution context has no ctx attribute for elicitation")
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "Which abstraction level should we step back to?",
                    options,
                    config=config,
                )
                if selection:
                    selected_abstraction = selection.selected
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
                # Fall back to default behavior on elicitation error
            except RuntimeError:
                # Re-raise RuntimeError (our own validation)
                raise

        # Create the initial abstraction thought (use sampling if available)
        if self._use_sampling:
            content = await self._sample_abstraction_thought(
                input_text, context, selected_abstraction
            )
        else:
            content = self._generate_abstraction_thought(input_text, context, selected_abstraction)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.STEP_BACK,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Moderate initial confidence - abstraction needs validation
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "step_back",
                "phase": self._current_phase,
                "abstraction_level": 1,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.STEP_BACK

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

        This method generates the next step in the step-back reasoning process,
        progressing through phases: abstraction, principles, framework,
        application, and refinement.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the step-back reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = StepBack()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Analyze problem")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Identify core principles"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.metadata["phase"] == StepBack.PHASE_PRINCIPLES
        """
        if not self._initialized:
            raise RuntimeError("Step Back method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Update phase based on progress
        self._update_phase(previous_thought, guidance)

        # Generate content based on current phase (use sampling if available)
        if self._use_sampling:
            content = await self._sample_phase_content(
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )
        else:
            content = self._generate_phase_content(
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )

        # Determine thought type
        thought_type = self._determine_thought_type()

        # Calculate confidence - increases as we move from abstract to concrete
        confidence = self._calculate_confidence(previous_thought)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.STEP_BACK,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "previous_step": previous_thought.step_number,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "step_back",
                "phase": self._current_phase,
                "abstraction_level": self._get_abstraction_level(),
                "principles_identified": len(self._identified_principles),
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Step Back, this checks initialization status and phase coherence.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = StepBack()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        if not self._initialized:
            return False

        # Verify phase is valid
        valid_phases = {
            self.PHASE_ABSTRACTION,
            self.PHASE_PRINCIPLES,
            self.PHASE_FRAMEWORK,
            self.PHASE_APPLICATION,
            self.PHASE_REFINEMENT,
        }
        return self._current_phase in valid_phases

    def _generate_abstraction_thought(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        selected_abstraction: str | None = None,
    ) -> str:
        """Generate the initial abstraction thought content.

        This method creates the first step where we step back from the specific
        problem to consider higher-level concepts and patterns.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context
            selected_abstraction: User-selected abstraction level (if elicited)

        Returns:
            The content for the abstraction thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual abstraction. This is a placeholder providing structure.
        """
        abstraction_focus = ""
        if selected_abstraction:
            focus_map = {
                "conceptual": "fundamental theories and concepts",
                "methodological": "approaches and methods",
                "domain": "core principles in the domain",
                "system": "the broader system context",
            }
            abstraction_focus = (
                f"\n\nFocus: Examining {focus_map.get(selected_abstraction, 'high-level patterns')}"
            )

        return (
            f"Step {self._step_counter}: Stepping Back - Abstraction\n\n"
            f"Specific Problem: {input_text}{abstraction_focus}\n\n"
            f"Before diving into the specifics, let me step back and consider:\n\n"
            f"1. What is the general class of problems this belongs to?\n"
            f"2. What are the underlying concepts and principles at play?\n"
            f"3. What higher-level patterns or frameworks are relevant?\n\n"
            f"By understanding the abstract structure first, we can apply "
            f"general principles that lead to more robust and insightful solutions."
        )

    def _generate_phase_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content based on the current phase.

        This method generates appropriate content for each phase of the
        step-back reasoning process.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The content for the current phase thought
        """
        phase_generators = {
            self.PHASE_ABSTRACTION: self._generate_abstraction_content,
            self.PHASE_PRINCIPLES: self._generate_principles_content,
            self.PHASE_FRAMEWORK: self._generate_framework_content,
            self.PHASE_APPLICATION: self._generate_application_content,
            self.PHASE_REFINEMENT: self._generate_refinement_content,
        }

        generator = phase_generators.get(
            self._current_phase,
            self._generate_default_content,
        )

        return generator(previous_thought, guidance, context)

    def _generate_abstraction_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for abstraction phase."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Deepening Abstraction\n\n"
            f"Building on the previous abstraction, let me identify even more "
            f"fundamental concepts and patterns that underlie this problem.{guidance_text}"
        )

    def _generate_principles_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for principles identification phase."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Identifying Core Principles\n\n"
            f"Now that we've stepped back to see the bigger picture, "
            f"let me identify the fundamental principles and rules that apply:\n\n"
            f"These principles will guide our approach to the specific problem.{guidance_text}"
        )

    def _generate_framework_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for framework building phase."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Building Conceptual Framework\n\n"
            f"With the principles identified, let me construct a conceptual framework "
            f"that connects these high-level ideas and shows how they relate to "
            f"our problem space.{guidance_text}"
        )

    def _generate_application_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for application phase."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        original_input = previous_thought.metadata.get("input", "the problem")

        return (
            f"Step {self._step_counter}: Applying Abstractions to Specific Problem\n\n"
            f"Now I'll apply the high-level insights and principles we've identified "
            f"to solve the original specific problem:\n\n"
            f"Problem: {original_input}\n\n"
            f"By grounding our solution in these abstract principles, we ensure "
            f"it's not just a narrow fix but a robust, principled approach.{guidance_text}"
        )

    def _generate_refinement_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for refinement phase."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Refining the Solution\n\n"
            f"Let me refine and validate the solution by checking it against "
            f"our identified principles and ensuring it addresses both the "
            f"specific problem and adheres to the general principles.{guidance_text}"
        )

    def _generate_default_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate default continuation content."""
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Continuing Step-Back Reasoning\n\n"
            f"Continuing from step {previous_thought.step_number}, "
            f"let me proceed with the next phase of abstraction-based reasoning.{guidance_text}"
        )

    def _update_phase(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> None:
        """Update the current reasoning phase.

        Phases progress: ABSTRACTION -> PRINCIPLES -> FRAMEWORK -> APPLICATION -> REFINEMENT

        Args:
            previous_thought: The previous thought to base progression on
            guidance: Optional guidance that might indicate phase transition
        """
        # Check if guidance explicitly requests a phase
        if guidance:
            guidance_lower = guidance.lower()
            if "principle" in guidance_lower:
                self._current_phase = self.PHASE_PRINCIPLES
                return
            elif "framework" in guidance_lower or "structure" in guidance_lower:
                self._current_phase = self.PHASE_FRAMEWORK
                return
            elif "apply" in guidance_lower or "specific" in guidance_lower:
                self._current_phase = self.PHASE_APPLICATION
                return
            elif "refine" in guidance_lower or "improve" in guidance_lower:
                self._current_phase = self.PHASE_REFINEMENT
                return

        # Natural progression based on step count
        if self._step_counter <= 2:
            self._current_phase = self.PHASE_ABSTRACTION
        elif self._step_counter <= 4:
            self._current_phase = self.PHASE_PRINCIPLES
        elif self._step_counter <= 6:
            self._current_phase = self.PHASE_FRAMEWORK
        elif self._step_counter <= 8:
            self._current_phase = self.PHASE_APPLICATION
        else:
            self._current_phase = self.PHASE_REFINEMENT

    def _determine_thought_type(self) -> ThoughtType:
        """Determine the appropriate thought type for the current phase.

        Returns:
            The ThoughtType appropriate for the current phase
        """
        if self._current_phase == self.PHASE_PRINCIPLES:
            return ThoughtType.HYPOTHESIS
        elif self._current_phase == self.PHASE_FRAMEWORK:
            return ThoughtType.SYNTHESIS
        elif self._current_phase == self.PHASE_APPLICATION:
            return ThoughtType.CONTINUATION
        elif self._current_phase == self.PHASE_REFINEMENT:
            return ThoughtType.VERIFICATION
        else:
            return ThoughtType.CONTINUATION

    def _calculate_confidence(self, previous_thought: ThoughtNode) -> float:
        """Calculate confidence for the current thought.

        Confidence generally increases as we move from abstract to concrete,
        as we're applying validated principles to specific problems.

        Args:
            previous_thought: The previous thought

        Returns:
            Confidence level (0.0 to 1.0)
        """
        base_confidence = previous_thought.confidence

        # Adjust based on phase
        phase_adjustments = {
            self.PHASE_ABSTRACTION: -0.05,  # Abstract is less certain
            self.PHASE_PRINCIPLES: 0.05,  # Principles add confidence
            self.PHASE_FRAMEWORK: 0.05,  # Framework adds structure
            self.PHASE_APPLICATION: 0.10,  # Application is more concrete
            self.PHASE_REFINEMENT: 0.05,  # Refinement increases confidence
        }

        adjustment = phase_adjustments.get(self._current_phase, 0.0)
        new_confidence = base_confidence + adjustment

        # Keep within bounds
        return max(0.3, min(0.95, new_confidence))

    def _get_abstraction_level(self) -> int:
        """Get the current abstraction level.

        Returns:
            Current abstraction level (higher = more abstract)
        """
        level_map = {
            self.PHASE_ABSTRACTION: 5,  # Highest abstraction
            self.PHASE_PRINCIPLES: 4,
            self.PHASE_FRAMEWORK: 3,
            self.PHASE_APPLICATION: 2,
            self.PHASE_REFINEMENT: 1,  # Most concrete
        }

        return level_map.get(self._current_phase, 3)

    async def _sample_abstraction_thought(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        selected_abstraction: str | None = None,
    ) -> str:
        """Generate the initial abstraction thought using LLM sampling.

        Uses the execution context's sampling capability to generate
        the actual abstraction step rather than placeholder content.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context
            selected_abstraction: User-selected abstraction level (if elicited)

        Returns:
            The content for the abstraction thought
        """
        # Build focus guidance based on user's selection
        focus_guidance = ""
        if selected_abstraction:
            focus_map = {
                "conceptual": "fundamental theories and concepts that underlie this domain",
                "methodological": "different approaches and methods that could be applied",
                "domain": "core principles and laws within this specific domain",
                "system": "the broader system context and how components interact",
            }
            focus_text = focus_map.get(selected_abstraction, "high-level patterns and frameworks")
            focus_guidance = f"\n\nUser guidance: Focus particularly on {focus_text}."

        system_prompt = f"""You are a reasoning assistant using the Step Back methodology.
This method involves stepping back from specific problems to consider higher-level concepts,
abstract principles, and general patterns before solving the specific problem.

For the initial abstraction step:
1. Acknowledge the specific problem
2. Step back to identify the general class or category of problems this belongs to
3. Consider underlying concepts and principles at play
4. Identify relevant higher-level patterns or frameworks
5. Explain how this abstraction will help solve the specific problem{focus_guidance}

Be thoughtful and thorough in your abstraction. Focus on conceptual understanding."""

        user_prompt = f"""Specific Problem: {input_text}

Apply the Step Back method's initial abstraction phase. Step back from this specific problem
to consider higher-level concepts, underlying principles, and general patterns that are relevant.
What is the broader class of problems this belongs to, and what fundamental concepts apply?"""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_abstraction_thought(
                input_text, context, selected_abstraction
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

    async def _sample_phase_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate phase content using LLM sampling.

        Uses the execution context's sampling capability to generate
        content for the current phase rather than placeholder content.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The content for the current phase thought
        """
        # Get the original input if available
        original_input = previous_thought.metadata.get("input", "the problem")

        # Phase-specific system prompts
        phase_prompts = {
            self.PHASE_ABSTRACTION: """You are continuing the Step Back abstraction phase.
Build upon the previous abstraction by identifying even more fundamental concepts and patterns.""",
            self.PHASE_PRINCIPLES: """You are in the Step Back principles identification phase.
Now that you've stepped back to see the bigger picture, identify the fundamental principles,
rules, and laws that apply to this problem domain.""",
            self.PHASE_FRAMEWORK: """You are in the Step Back framework building phase.
With principles identified, construct a conceptual framework that connects these high-level
ideas and shows how they relate to the problem space.""",
            self.PHASE_APPLICATION: """You are in the Step Back application phase.
Apply the high-level insights, principles, and framework you've developed to solve the
original specific problem. Ground your solution in the abstract principles.""",
            self.PHASE_REFINEMENT: """You are in the Step Back refinement phase.
Refine and validate the solution by checking it against the identified principles and
ensuring it addresses both the specific problem and adheres to general principles.""",
        }

        system_prompt = phase_prompts.get(
            self._current_phase,
            "You are continuing the Step Back reasoning process.",
        )

        # Build user prompt based on phase
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Previous reasoning:
{previous_thought.content}

Current phase: {self._current_phase}
Original problem: {original_input}{guidance_text}

Continue the Step Back reasoning in the {self._current_phase} phase."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_phase_content(
                previous_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )
