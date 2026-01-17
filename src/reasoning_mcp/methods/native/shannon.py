"""Shannon Thinking reasoning method.

This module implements Claude Shannon's systematic problem-solving approach,
focusing on rigorous engineering methodology with 5 distinct phases:
1. Problem Definition - Clearly define the problem
2. Constraints - Identify constraints and limitations
3. Model - Build a mathematical/theoretical model
4. Proof - Validate through proofs or experimental validation
5. Implementation - Design practical solution

This method is inspired by Shannon's information theory work and his systematic
approach to solving complex technical problems.
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


# Metadata for Shannon Thinking method
SHANNON_THINKING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SHANNON_THINKING,
    name="Shannon Thinking",
    description="Information-theoretic reasoning with entropy and uncertainty analysis. "
    "Systematic engineering approach through Problem Definition, Constraints, "
    "Model, Proof, and Implementation phases.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset(
        {
            "shannon",
            "information-theory",
            "engineering",
            "systematic",
            "mathematical",
            "technical",
            "rigorous",
            "five-phase",
        }
    ),
    complexity=7,  # High complexity due to rigorous engineering approach
    supports_branching=True,  # Can branch during model/proof phases
    supports_revision=True,  # Can revise assumptions and models
    requires_context=False,  # Self-contained approach
    min_thoughts=5,  # One per phase minimum
    max_thoughts=0,  # No limit - can iterate within phases
    avg_tokens_per_thought=600,  # Detailed technical analysis
    best_for=(
        "technical problems",
        "engineering challenges",
        "system design",
        "mathematical modeling",
        "information theory problems",
        "optimization problems",
        "communication systems",
        "algorithmic design",
        "theoretical validation",
    ),
    not_recommended_for=(
        "purely creative tasks",
        "subjective decision-making",
        "emotional reasoning",
        "informal brainstorming",
        "simple procedural tasks",
    ),
)


# Shannon's 5 phases
class ShannonPhase:
    """Shannon Thinking phases enumeration."""

    PROBLEM_DEFINITION = "problem_definition"
    CONSTRAINTS = "constraints"
    MODEL = "model"
    PROOF = "proof"
    IMPLEMENTATION = "implementation"


class ShannonThinking(ReasoningMethodBase):
    """Shannon Thinking reasoning method implementation.

    This class implements Claude Shannon's systematic problem-solving methodology,
    characterized by rigorous engineering discipline and mathematical formalization.
    The method progresses through five distinct phases, each building on the previous.

    Key characteristics:
    - Five distinct phases with clear objectives
    - Mathematical/engineering rigor
    - Systematic constraint identification
    - Formal modeling and validation
    - Practical implementation focus
    - Supports iteration and revision within phases
    - Optional branching for exploring alternative models

    Phases:
        1. Problem Definition: Clearly articulate what needs to be solved
        2. Constraints: Identify all constraints, limitations, and boundary conditions
        3. Model: Build mathematical or theoretical model of the system
        4. Proof: Validate the model through formal proofs or experimental validation
        5. Implementation: Design practical solution based on validated model

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = ShannonThinking()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Design an efficient data compression algorithm"
        ... )
        >>> print(result.metadata["phase"])  # "problem_definition"

        Continue through phases:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Proceed to constraints analysis"
        ... )
        >>> print(next_thought.metadata["phase"])  # "constraints"
    """

    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Shannon Thinking method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase = ShannonPhase.PROBLEM_DEFINITION
        self._phase_history: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SHANNON_THINKING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SHANNON_THINKING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SHANNON_THINKING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HIGH_VALUE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Shannon Thinking method for execution,
        resetting all phase tracking and counters.

        Examples:
            >>> method = ShannonThinking()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._current_phase == ShannonPhase.PROBLEM_DEFINITION
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = ShannonPhase.PROBLEM_DEFINITION
        self._phase_history = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Shannon Thinking method.

        This method creates the initial thought, starting with the Problem Definition
        phase. It analyzes the input to clearly articulate what needs to be solved,
        following Shannon's systematic approach.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (may include domain info, constraints)
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the problem definition phase

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = ShannonThinking()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Optimize signal transmission over noisy channel"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION
            >>> assert thought.method_id == MethodIdentifier.SHANNON_THINKING
        """
        if not self._initialized:
            raise RuntimeError("Shannon Thinking method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = ShannonPhase.PROBLEM_DEFINITION
        self._phase_history = [ShannonPhase.PROBLEM_DEFINITION]

        # Create the initial thought - Problem Definition phase
        if self._use_sampling and execution_context and execution_context.can_sample:
            content = await self._sample_problem_definition(input_text, context)
        else:
            content = self._generate_problem_definition(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SHANNON_THINKING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial confidence - will increase with validation
            metadata={
                "input": input_text,
                "context": context or {},
                "phase": ShannonPhase.PROBLEM_DEFINITION,
                "phase_number": 1,
                "total_phases": 5,
                "reasoning_type": "shannon",
                "approach": "systematic engineering",
                "sampled": (
                    self._use_sampling
                    and execution_context is not None
                    and execution_context.can_sample
                ),
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SHANNON_THINKING

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

        This method progresses through Shannon's phases, or iterates within a phase
        if needed. The guidance parameter can specify phase transitions or refinements.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "next phase", "refine model", "branch")
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the Shannon reasoning process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = ShannonThinking()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Design error correction")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Proceed to constraints"
            ... )
            >>> assert second.metadata["phase"] == ShannonPhase.CONSTRAINTS
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError("Shannon Thinking method must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on current phase and guidance
        next_phase = self._determine_next_phase(
            previous_thought=previous_thought,
            guidance=guidance,
        )

        # Determine thought type
        thought_type = self._determine_thought_type(
            previous_thought=previous_thought,
            next_phase=next_phase,
            guidance=guidance,
        )

        # Generate content based on phase
        if self._use_sampling and execution_context and execution_context.can_sample:
            content = await self._sample_phase_content(
                phase=next_phase,
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )
        else:
            content = self._generate_phase_content(
                phase=next_phase,
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )

        # Update phase tracking
        if next_phase != self._current_phase:
            self._current_phase = next_phase
            self._phase_history.append(next_phase)

        # Calculate confidence based on phase and depth
        confidence = self._calculate_confidence(
            phase=next_phase,
            depth=previous_thought.depth + 1,
            thought_type=thought_type,
        )

        # Get phase number
        phase_number = self._get_phase_number(next_phase)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SHANNON_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "phase": next_phase,
                "phase_number": phase_number,
                "total_phases": 5,
                "previous_phase": previous_thought.metadata.get("phase"),
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "shannon",
                "phase_history": list(self._phase_history),
                "sampled": (
                    self._use_sampling
                    and execution_context is not None
                    and execution_context.can_sample
                ),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Shannon Thinking, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = ShannonThinking()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_problem_definition(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the problem definition phase content.

        This phase focuses on clearly articulating what needs to be solved,
        identifying key variables, and establishing the scope.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the problem definition phase
        """
        return (
            f"Phase 1: Problem Definition\n\n"
            f"Input Problem: {input_text}\n\n"
            f"Following Shannon's systematic approach, I will first clearly define "
            f"the problem space. This involves:\n\n"
            f"1. Articulating the core problem in precise terms\n"
            f"2. Identifying key variables and parameters\n"
            f"3. Establishing the scope and boundaries\n"
            f"4. Determining what constitutes a successful solution\n"
            f"5. Recognizing any uncertainty or entropy in the problem statement\n\n"
            f"Let me begin by formally stating the problem and its essential elements."
        )

    def _generate_phase_content(
        self,
        phase: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for a specific phase.

        Args:
            phase: The current Shannon phase
            previous_thought: The previous thought node
            guidance: Optional guidance text
            context: Optional context dictionary

        Returns:
            Generated content for the phase
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        if phase == ShannonPhase.PROBLEM_DEFINITION:
            return (
                f"Phase 1: Problem Definition (Refinement)\n\n"
                f"Building on step {previous_thought.step_number}, "
                f"let me further refine the problem definition.{guidance_text}"
            )

        elif phase == ShannonPhase.CONSTRAINTS:
            return (
                f"Phase 2: Constraints Analysis\n\n"
                f"Having defined the problem, I now systematically identify all "
                f"constraints and limitations:\n\n"
                f"1. Physical constraints (resources, capacity, bandwidth, etc.)\n"
                f"2. Theoretical constraints (mathematical limits, information bounds)\n"
                f"3. Practical constraints (implementation, cost, time)\n"
                f"4. Environmental constraints (noise, interference, uncertainty)\n"
                f"5. Boundary conditions and edge cases\n\n"
                f"Understanding these constraints is crucial for developing a "
                f"realistic and optimal solution.{guidance_text}"
            )

        elif phase == ShannonPhase.MODEL:
            return (
                f"Phase 3: Mathematical/Theoretical Model\n\n"
                f"With the problem defined and constraints identified, I will now "
                f"construct a formal model:\n\n"
                f"1. Mathematical representation of the system\n"
                f"2. Key equations and relationships\n"
                f"3. Information-theoretic formulation (entropy, capacity, etc.)\n"
                f"4. Abstraction of essential components\n"
                f"5. Predictive framework for system behavior\n\n"
                f"The model should capture the essence of the problem while "
                f"remaining tractable for analysis.{guidance_text}"
            )

        elif phase == ShannonPhase.PROOF:
            return (
                f"Phase 4: Proof and Validation\n\n"
                f"Now I will validate the model through rigorous analysis:\n\n"
                f"1. Formal mathematical proofs (where applicable)\n"
                f"2. Theoretical analysis of model properties\n"
                f"3. Verification against known cases\n"
                f"4. Experimental validation approach\n"
                f"5. Sensitivity analysis and robustness testing\n\n"
                f"This phase ensures the model is sound and the solution "
                f"will work as intended.{guidance_text}"
            )

        elif phase == ShannonPhase.IMPLEMENTATION:
            return (
                f"Phase 5: Implementation Design\n\n"
                f"With a validated model, I will now design the practical implementation:\n\n"
                f"1. Concrete algorithmic or architectural design\n"
                f"2. Optimization strategies\n"
                f"3. Practical considerations and trade-offs\n"
                f"4. Implementation steps and methodology\n"
                f"5. Performance expectations and metrics\n\n"
                f"This phase translates theoretical insights into "
                f"a working solution.{guidance_text}"
            )

        else:
            return (
                f"Continuing Shannon thinking process from step "
                f"{previous_thought.step_number}.{guidance_text}"
            )

    def _determine_next_phase(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> str:
        """Determine the next phase based on current state and guidance.

        Args:
            previous_thought: The previous thought node
            guidance: Optional guidance string

        Returns:
            The next phase to execute
        """
        current_phase = previous_thought.metadata.get("phase", ShannonPhase.PROBLEM_DEFINITION)

        # Check if guidance specifies a phase
        if guidance:
            guidance_lower = guidance.lower()
            if "problem" in guidance_lower or "definition" in guidance_lower:
                return ShannonPhase.PROBLEM_DEFINITION
            elif "constraint" in guidance_lower:
                return ShannonPhase.CONSTRAINTS
            elif "model" in guidance_lower:
                return ShannonPhase.MODEL
            elif "proof" in guidance_lower or "validat" in guidance_lower:
                return ShannonPhase.PROOF
            elif "implement" in guidance_lower:
                return ShannonPhase.IMPLEMENTATION
            elif "refine" in guidance_lower or "iterate" in guidance_lower:
                return str(current_phase)  # Stay in current phase

        # Default progression through phases
        phase_order = [
            ShannonPhase.PROBLEM_DEFINITION,
            ShannonPhase.CONSTRAINTS,
            ShannonPhase.MODEL,
            ShannonPhase.PROOF,
            ShannonPhase.IMPLEMENTATION,
        ]

        try:
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                return phase_order[current_idx + 1]
            else:
                return str(current_phase)  # Stay in final phase
        except ValueError:
            return ShannonPhase.PROBLEM_DEFINITION

    def _determine_thought_type(
        self,
        previous_thought: ThoughtNode,
        next_phase: str,
        guidance: str | None,
    ) -> ThoughtType:
        """Determine the appropriate thought type.

        Args:
            previous_thought: The previous thought node
            next_phase: The next phase to execute
            guidance: Optional guidance string

        Returns:
            The appropriate ThoughtType
        """
        current_phase = previous_thought.metadata.get("phase", ShannonPhase.PROBLEM_DEFINITION)

        # Check for explicit revision
        if guidance and ("revise" in guidance.lower() or "correct" in guidance.lower()):
            return ThoughtType.REVISION

        # Check for branching
        if guidance and "branch" in guidance.lower():
            return ThoughtType.BRANCH

        # Phase transition
        if next_phase != current_phase:
            # Special types for certain phases
            if next_phase == ShannonPhase.MODEL:
                return ThoughtType.HYPOTHESIS  # Models are hypotheses to be tested
            elif next_phase == ShannonPhase.PROOF:
                return ThoughtType.VERIFICATION  # Proof validates the model
            elif next_phase == ShannonPhase.IMPLEMENTATION:
                return ThoughtType.SYNTHESIS  # Implementation synthesizes everything
            else:
                return ThoughtType.CONTINUATION

        # Within same phase
        return ThoughtType.CONTINUATION

    def _calculate_confidence(
        self,
        phase: str,
        depth: int,
        thought_type: ThoughtType,
    ) -> float:
        """Calculate confidence score based on phase and progress.

        Confidence generally increases as we progress through phases and
        validate our models.

        Args:
            phase: Current Shannon phase
            depth: Current depth in reasoning tree
            thought_type: Type of thought

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence by phase (increases with validation)
        phase_confidence = {
            ShannonPhase.PROBLEM_DEFINITION: 0.6,
            ShannonPhase.CONSTRAINTS: 0.65,
            ShannonPhase.MODEL: 0.7,
            ShannonPhase.PROOF: 0.85,  # High confidence after validation
            ShannonPhase.IMPLEMENTATION: 0.8,  # Practical concerns may reduce slightly
        }

        base = phase_confidence.get(phase, 0.6)

        # Adjust for thought type
        if thought_type == ThoughtType.VERIFICATION:
            base += 0.05  # Validation increases confidence
        elif thought_type == ThoughtType.REVISION:
            base -= 0.1  # Revisions indicate uncertainty

        # Slight decrease with depth (complexity)
        depth_penalty = min(0.1, depth * 0.01)

        return max(0.4, min(0.95, base - depth_penalty))

    def _get_phase_number(self, phase: str) -> int:
        """Get the numerical order of a phase.

        Args:
            phase: Shannon phase name

        Returns:
            Phase number (1-5)
        """
        phase_numbers = {
            ShannonPhase.PROBLEM_DEFINITION: 1,
            ShannonPhase.CONSTRAINTS: 2,
            ShannonPhase.MODEL: 3,
            ShannonPhase.PROOF: 4,
            ShannonPhase.IMPLEMENTATION: 5,
        }
        return phase_numbers.get(phase, 1)

    async def _sample_problem_definition(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate problem definition using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual problem definition analysis rather than placeholder content.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled problem definition content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for sampling but was not provided")

        system_prompt = """You are a reasoning assistant using Shannon Thinking methodology.
You are in Phase 1: Problem Definition.

Following Claude Shannon's systematic engineering approach, clearly define the problem space.
Your response should:
1. Articulate the core problem in precise technical terms
2. Identify key variables, parameters, and components
3. Establish the scope and boundaries of the problem
4. Determine what constitutes a successful solution
5. Recognize any uncertainty, entropy, or information gaps in the problem statement

Use rigorous engineering language and mathematical precision where applicable."""

        user_prompt = f"""Problem: {input_text}

Context: {context or "No additional context provided."}

Apply Shannon's Problem Definition phase. Clearly and systematically define this problem,
identifying all essential elements, variables, and success criteria."""

        def fallback() -> str:
            return self._generate_problem_definition(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )
        return f"Phase 1: Problem Definition\n\n{content}"

    async def _sample_phase_content(
        self,
        phase: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate phase content using LLM sampling.

        Uses the execution context's sampling capability to generate
        phase-specific content rather than placeholder templates.

        Args:
            phase: The current Shannon phase
            previous_thought: The previous thought node
            guidance: Optional guidance text
            context: Optional context dictionary

        Returns:
            The sampled phase content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for sampling but was not provided")

        # Get phase-specific instructions
        phase_instructions = self._get_phase_instructions(phase)
        phase_number = self._get_phase_number(phase)

        system_prompt = f"""You are a reasoning assistant using Shannon Thinking methodology.
You are in Phase {phase_number}: {phase.replace("_", " ").title()}.

{phase_instructions}

Use rigorous engineering language and mathematical precision. Build upon previous work
while maintaining Shannon's systematic approach to problem-solving."""

        # Build context from previous thought
        previous_context = f"""Previous Phase: {previous_thought.metadata.get("phase", "unknown")}
Previous Analysis:
{previous_thought.content}"""

        user_prompt = f"""{previous_context}

Guidance: {guidance or "Continue to the next logical step in Shannon's methodology."}
Additional Context: {context or "None"}

Apply Shannon's {phase.replace("_", " ").title()} phase. Build systematically on the \
previous analysis."""

        def fallback() -> str:
            return self._generate_phase_content(phase, previous_thought, guidance, context)

        phase_title = phase.replace("_", " ").title()
        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1200,
        )
        return f"Phase {phase_number}: {phase_title}\n\n{content}"

    def _get_phase_instructions(self, phase: str) -> str:
        """Get detailed instructions for a specific Shannon phase.

        Args:
            phase: The Shannon phase

        Returns:
            Detailed instructions for that phase
        """
        instructions = {
            ShannonPhase.PROBLEM_DEFINITION: """Clearly define the problem space:
- Articulate the core problem in precise technical terms
- Identify key variables, parameters, and components
- Establish the scope and boundaries
- Determine success criteria
- Recognize uncertainty and information gaps""",
            ShannonPhase.CONSTRAINTS: """Systematically identify all constraints:
- Physical constraints (resources, capacity, bandwidth, etc.)
- Theoretical constraints (mathematical limits, information bounds)
- Practical constraints (implementation, cost, time)
- Environmental constraints (noise, interference, uncertainty)
- Boundary conditions and edge cases""",
            ShannonPhase.MODEL: """Construct a formal mathematical/theoretical model:
- Mathematical representation of the system
- Key equations and relationships
- Information-theoretic formulation (entropy, capacity, etc.)
- Abstraction of essential components
- Predictive framework for system behavior""",
            ShannonPhase.PROOF: """Validate the model through rigorous analysis:
- Formal mathematical proofs (where applicable)
- Theoretical analysis of model properties
- Verification against known cases
- Experimental validation approach
- Sensitivity analysis and robustness testing""",
            ShannonPhase.IMPLEMENTATION: """Design the practical implementation:
- Concrete algorithmic or architectural design
- Optimization strategies
- Practical considerations and trade-offs
- Implementation steps and methodology
- Performance expectations and metrics""",
        }
        return instructions.get(phase, "Continue Shannon's systematic analysis.")
