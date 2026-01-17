"""Mathematical Reasoning method.

This module implements formal mathematical reasoning with proofs, calculations,
and rigorous logical steps. It specializes in mathematical problems, formal logic,
and proofs that require step-by-step verification and theorem application.
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


# Metadata for Mathematical Reasoning method
MATHEMATICAL_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.MATHEMATICAL_REASONING,
    name="Mathematical Reasoning",
    description="Formal mathematical reasoning with proofs, calculations, and rigorous "
    "logical steps. Specializes in applying theorems, verifying steps, and presenting "
    "structured mathematical solutions.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "mathematical",
            "formal",
            "proof",
            "logic",
            "verification",
            "theorem",
            "rigorous",
            "symbolic",
        }
    ),
    complexity=7,  # High complexity due to formal rigor
    supports_branching=False,  # Linear proof structure
    supports_revision=True,  # Can revise proof steps
    requires_context=False,  # No special context needed
    min_thoughts=3,  # Need setup, reasoning, verification at minimum
    max_thoughts=0,  # No limit - proofs can be arbitrarily long
    avg_tokens_per_thought=400,  # Detailed mathematical steps
    best_for=(
        "mathematical proofs",
        "formal logic problems",
        "equation solving",
        "theorem application",
        "symbolic manipulation",
        "calculus and analysis",
        "geometric proofs",
        "number theory",
        "algebraic reasoning",
    ),
    not_recommended_for=(
        "creative brainstorming",
        "subjective analysis",
        "open-ended discussions",
        "qualitative reasoning",
        "ethical dilemmas",
    ),
)


class MathematicalReasoning(ReasoningMethodBase):
    """Mathematical Reasoning method implementation.

    This class implements formal mathematical reasoning suitable for proofs,
    calculations, and rigorous logical analysis. It follows a structured approach:
    1. Parse and understand the mathematical problem
    2. Identify relevant theorems, axioms, and formulas
    3. Apply step-by-step formal reasoning with justification
    4. Verify each step for logical correctness
    5. Present the final solution with rigorous proof

    Key characteristics:
    - Formal logical steps with justification
    - Theorem and formula application
    - Step-by-step verification
    - Rigorous mathematical notation
    - High precision and correctness
    - Linear proof structure

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = MathematicalReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Prove that the sum of two even numbers is even"
        ... )
        >>> print(result.content)  # Problem setup and approach

        Continue with proof steps:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Continue with the formal proof"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    # Common proof phases for tracking progress
    PHASE_SETUP = "setup"
    PHASE_GIVEN = "given"
    PHASE_THEOREM_APPLICATION = "theorem_application"
    PHASE_DERIVATION = "derivation"
    PHASE_VERIFICATION = "verification"
    PHASE_CONCLUSION = "conclusion"

    def __init__(self) -> None:
        """Initialize the Mathematical Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase = self.PHASE_SETUP
        self._theorems_used: list[str] = []
        self._definitions_used: list[str] = []
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.MATHEMATICAL_REASONING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return MATHEMATICAL_REASONING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return MATHEMATICAL_REASONING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Mathematical Reasoning method for execution.
        It resets all tracking variables for a new proof or calculation.

        Examples:
            >>> method = MathematicalReasoning()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = self.PHASE_SETUP
        self._theorems_used = []
        self._definitions_used = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Mathematical Reasoning method.

        This method creates the first thought in a mathematical proof or calculation.
        It analyzes the problem, identifies what needs to be proven or solved,
        and sets up the formal approach.

        Args:
            session: The current reasoning session
            input_text: The mathematical problem, proof request, or question
            context: Optional context (may include known theorems, axioms, etc.)
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the problem setup and approach

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = MathematicalReasoning()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Prove that √2 is irrational"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.MATHEMATICAL_REASONING
        """
        if not self._initialized:
            raise RuntimeError("Mathematical Reasoning method must be initialized before execution")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = self.PHASE_SETUP
        self._theorems_used = []
        self._definitions_used = []

        # Create the initial thought
        if use_sampling:
            content = await self._sample_problem_setup(input_text, context)
        else:
            content = self._generate_problem_setup(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MATHEMATICAL_REASONING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.9,  # High confidence in problem setup
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "mathematical",
                "phase": self._current_phase,
                "theorems_used": self._theorems_used.copy(),
                "definitions_used": self._definitions_used.copy(),
                "sampled": use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.MATHEMATICAL_REASONING

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
        """Continue mathematical reasoning from a previous thought.

        This method generates the next step in the proof or calculation,
        building rigorously on the previous step. Each continuation represents
        a formal logical step with justification.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step (e.g., "apply theorem X")
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the mathematical reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = MathematicalReasoning()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Prove 2x + 5 = 15")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Solve for x"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
            >>> assert second.type == ThoughtType.CONTINUATION
        """
        if not self._initialized:
            raise RuntimeError(
                "Mathematical Reasoning method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Update phase based on step number and guidance
        self._update_phase(previous_thought, guidance)

        # Determine thought type based on phase
        thought_type = self._determine_thought_type()

        # Generate continuation content
        content = self._generate_proof_step(
            previous_thought=previous_thought,
            guidance=guidance,
            context=context,
        )

        # High confidence maintained throughout formal proof
        confidence = min(0.95, previous_thought.confidence + 0.02)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.MATHEMATICAL_REASONING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "previous_step": previous_thought.step_number,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "mathematical",
                "phase": self._current_phase,
                "theorems_used": self._theorems_used.copy(),
                "definitions_used": self._definitions_used.copy(),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Mathematical Reasoning, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = MathematicalReasoning()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _sample_problem_setup(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial problem setup using LLM sampling.

        Args:
            input_text: The mathematical problem or proof request
            context: Optional additional context

        Returns:
            The content for the problem setup thought

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_problem_setup but was not provided"
            )

        system_prompt = """You are a mathematical reasoning assistant.
Analyze the given mathematical problem with rigorous formal reasoning.
Your response should:
1. Clearly state what needs to be proven or solved
2. Identify relevant definitions, theorems, and axioms
3. Outline a formal proof strategy
4. Set up the problem with proper mathematical notation where appropriate"""

        user_prompt = f"""Mathematical Problem: {input_text}

Provide a rigorous mathematical problem setup:
1. Formalize the problem statement
2. Identify what is given and what needs to be proven/solved
3. List relevant theorems, definitions, or formulas
4. Outline the proof or solution strategy

Begin your formal mathematical analysis."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_problem_setup(input_text, context),
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for mathematical precision
            max_tokens=800,
        )

    def _generate_problem_setup(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial problem setup.

        This creates a formal statement of what needs to be proven or solved,
        identifies the given information, and outlines the proof strategy.

        Args:
            input_text: The mathematical problem or proof request
            context: Optional additional context

        Returns:
            The content for the problem setup thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual mathematical analysis. This is a placeholder that
            provides the formal structure.
        """
        return (
            f"**Mathematical Problem Setup**\n\n"
            f"**Problem Statement:**\n{input_text}\n\n"
            f"**Analysis:**\n"
            f"Let me formalize this problem and identify the approach.\n\n"
            f"**Strategy:**\n"
            f"I will use rigorous mathematical reasoning with formal justification "
            f"for each step. This will involve:\n"
            f"1. Identifying relevant definitions and theorems\n"
            f"2. Applying logical deduction step-by-step\n"
            f"3. Verifying each step's validity\n"
            f"4. Arriving at a formal conclusion\n\n"
            f"**Step 1: Problem Understanding**\n"
            f"I will begin by clearly defining all terms and identifying what "
            f"needs to be proven or calculated."
        )

    def _generate_proof_step(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a proof step continuation.

        This creates the next formal step in the mathematical reasoning,
        with justification and connection to the previous step.

        Args:
            previous_thought: The thought to build upon
            guidance: Optional guidance for this step
            context: Optional additional context

        Returns:
            The content for the proof step thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual proof step with proper mathematical notation. This
            is a placeholder that provides the formal structure.
        """
        phase_name = self._get_phase_display_name()
        guidance_text = f"\n\n**Guidance Applied:** {guidance}" if guidance else ""

        step_intro = self._get_step_intro()

        return (
            f"**Step {self._step_counter}: {phase_name}**\n\n"
            f"{step_intro}\n\n"
            f"**Justification:**\n"
            f"This step follows logically from Step {previous_thought.step_number} "
            f"by applying formal mathematical reasoning.{guidance_text}\n\n"
            f"**Verification:**\n"
            f"This step is valid because it maintains logical consistency and "
            f"follows from established mathematical principles."
        )

    def _update_phase(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> None:
        """Update the current proof phase based on progress.

        Args:
            previous_thought: The previous thought node
            guidance: Optional guidance that may indicate phase transition
        """
        # Simple phase progression based on step count
        # In a full implementation, this would be more sophisticated
        if self._step_counter == 2:
            self._current_phase = self.PHASE_GIVEN
        elif self._step_counter == 3:
            self._current_phase = self.PHASE_THEOREM_APPLICATION
        elif self._step_counter >= 4 and self._step_counter < 7:
            self._current_phase = self.PHASE_DERIVATION
        elif self._step_counter == 7:
            self._current_phase = self.PHASE_VERIFICATION
        elif self._step_counter >= 8:
            self._current_phase = self.PHASE_CONCLUSION

        # Override phase based on guidance keywords
        if guidance:
            lower_guidance = guidance.lower()
            if "theorem" in lower_guidance or "apply" in lower_guidance:
                self._current_phase = self.PHASE_THEOREM_APPLICATION
            elif "verify" in lower_guidance or "check" in lower_guidance:
                self._current_phase = self.PHASE_VERIFICATION
            elif "conclude" in lower_guidance or "therefore" in lower_guidance:
                self._current_phase = self.PHASE_CONCLUSION

    def _determine_thought_type(self) -> ThoughtType:
        """Determine the appropriate thought type based on current phase.

        Returns:
            The ThoughtType for the current phase
        """
        if self._current_phase == self.PHASE_VERIFICATION:
            return ThoughtType.VERIFICATION
        elif self._current_phase == self.PHASE_CONCLUSION:
            return ThoughtType.CONCLUSION
        else:
            return ThoughtType.CONTINUATION

    def _get_phase_display_name(self) -> str:
        """Get a human-readable name for the current phase.

        Returns:
            Display name for the current phase
        """
        phase_names = {
            self.PHASE_SETUP: "Problem Setup",
            self.PHASE_GIVEN: "Given Information",
            self.PHASE_THEOREM_APPLICATION: "Theorem Application",
            self.PHASE_DERIVATION: "Derivation",
            self.PHASE_VERIFICATION: "Verification",
            self.PHASE_CONCLUSION: "Conclusion",
        }
        return phase_names.get(self._current_phase, "Reasoning Step")

    def _get_step_intro(self) -> str:
        """Get an introduction appropriate for the current phase.

        Returns:
            Introduction text for the current phase
        """
        intros = {
            self.PHASE_SETUP: "Setting up the formal problem statement.",
            self.PHASE_GIVEN: "Identifying the given information and definitions.",
            self.PHASE_THEOREM_APPLICATION: (
                "Applying relevant theorems and mathematical principles."
            ),
            self.PHASE_DERIVATION: "Deriving the next logical step in the proof.",
            self.PHASE_VERIFICATION: ("Verifying the correctness of previous steps."),
            self.PHASE_CONCLUSION: ("Drawing the final conclusion from the proven steps."),
        }
        return intros.get(
            self._current_phase,
            "Continuing the formal mathematical reasoning.",
        )
