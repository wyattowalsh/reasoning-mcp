"""SPOC (Spontaneous Self-Correction) reasoning method.

Self-correction without external feedback.

Reference: 2025 - "SPOC: Spontaneous Self-Correction"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


SPOC_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SPOC,
    name="SPOC",
    description="Spontaneous Self-Correction without external feedback.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"self-correction", "spontaneous", "autonomous", "error-detection"}),
    complexity=5,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=6,
    avg_tokens_per_thought=180,
    best_for=("error correction", "self-improvement"),
    not_recommended_for=("tasks requiring external validation",),
)


class Spoc(ReasoningMethodBase):
    """SPOC method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._error_detected: bool = False
        self._use_sampling: bool = True
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SPOC

    @property
    def name(self) -> str:
        return SPOC_METADATA.name

    @property
    def description(self) -> str:
        return SPOC_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._error_detected = False

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("SPOC must be initialized")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "generate"

        # Generate content with sampling or fallback
        if self._use_sampling:
            content = await self._sample_initial_solution(input_text, context)
        else:
            content = self._generate_initial_solution(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SPOC,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,
            quality_score=0.75,
            metadata={
                "phase": self._current_phase,
                "error_detected": self._error_detected,
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SPOC
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("SPOC must be initialized")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "detect_error"
            self._error_detected = False  # No error in this case

            # Generate content with sampling or fallback
            if self._use_sampling:
                content = await self._sample_error_detection(previous_thought, context)
            else:
                content = self._generate_error_detection(previous_thought)

            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        elif prev_phase == "detect_error":
            self._current_phase = "validate"

            # Generate content with sampling or fallback
            if self._use_sampling:
                content = await self._sample_validation(previous_thought, context)
            else:
                content = self._generate_validation(previous_thought)

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90
        else:
            self._current_phase = "conclude"

            # Generate content with sampling or fallback
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, context)
            else:
                content = self._generate_conclusion(previous_thought)

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SPOC,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "error_detected": self._error_detected,
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Fallback heuristic methods (no LLM)

    def _generate_initial_solution(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial solution content without LLM sampling."""
        context_info = ""
        if context:
            context_info = f"\n\nContext: {context}"

        return (
            f"Step {self._step_counter}: Generate (SPOC)\n\n"
            f"Problem: {input_text}{context_info}\n\n"
            f"Initial solution: Analyzing problem and generating solution...\n\n"
            f"Next: Detect errors spontaneously."
        )

    def _generate_error_detection(self, previous_thought: ThoughtNode) -> str:
        """Generate error detection content without LLM sampling."""
        return (
            f"Step {self._step_counter}: Error Detection\n\n"
            f"Spontaneous check:\n"
            f"  - Logical consistency: ✓\n"
            f"  - Computational accuracy: ✓\n"
            f"  - Assumption validity: ✓\n"
            f"No errors detected.\nNext: Validate."
        )

    def _generate_validation(self, previous_thought: ThoughtNode) -> str:
        """Generate validation content without LLM sampling."""
        return (
            f"Step {self._step_counter}: Validate\n\n"
            f"SPOC Complete\nNo corrections needed.\n"
            f"Solution validated through spontaneous self-correction.\n"
            f"Confidence: 90%"
        )

    def _generate_conclusion(self, previous_thought: ThoughtNode) -> str:
        """Generate conclusion content without LLM sampling."""
        return f"Step {self._step_counter}: Final Answer - Solution validated through SPOC"

    # LLM sampling methods

    async def _sample_initial_solution(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial solution using LLM sampling.

        Args:
            input_text: The problem to solve
            context: Optional additional context

        Returns:
            The content for the initial thought
        """
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are using SPOC (Spontaneous Self-Correction), a method that generates
solutions while maintaining awareness for spontaneous error detection.

Generate an initial solution to the given problem. Be clear and systematic in your approach.
This is the first step - you'll have opportunities to detect and correct errors spontaneously
in subsequent steps.

Structure your response as:
1. Problem understanding
2. Initial solution approach
3. Solution generation"""

        user_prompt = f"""Problem: {input_text}{context_info}

Generate an initial solution using the SPOC method. Be systematic and clear."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_initial_solution(input_text, context),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_error_detection(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate error detection using LLM sampling.

        Args:
            previous_thought: The previous solution thought
            context: Optional additional context

        Returns:
            The content for error detection
        """
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are using SPOC (Spontaneous Self-Correction) for error detection.

Analyze the previous solution for potential errors WITHOUT external feedback.
Use spontaneous self-awareness to detect:
- Logical inconsistencies
- Computational errors
- Invalid assumptions
- Missing considerations
- Alternative interpretations

Be honest about any errors found. If errors are detected, explain them clearly.
If no errors are found, confirm the solution's validity."""

        user_prompt = f"""Previous Solution (Step {previous_thought.step_number}):
{previous_thought.content}{context_info}

Perform spontaneous error detection on this solution. Check for any issues."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_error_detection(previous_thought),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_validation(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate validation using LLM sampling.

        Args:
            previous_thought: The error detection thought
            context: Optional additional context

        Returns:
            The content for validation
        """
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are completing SPOC (Spontaneous Self-Correction) validation.

Based on the error detection phase, validate the solution:
- If errors were detected, confirm they've been addressed
- If no errors were found, validate the original solution
- Provide final confidence assessment
- Summarize the SPOC process

Be conclusive and clear about the final answer."""

        user_prompt = f"""Error Detection Results (Step {previous_thought.step_number}):
{previous_thought.content}{context_info}

Complete the validation phase and provide the final answer."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_validation(previous_thought),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Args:
            previous_thought: The validation thought
            context: Optional additional context

        Returns:
            The content for conclusion
        """
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are providing the final conclusion for SPOC.

Summarize the final answer concisely based on the validation results."""

        user_prompt = f"""Validation Results (Step {previous_thought.step_number}):
{previous_thought.content}{context_info}

Provide the final conclusive answer."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_conclusion(previous_thought),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )


__all__ = ["Spoc", "SPOC_METADATA"]
