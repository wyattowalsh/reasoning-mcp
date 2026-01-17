"""SETS (Self-Verification and Self-Correction) reasoning method.

This module implements a combined self-verification and targeted self-correction
reasoning method based on Chen et al. (Jan 2025). The method integrates systematic
step-by-step verification with error detection and correction capabilities.

The key innovation is combining:
- Step-by-step verification of reasoning chains
- Specific error type detection (logical, arithmetic, factual)
- Targeted corrections for identified errors
- Validation of corrections

Based on: Chen et al. (Jan 2025) - SETS: Self-Verification and Self-Correction
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


# Metadata for SETS method
SETS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SETS,
    name="SETS",
    description="Combined self-verification and self-correction that generates initial "
    "reasoning, verifies each step, detects specific error types (logical, arithmetic, "
    "factual), applies targeted corrections, validates fixes, and produces verified conclusion.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "verification",
            "error-correction",
            "self-checking",
            "accuracy",
            "step-by-step",
            "error-detection",
            "quality-assurance",
            "systematic",
        }
    ),
    complexity=6,  # Medium-high complexity - multi-phase verification and correction
    supports_branching=False,  # Linear verification and correction path
    supports_revision=True,  # Core feature - correcting through verification
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: generate + verify + detect + correct + validate
    max_thoughts=20,  # Multiple verification and correction cycles possible
    avg_tokens_per_thought=400,  # Moderate to high - includes detailed analysis
    best_for=(
        "complex reasoning tasks",
        "mathematical problem solving",
        "logical reasoning",
        "accuracy-critical tasks",
        "multi-step problem solving",
        "error-prone reasoning",
        "verification-sensitive outputs",
    ),
    not_recommended_for=(
        "simple factual queries",
        "creative writing tasks",
        "subjective opinion generation",
        "time-critical decisions",
        "tasks without verifiable steps",
    ),
)

logger = structlog.get_logger(__name__)


class Sets(ReasoningMethodBase):
    """SETS (Self-Verification and Self-Correction) reasoning method implementation.

    This class implements a combined verification and correction pattern where the system
    generates initial reasoning, then systematically verifies and corrects through:
    1. Generating initial reasoning steps
    2. Verifying each step systematically
    3. Detecting specific error types (logical, arithmetic, factual)
    4. Applying targeted corrections to errors
    5. Validating corrections
    6. Producing final verified conclusion

    The method reduces errors through explicit verification and correction cycles.

    Key characteristics:
    - Multi-phase verification process
    - Specific error type detection
    - Targeted error correction
    - Step-by-step verification
    - Quality improvement through correction
    - Medium-high complexity (6)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = Sets()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Calculate the compound interest on $1000 at 5% for 3 years"
        ... )
        >>> print(result.content)  # Initial reasoning

        Continue with verification:
        >>> verification = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Verify steps"
        ... )
        >>> print(verification.type)  # ThoughtType.VERIFICATION

        Continue with error detection:
        >>> detection = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=verification,
        ...     guidance="Detect errors"
        ... )
        >>> print(detection.type)  # ThoughtType.VERIFICATION

        Apply corrections:
        >>> correction = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=detection,
        ...     guidance="Apply corrections"
        ... )
        >>> print(correction.type)  # ThoughtType.REVISION

        Validate and conclude:
        >>> validated = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=correction,
        ...     guidance="Validate"
        ... )
        >>> print(validated.type)  # ThoughtType.CONCLUSION
    """

    # Error types that SETS can detect
    ERROR_TYPES = ["logical", "arithmetic", "factual", "procedural"]

    def __init__(self) -> None:
        """Initialize the SETS method."""
        self._initialized = False
        self._step_counter = 0
        # Phases: generate, verify_steps, correct_errors, validate, conclude
        self._current_phase: str = "generate"
        self._detected_errors: list[dict[str, Any]] = []
        self._corrections_applied: int = 0
        self._verification_stats: dict[str, int] = {
            "steps_verified": 0,
            "errors_detected": 0,
            "corrections_applied": 0,
            "validations_passed": 0,
        }
        self._use_sampling: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SETS

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SETS_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SETS_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the SETS method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = Sets()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "generate"
            >>> assert len(method._detected_errors) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._detected_errors = []
        self._corrections_applied = 0
        self._verification_stats = {
            "steps_verified": 0,
            "errors_detected": 0,
            "corrections_applied": 0,
            "validations_passed": 0,
        }

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the SETS method.

        This method creates the initial reasoning that will be verified
        and corrected through subsequent phases. It generates a first attempt
        at solving the problem with step-by-step reasoning.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the initial reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Sets()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Solve: If x + 5 = 12, what is x?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SETS
            >>> assert "verification_stats" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("SETS method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "generate"
        self._detected_errors = []
        self._corrections_applied = 0
        self._verification_stats = {
            "steps_verified": 0,
            "errors_detected": 0,
            "corrections_applied": 0,
            "validations_passed": 0,
        }

        # Generate initial reasoning (use sampling if available)
        if self._use_sampling:
            content = await self._sample_initial_reasoning(input_text, context)
        else:
            content = self._generate_initial_reasoning(input_text, context)

        # Initial confidence - moderate (will improve through verification/correction)
        initial_confidence = 0.5

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SETS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=initial_confidence,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "sets",
                "phase": self._current_phase,
                "detected_errors": [],
                "verification_stats": dict(self._verification_stats),
                "error_types_checked": self.ERROR_TYPES,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SETS

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

        This method implements the SETS cycle logic:
        - If previous was generate: verify steps
        - If previous was verify_steps: detect errors (or conclude if none)
        - If previous was correct_errors: validate corrections
        - If previous was validate: conclude
        - Follows the five-phase SETS structure

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the SETS process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Sets()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Calculate 15% of 80")
            >>> verification = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert verification.type == ThoughtType.VERIFICATION
            >>> assert verification.metadata["phase"] == "verify_steps"
        """
        if not self._initialized:
            raise RuntimeError("SETS method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "generate")

        # Get verification data from previous thought
        detected_errors = previous_thought.metadata.get("detected_errors", [])
        verification_stats = previous_thought.metadata.get(
            "verification_stats", dict(self._verification_stats)
        )

        if prev_phase == "generate":
            # Next: verify steps
            self._current_phase = "verify_steps"
            thought_type = ThoughtType.VERIFICATION
            if self._use_sampling:
                content, num_steps_verified = await self._sample_verify_reasoning_steps(
                    previous_thought, guidance, context
                )
            else:
                content, num_steps_verified = self._verify_reasoning_steps(
                    previous_thought, guidance, context
                )
            verification_stats["steps_verified"] = num_steps_verified
            confidence = 0.6
            quality_score = 0.65

        elif prev_phase == "verify_steps":
            # Next: detect errors (or go directly to conclude if none found)
            if self._use_sampling:
                content, new_errors = await self._sample_detect_errors(
                    previous_thought, guidance, context
                )
            else:
                content, new_errors = self._detect_errors(previous_thought, guidance, context)
            detected_errors = new_errors
            verification_stats["errors_detected"] = len(new_errors)

            if len(new_errors) > 0:
                # Errors found, need to correct
                self._current_phase = "correct_errors"
                thought_type = ThoughtType.VERIFICATION
                confidence = 0.5  # Lower confidence when errors detected
                quality_score = 0.55
            else:
                # No errors, can conclude
                self._current_phase = "conclude"
                thought_type = ThoughtType.CONCLUSION
                confidence = 0.9  # High confidence - verified with no errors
                quality_score = 0.95

        elif prev_phase == "correct_errors":
            # Next: apply corrections and validate
            self._current_phase = "validate"
            thought_type = ThoughtType.REVISION
            if self._use_sampling:
                content, corrections_count = await self._sample_apply_corrections(
                    previous_thought, detected_errors, guidance, context
                )
            else:
                content, corrections_count = self._apply_corrections(
                    previous_thought, detected_errors, guidance, context
                )
            verification_stats["corrections_applied"] = corrections_count
            confidence = 0.75
            quality_score = 0.8

        elif prev_phase == "validate":
            # Next: conclude with final verified result
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_generate_conclusion(
                    previous_thought, verification_stats, guidance, context
                )
            else:
                content = self._generate_conclusion(
                    previous_thought, verification_stats, guidance, context
                )
            verification_stats["validations_passed"] = 1
            confidence = 0.9  # High confidence after validation
            quality_score = 0.95

        else:
            # Fallback to conclusion (already done)
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_generate_conclusion(
                    previous_thought, verification_stats, guidance, context
                )
            else:
                content = self._generate_conclusion(
                    previous_thought, verification_stats, guidance, context
                )
            confidence = 0.9
            quality_score = 0.95

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SETS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "detected_errors": detected_errors,
                "verification_stats": verification_stats,
                "error_types_checked": self.ERROR_TYPES,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "sets",
                "previous_phase": prev_phase,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For SETS, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Sets()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_reasoning(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial step-by-step reasoning.

        This is a helper method that would typically call an LLM to generate
        the initial reasoning steps for the problem.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial reasoning

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reasoning steps. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: Initial Reasoning (Phase 1/5: Generate)\n\n"
            f"Problem: {input_text}\n\n"
            f"I will solve this problem step-by-step, generating a reasoning chain "
            f"that will then be verified and corrected through SETS.\n\n"
            f"Initial reasoning steps:\n"
            f"1. [LLM would generate step 1 of reasoning]\n"
            f"2. [LLM would generate step 2 of reasoning]\n"
            f"3. [LLM would generate step 3 of reasoning]\n"
            f"...\n\n"
            f"Initial answer: [LLM would provide initial answer]\n\n"
            f"Note: These steps will now be systematically verified for errors."
        )

    def _verify_reasoning_steps(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Verify each step of the reasoning chain.

        This is a helper method that would typically call an LLM to verify
        each step of the reasoning for correctness.

        Args:
            previous_thought: The reasoning to verify
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            A tuple of (verification content, number of steps verified)

        Note:
            In a full implementation, this would use an LLM to perform
            actual step-by-step verification. This is a placeholder.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # In real implementation, would extract and verify actual steps
        num_steps = 5  # Placeholder

        content = (
            f"Step {self._step_counter}: Step-by-Step Verification (Phase 2/5: Verify Steps)\n\n"
            f"Verifying reasoning from Step {previous_thought.step_number}...\n\n"
            f"Verification results:\n"
            f"- Step 1: [Verification result] ✓/✗\n"
            f"- Step 2: [Verification result] ✓/✗\n"
            f"- Step 3: [Verification result] ✓/✗\n"
            f"- Step 4: [Verification result] ✓/✗\n"
            f"- Step 5: [Verification result] ✓/✗\n\n"
            f"Total steps verified: {num_steps}\n"
            f"Next phase will detect specific error types if any issues found.{guidance_text}"
        )

        return content, num_steps

    def _detect_errors(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Detect specific error types in the verified reasoning.

        This is a helper method that would typically call an LLM to identify
        specific types of errors (logical, arithmetic, factual, procedural).

        Args:
            previous_thought: The verified reasoning to check
            guidance: Optional guidance for error detection
            context: Optional additional context

        Returns:
            A tuple of (detection content, list of detected errors)

        Note:
            In a full implementation, this would use an LLM to detect
            actual errors. This is a placeholder.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Example errors (in real implementation, would be LLM-detected)
        # For demonstration, sometimes find errors, sometimes don't
        import random

        has_errors = random.random() > 0.5

        if has_errors:
            detected_errors = [
                {
                    "step": 2,
                    "type": "arithmetic",
                    "description": "Calculation error in step 2",
                    "severity": "high",
                },
                {
                    "step": 4,
                    "type": "logical",
                    "description": "Logical inconsistency in step 4",
                    "severity": "medium",
                },
            ]

            content = (
                f"Step {self._step_counter}: Error Detection (Phase 3/5: Detect Errors)\n\n"
                f"Analyzing verification results from Step {previous_thought.step_number} "
                f"to detect specific error types...\n\n"
                f"Error types checked: {', '.join(self.ERROR_TYPES)}\n\n"
                f"Detected errors:\n"
            )

            for i, error in enumerate(detected_errors, 1):
                error_type = str(error.get("type", "unknown"))
                content += (
                    f"{i}. Step {error.get('step', 0)}: {error_type.upper()} error\n"
                    f"   Description: {error.get('description', 'N/A')}\n"
                    f"   Severity: {error.get('severity', 'unknown')}\n\n"
                )

            content += (
                f"Total errors detected: {len(detected_errors)}\n"
                f"Next phase will apply targeted corrections.{guidance_text}"
            )
        else:
            detected_errors = []

            content = (
                f"Step {self._step_counter}: Error Detection (Phase 3/5: Detect Errors)\n\n"
                f"Analyzing verification results from Step {previous_thought.step_number} "
                f"to detect specific error types...\n\n"
                f"Error types checked: {', '.join(self.ERROR_TYPES)}\n\n"
                f"No errors detected! ✓\n\n"
                f"All reasoning steps are verified as correct.\n"
                f"Proceeding directly to conclusion.{guidance_text}"
            )

        return content, detected_errors

    def _apply_corrections(
        self,
        previous_thought: ThoughtNode,
        detected_errors: list[dict[str, Any]],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Apply targeted corrections to detected errors.

        This is a helper method that would typically call an LLM to fix
        each detected error with targeted corrections.

        Args:
            previous_thought: The thought containing detected errors
            detected_errors: List of errors to correct
            guidance: Optional guidance for corrections
            context: Optional additional context

        Returns:
            A tuple of (correction content, number of corrections applied)

        Note:
            In a full implementation, this would use an LLM to generate
            actual corrections. This is a placeholder.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        corrections_count = len(detected_errors)

        content = (
            f"Step {self._step_counter}: Apply Corrections (Phase 4/5: Correct Errors)\n\n"
            f"Applying targeted corrections based on errors detected in "
            f"Step {previous_thought.step_number}...\n\n"
            f"Corrections applied:\n"
        )

        for i, error in enumerate(detected_errors, 1):
            error_type = str(error.get("type", "unknown"))
            content += (
                f"{i}. Step {error.get('step', 0)} - {error_type.upper()} error:\n"
                f"   Issue: {error.get('description', 'N/A')}\n"
                f"   Correction: [LLM would provide targeted correction]\n"
                f"   Status: ✓ Corrected\n\n"
            )

        content += (
            f"Total corrections applied: {corrections_count}\n"
            f"Next phase will validate these corrections.{guidance_text}"
        )

        return content, corrections_count

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        verification_stats: dict[str, int],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final verified conclusion.

        This is a helper method that would typically call an LLM to synthesize
        the final answer after verification and correction.

        Args:
            previous_thought: The validated reasoning
            verification_stats: Statistics from verification process
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the final conclusion

        Note:
            In a full implementation, this would use an LLM to generate
            the actual conclusion. This is a placeholder.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verified Conclusion (Phase 5/5: Conclude)\n\n"
            f"Based on the SETS verification and correction process, "
            f"I can now provide a verified conclusion.\n\n"
            f"Verification Summary:\n"
            f"- Steps verified: {verification_stats.get('steps_verified', 0)}\n"
            f"- Errors detected: {verification_stats.get('errors_detected', 0)}\n"
            f"- Corrections applied: {verification_stats.get('corrections_applied', 0)}\n"
            f"- Validations passed: {verification_stats.get('validations_passed', 0)}\n\n"
            f"Final verified answer:\n"
            f"[LLM would provide final answer after verification and correction]\n\n"
            f"Confidence: HIGH - This answer has been systematically verified "
            f"and corrected through the SETS process.{guidance_text}"
        )

        return content

    async def _sample_initial_reasoning(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial reasoning using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual reasoning steps rather than placeholder content.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial reasoning
        """
        system_prompt = """You are a reasoning assistant using SETS methodology.
Generate step-by-step reasoning that will be systematically verified and corrected.

Structure your initial reasoning:
1. Break down the problem into clear steps
2. Show your work explicitly for each step
3. Number each reasoning step
4. Include calculations, logical deductions, or factual claims
5. Provide an initial answer

Be thorough but know that errors will be caught in verification."""

        user_prompt = f"""Problem: {input_text}

Generate initial step-by-step reasoning for this problem. Show all your work.
This is Phase 1/5 (Generate) of the SETS process."""

        def fallback() -> str:
            return self._generate_initial_reasoning(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )
        phase_header = f"Step {self._step_counter}: Initial Reasoning (Phase 1/5: Generate)"
        return f"{phase_header}\n\n{content}"

    async def _sample_verify_reasoning_steps(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Verify reasoning steps using LLM sampling.

        Uses the execution context's sampling capability to perform
        actual step-by-step verification.

        Args:
            previous_thought: The reasoning to verify
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            A tuple of (verification content, number of steps verified)
        """
        system_prompt = """You are a reasoning verifier using SETS methodology.
Systematically verify each step of the reasoning for correctness.

For each step, check:
1. Logical consistency
2. Mathematical accuracy
3. Factual correctness
4. Procedural validity

Mark each step as ✓ (correct) or ✗ (needs review)."""

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Previous reasoning:
{previous_thought.content}

Verify each step systematically. This is Phase 2/5 (Verify Steps) of SETS.{guidance_text}"""

        def fallback() -> str:
            content, _ = self._verify_reasoning_steps(previous_thought, guidance, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for verification
            max_tokens=1200,
        )

        # Estimate number of steps verified (count checkmarks and X marks)
        num_steps = content.count("✓") + content.count("✗")
        if num_steps == 0:
            num_steps = 5  # Default estimate

        phase_header = (
            f"Step {self._step_counter}: Step-by-Step Verification (Phase 2/5: Verify Steps)"
        )
        full_content = f"{phase_header}\n\n{content}"
        return full_content, num_steps

    async def _sample_detect_errors(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Detect errors using LLM sampling.

        Uses the execution context's sampling capability to identify
        specific types of errors.

        Args:
            previous_thought: The verified reasoning to check
            guidance: Optional guidance for error detection
            context: Optional additional context

        Returns:
            A tuple of (detection content, list of detected errors)
        """
        system_prompt = f"""You are an error detector using SETS methodology.
Analyze the verification results and identify specific error types:
- Logical errors: Flawed reasoning or invalid inferences
- Arithmetic errors: Mathematical calculation mistakes
- Factual errors: Incorrect facts or information
- Procedural errors: Wrong steps or order of operations

For each error found, specify:
1. Step number
2. Error type ({", ".join(self.ERROR_TYPES)})
3. Description of the error
4. Severity (low, medium, high)

If no errors are found, explicitly state "No errors detected!" """

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Verification results:
{previous_thought.content}

Detect and categorize any errors. This is Phase 3/5 (Detect Errors) of SETS.{guidance_text}"""

        def fallback() -> str:
            content, _ = self._detect_errors(previous_thought, guidance, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for accuracy
            max_tokens=1000,
        )

        # Parse errors from the response
        detected_errors: list[dict[str, Any]] = []
        if "no errors detected" not in content.lower():
            # Simple heuristic: look for numbered errors or specific error types mentioned
            for error_type in self.ERROR_TYPES:
                if error_type.lower() in content.lower():
                    # Extract error information (simplified)
                    detected_errors.append(
                        {
                            "type": error_type,
                            "description": f"Detected {error_type} error",
                            "severity": "medium",
                            "step": len(detected_errors) + 1,
                        }
                    )

        phase_header = f"Step {self._step_counter}: Error Detection (Phase 3/5: Detect Errors)"
        full_content = f"{phase_header}\n\n{content}"
        return full_content, detected_errors

    async def _sample_apply_corrections(
        self,
        previous_thought: ThoughtNode,
        detected_errors: list[dict[str, Any]],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Apply corrections using LLM sampling.

        Uses the execution context's sampling capability to fix
        detected errors with targeted corrections.

        Args:
            previous_thought: The thought containing detected errors
            detected_errors: List of errors to correct
            guidance: Optional guidance for corrections
            context: Optional additional context

        Returns:
            A tuple of (correction content, number of corrections applied)
        """
        system_prompt = """You are a correction specialist using SETS methodology.
Apply targeted corrections to each detected error.

For each error:
1. Explain the issue clearly
2. Provide the corrected version
3. Show why the correction is valid
4. Mark as corrected with ✓

Be precise and thorough in your corrections."""

        errors_text = "\n".join(
            [
                (
                    f"- Step {err.get('step', 'N/A')}: {err.get('type', 'unknown')} error - "
                    f"{err.get('description', 'N/A')}"
                )
                for err in detected_errors
            ]
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Detected errors:
{errors_text}

Previous reasoning context:
{previous_thought.content[:500]}...

Apply targeted corrections to fix these errors.
This is Phase 4/5 (Correct Errors) of SETS.{guidance_text}"""

        def fallback() -> str:
            content, _ = self._apply_corrections(previous_thought, detected_errors, guidance, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=1500,
        )
        corrections_count = len(detected_errors)

        phase_header = (
            f"Step {self._step_counter}: Apply Corrections (Phase 4/5: Correct Errors)"
        )
        full_content = f"{phase_header}\n\n{content}"
        return full_content, corrections_count

    async def _sample_generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        verification_stats: dict[str, int],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Uses the execution context's sampling capability to synthesize
        the final verified answer.

        Args:
            previous_thought: The validated reasoning
            verification_stats: Statistics from verification process
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the final conclusion
        """
        system_prompt = """You are a conclusion synthesizer using SETS methodology.
Generate a final verified answer based on the complete SETS process.

Your conclusion should:
1. State the final answer clearly
2. Reference key steps from the verified reasoning
3. Note that the answer has been verified and corrected
4. Express high confidence in the result
5. Be concise but complete"""

        stats_text = f"""- Steps verified: {verification_stats.get("steps_verified", 0)}
- Errors detected: {verification_stats.get("errors_detected", 0)}
- Corrections applied: {verification_stats.get("corrections_applied", 0)}
- Validations passed: {verification_stats.get("validations_passed", 0)}"""

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Verification process summary:
{stats_text}

Previous reasoning:
{previous_thought.content[:500]}...

Generate the final verified conclusion. This is Phase 5/5 (Conclude) of SETS.{guidance_text}"""

        def fallback() -> str:
            return self._generate_conclusion(
                previous_thought, verification_stats, guidance, context
            )

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1000,
        )

        phase_header = f"Step {self._step_counter}: Verified Conclusion (Phase 5/5: Conclude)"
        full_content = f"{phase_header}\n\n{content}"
        return full_content


# Export public interface
__all__ = ["Sets", "SETS_METADATA"]
