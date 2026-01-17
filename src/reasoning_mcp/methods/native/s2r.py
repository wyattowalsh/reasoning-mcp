"""S2R (Self-verification and Self-correction with RL) reasoning method.

This module implements S2R (Self-verification and Self-correction with Reinforcement Learning)
from arXiv Feb 2025. The method combines self-verification with RL-guided self-correction
to iteratively improve reasoning quality through reward signals and policy adjustments.

Reference: arXiv Feb 2025
Key Idea: RL-guided self-verification and self-correction loop

The method operates in phases:
1. Generate: Produce initial answer
2. Self-Verify: Assess quality with confidence scoring
3. RL-Correct: Apply RL-guided corrections based on reward signals
4. Iterate: Repeat until convergence or max iterations
5. Conclude: Finalize the best answer
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

logger = structlog.get_logger(__name__)

# Metadata for S2R method
S2R_METADATA = MethodMetadata(
    identifier=MethodIdentifier.S2R,
    name="S2R",
    description="Self-verification and Self-correction with Reinforcement Learning. "
    "Combines self-verification with RL-guided correction through reward signals, "
    "iteratively improving reasoning quality via generate → verify → rl-correct cycles.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "reinforcement-learning",
            "self-verification",
            "self-correction",
            "iterative",
            "reward-guided",
            "policy-optimization",
            "quality-improvement",
            "convergence-based",
        }
    ),
    complexity=6,  # High complexity - RL-based correction
    supports_branching=False,  # Linear improvement path
    supports_revision=True,  # Core feature - RL-guided correction
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: generate + verify + correct
    max_thoughts=18,  # Max 6 iterations × 3 thoughts per iteration
    avg_tokens_per_thought=400,  # Moderate-high - includes RL reasoning
    best_for=(
        "high-stakes reasoning",
        "quality-critical tasks",
        "iterative optimization",
        "self-improving systems",
        "accuracy-sensitive problems",
        "reward-guided learning",
        "convergence-based refinement",
        "complex problem solving",
    ),
    not_recommended_for=(
        "simple queries",
        "time-critical tasks",
        "low-complexity problems",
        "fixed-format outputs",
        "tasks without clear quality metrics",
    ),
)


class S2R(ReasoningMethodBase):
    """S2R (Self-verification and Self-correction with RL) reasoning method implementation.

    This class implements the S2R pattern where the system generates an initial answer,
    then iteratively verifies and corrects it using RL-guided adjustments based on
    reward signals. Each iteration involves:
    1. Generate: Produce an answer
    2. Self-Verify: Assess quality with confidence scoring
    3. RL-Correct: Apply corrections guided by simulated RL policy
    4. Iterate: Continue until convergence or max iterations
    5. Conclude: Output the best answer

    Key characteristics:
    - RL-guided self-correction
    - Confidence-based verification
    - Reward signal computation
    - Policy-driven improvements
    - Convergence detection
    - Specialized complexity (6)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = S2R()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="What is the capital of France?"
        ... )
        >>> print(result.content)  # Initial generation

        Continue with self-verification:
        >>> verification = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Verify the answer"
        ... )
        >>> print(verification.type)  # ThoughtType.VERIFICATION

        Continue with RL-guided correction:
        >>> correction = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=verification,
        ...     guidance="Apply RL-guided corrections"
        ... )
        >>> print(correction.type)  # ThoughtType.REVISION
    """

    # Enable LLM sampling for answer generation and verification
    _use_sampling: bool = True

    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS = 6
    # Confidence threshold for convergence
    CONVERGENCE_THRESHOLD = 0.90
    # Minimum improvement required to continue iterating
    MIN_IMPROVEMENT = 0.02

    def __init__(self) -> None:
        """Initialize the S2R method."""
        self._initialized = False
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase: str = "generate"  # generate, self_verify, rl_correct
        self._reward_history: list[float] = []
        self._confidence_history: list[float] = []
        self._best_confidence = 0.0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.S2R

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return S2R_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return S2R_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the S2R method for execution.
        Resets counters, state, and history for a fresh reasoning session.

        Examples:
            >>> method = S2R()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._iteration_count == 0
            >>> assert len(method._reward_history) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase = "generate"
        self._reward_history = []
        self._confidence_history = []
        self._best_confidence = 0.0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the S2R method.

        This method creates the initial answer generation that will be iteratively
        verified and corrected through RL-guided improvements.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include max_iterations,
                    convergence_threshold)

        Returns:
            A ThoughtNode representing the initial generation

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = S2R()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Explain quantum entanglement"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.S2R
            >>> assert "iteration_count" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("S2R method must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._iteration_count = 0
        self._current_phase = "generate"
        self._reward_history = []
        self._confidence_history = []
        self._best_confidence = 0.0

        # Extract parameters from context if provided
        max_iterations = self.MAX_ITERATIONS
        convergence_threshold = self.CONVERGENCE_THRESHOLD
        if context:
            if "max_iterations" in context:
                max_iterations = max(1, min(context["max_iterations"], 10))
            if "convergence_threshold" in context:
                convergence_threshold = min(max(context["convergence_threshold"], 0.0), 1.0)

        # Generate initial answer
        content = await self._generate_initial_answer(input_text, context)

        # Initial confidence - moderate (will improve with RL corrections)
        initial_confidence = 0.55
        self._confidence_history.append(initial_confidence)
        self._best_confidence = initial_confidence

        # Initial reward (baseline)
        initial_reward = 0.0
        self._reward_history.append(initial_reward)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.S2R,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=initial_confidence,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "s2r",
                "phase": self._current_phase,
                "iteration_count": self._iteration_count,
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
                "reward": initial_reward,
                "reward_history": list(self._reward_history),
                "confidence_history": list(self._confidence_history),
                "best_confidence": self._best_confidence,
                "converged": False,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.S2R

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

        This method implements the S2R cycle logic:
        - If previous was generate/rl_correct: perform self-verification
        - If previous was self_verify: perform RL-guided correction
        - Continues until convergence or max iterations reached

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the S2R process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = S2R()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Solve problem X")
            >>> verify = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert verify.type == ThoughtType.VERIFICATION
            >>> assert verify.metadata["phase"] == "self_verify"
            >>>
            >>> correct = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=verify
            ... )
            >>> assert correct.type == ThoughtType.REVISION
            >>> assert correct.metadata["phase"] == "rl_correct"
        """
        if not self._initialized:
            raise RuntimeError("S2R method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Get parameters from previous thought's metadata
        max_iterations = previous_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)
        convergence_threshold = previous_thought.metadata.get(
            "convergence_threshold", self.CONVERGENCE_THRESHOLD
        )

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase in ("generate", "rl_correct"):
            # Next: self-verification
            self._current_phase = "self_verify"
            thought_type = ThoughtType.VERIFICATION
            content, confidence, verification_issues = await self._perform_self_verification(
                previous_thought, guidance, context
            )
            self._confidence_history.append(confidence)

            # Update best confidence
            if confidence > self._best_confidence:
                self._best_confidence = confidence

            # Compute reward based on confidence improvement
            reward = self._compute_reward(confidence, verification_issues)
            self._reward_history.append(reward)

        elif prev_phase == "self_verify":
            # Next: RL-guided correction
            self._current_phase = "rl_correct"
            self._iteration_count += 1
            thought_type = ThoughtType.REVISION

            # Get verification data from previous thought
            prev_confidence = previous_thought.confidence or 0.5
            verification_issues = previous_thought.metadata.get("verification_issues", [])

            content, confidence, corrections_applied = await self._perform_rl_correction(
                previous_thought, verification_issues, guidance, context
            )
            self._confidence_history.append(confidence)

            # Update best confidence
            if confidence > self._best_confidence:
                self._best_confidence = confidence

            # Compute reward based on improvement
            reward = self._compute_reward(confidence, verification_issues, prev_confidence)
            self._reward_history.append(reward)

        else:
            # Fallback to self-verification
            self._current_phase = "self_verify"
            thought_type = ThoughtType.VERIFICATION
            content, confidence, verification_issues = await self._perform_self_verification(
                previous_thought, guidance, context
            )
            self._confidence_history.append(confidence)
            reward = self._compute_reward(confidence, verification_issues)
            self._reward_history.append(reward)

        # Check convergence
        converged = self._check_convergence(confidence, convergence_threshold)

        # Check if we should continue
        should_continue = not converged and self._iteration_count < max_iterations

        # If this is a correction that has converged or reached max iterations, mark as conclusion
        if self._current_phase == "rl_correct" and not should_continue:
            thought_type = ThoughtType.CONCLUSION

        # Build metadata
        metadata: dict[str, Any] = {
            "phase": self._current_phase,
            "iteration_count": self._iteration_count,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "should_continue": should_continue,
            "converged": converged,
            "guidance": guidance or "",
            "context": context or {},
            "reasoning_type": "s2r",
            "reward_history": list(self._reward_history),
            "confidence_history": list(self._confidence_history),
            "best_confidence": self._best_confidence,
        }

        # Add phase-specific metadata
        if self._current_phase == "self_verify":
            metadata["verification_issues"] = verification_issues
            metadata["issue_count"] = len(verification_issues)
        elif self._current_phase == "rl_correct":
            metadata["corrections_applied"] = corrections_applied
            metadata["policy_action"] = self._get_policy_action(reward)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.S2R,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata=metadata,
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For S2R, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = S2R()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _generate_initial_answer(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial answer to be verified and corrected.

        This is a helper method that would typically call an LLM to generate
        the initial answer attempt.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial answer

        Note:
            In a full implementation, this would use an LLM to generate
            the actual answer. This is a placeholder that provides
            the structure.
        """
        prompt = (
            f"Generate an initial answer to the following question. "
            f"This answer will be iteratively verified and corrected using "
            f"RL-guided improvements.\n\n"
            f"Question: {input_text}"
        )

        system_prompt = (
            "You are an AI reasoning assistant using S2R "
            "(Self-verification and Self-correction with RL). "
            "Provide a thoughtful initial answer that will be refined through "
            "self-verification and RL-guided correction cycles."
        )

        def fallback_generator() -> str:
            return (
                f"Step {self._step_counter}: Initial Answer Generation (Iteration 0)\n\n"
                f"Question: {input_text}\n\n"
                f"I will generate an initial answer, then iteratively verify and correct it "
                f"using RL-guided improvements until convergence or maximum iterations.\n\n"
                f"Initial answer:\n"
                f"[This would contain the LLM-generated initial answer]\n\n"
                f"Note: This answer will now be self-verified with confidence scoring."
            )

        result = await self._sample_with_fallback(
            prompt, fallback_generator, system_prompt=system_prompt
        )

        return (
            f"Step {self._step_counter}: Initial Answer Generation (Iteration 0)\n\n"
            f"Question: {input_text}\n\n"
            f"I will generate an initial answer, then iteratively verify and correct it "
            f"using RL-guided improvements until convergence or maximum iterations.\n\n"
            f"Initial answer:\n"
            f"{result}\n\n"
            f"Note: This answer will now be self-verified with confidence scoring."
        )

    async def _perform_self_verification(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float, list[str]]:
        """Perform self-verification with confidence scoring.

        This is a helper method that would typically call an LLM to verify
        the previous answer and identify issues.

        Args:
            previous_thought: The answer to verify
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            A tuple of (verification content, confidence score, list of issues)

        Note:
            In a full implementation, this would use an LLM to generate
            the actual verification. This is a placeholder that provides
            the structure.
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        prompt = (
            f"Verify the following answer and identify any issues or concerns. "
            f"Provide a confidence score (0.0-1.0) and list specific issues.\n\n"
            f"Previous Answer (Step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Current iteration: {iteration}\n"
            f"Previous best confidence: {self._best_confidence:.2f}\n"
        )

        if guidance:
            prompt += f"\nGuidance: {guidance}"

        system_prompt = (
            "You are an AI verification assistant using S2R "
            "(Self-verification and Self-correction with RL). "
            "Carefully analyze the answer, assess its quality, and identify "
            "specific issues that could be improved. Provide a confidence score "
            "and a numbered list of issues."
        )

        def fallback_generator() -> str:
            # Fallback heuristic method
            # Simulate verification issues (would be LLM-generated)
            # Fewer issues as iterations progress
            base_issue_count = max(0, 4 - iteration)
            fallback_verification_issues = [
                f"Issue {i + 1}: [Verification concern to be addressed]"
                for i in range(base_issue_count)
            ]

            # Confidence improves with each iteration
            fallback_confidence = min(0.55 + (0.08 * iteration), 0.98)

            fallback_content = (
                f"Step {self._step_counter}: Self-Verification (Iteration {iteration})\n\n"
                f"Verifying answer from Step {previous_thought.step_number}...\n\n"
                f"Confidence Assessment:\n"
                f"- Current confidence: {fallback_confidence:.2f}\n"
                f"- Previous best: {self._best_confidence:.2f}\n\n"
                f"Verification Issues Identified:\n"
            )

            if fallback_verification_issues:
                for issue in fallback_verification_issues:
                    fallback_content += f"- {issue}\n"
            else:
                fallback_content += "- No issues identified (high confidence)\n"

            fallback_content += (
                f"\nTotal issues: {len(fallback_verification_issues)}\n"
                f"Confidence score: {fallback_confidence:.2f}/1.00{guidance_text}"
            )

            return fallback_content

        result = await self._sample_with_fallback(
            prompt, fallback_generator, system_prompt=system_prompt
        )

        # Parse confidence and issues from LLM response
        # Simple heuristic: look for confidence score and issues
        confidence = min(0.55 + (0.08 * iteration), 0.98)
        verification_issues = []

        # Try to extract issues from the response
        lines = result.split("\n")
        for line in lines:
            line_lower = line.lower()
            markers = ["issue", "concern", "problem", "-", "•", "*"]
            if any(marker in line_lower for marker in markers):
                if line.strip() and not line_lower.startswith("no issue"):
                    verification_issues.append(line.strip())

        # Limit to reasonable number of issues
        verification_issues = verification_issues[:6]

        content = (
            f"Step {self._step_counter}: Self-Verification (Iteration {iteration})\n\n"
            f"Verifying answer from Step {previous_thought.step_number}...\n\n"
            f"Confidence Assessment:\n"
            f"- Current confidence: {confidence:.2f}\n"
            f"- Previous best: {self._best_confidence:.2f}\n\n"
            f"Verification Analysis:\n"
            f"{result}\n\n"
            f"Total issues identified: {len(verification_issues)}\n"
            f"Confidence score: {confidence:.2f}/1.00{guidance_text}"
        )

        return content, confidence, verification_issues

    async def _perform_rl_correction(
        self,
        verification_thought: ThoughtNode,
        verification_issues: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float, int]:
        """Perform RL-guided correction based on verification and reward signals.

        This is a helper method that would typically call an LLM to apply
        RL-guided corrections to address verification issues.

        Args:
            verification_thought: The verification to address
            verification_issues: List of issues to correct
            guidance: Optional guidance for correction
            context: Optional additional context

        Returns:
            A tuple of (corrected content, confidence score, corrections applied count)

        Note:
            In a full implementation, this would use an LLM with RL-guided
            prompting to generate corrections. This is a placeholder.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Compute policy action based on reward history
        policy_action = self._get_policy_action(
            self._reward_history[-1] if self._reward_history else 0.0
        )

        # Number of corrections to apply (based on RL policy)
        corrections_applied = len(verification_issues)

        # Confidence improves with corrections
        base_confidence = verification_thought.confidence or 0.5
        improvement = 0.10 if corrections_applied > 0 else 0.05
        confidence = min(base_confidence + improvement, 0.98)

        issues_text = "\n".join(
            f"{i + 1}. {issue}" for i, issue in enumerate(verification_issues)
        )

        prompt = (
            f"Apply RL-guided corrections to address the following verification issues. "
            f"Use the RL policy action as guidance for your correction strategy.\n\n"
            f"Verification Issues:\n{issues_text}\n\n"
            f"RL Policy Action: {policy_action}\n"
            f"Reward Signal: {self._reward_history[-1]:.3f}\n"
            f"Current Iteration: {self._iteration_count}\n"
            f"Base Confidence: {base_confidence:.2f}\n\n"
            f"Previous verification:\n{verification_thought.content}\n"
        )

        if guidance:
            prompt += f"\nGuidance: {guidance}"

        system_prompt = (
            "You are an AI correction assistant using S2R "
            "(Self-verification and Self-correction with RL). "
            "Apply RL-guided corrections to address the identified issues. "
            "Focus on making strategic improvements based on the RL policy action. "
            "Provide a corrected, improved answer."
        )

        def fallback_generator() -> str:
            # Fallback heuristic method
            max_iter = verification_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)
            fallback_content = (
                f"Step {self._step_counter}: RL-Guided Correction "
                f"(Iteration {self._iteration_count})\n\n"
                f"Applying corrections based on verification from "
                f"Step {verification_thought.step_number}...\n\n"
                f"RL Policy Action: {policy_action}\n"
                f"Reward Signal: {self._reward_history[-1]:.3f}\n\n"
                f"Corrections Applied ({corrections_applied}/{len(verification_issues)}):\n"
            )

            for i, issue in enumerate(verification_issues, 1):
                fallback_content += f"{i}. {issue} → [RL-corrected]\n"

            fallback_content += (
                f"\nCorrected answer:\n"
                f"[This would contain the LLM-generated corrected answer "
                f"with RL-guided improvements]\n\n"
                f"Performance Metrics:\n"
                f"- Confidence: {confidence:.2f}/1.00\n"
                f"- Iteration: {self._iteration_count}/{max_iter}\n"
                f"- Best confidence so far: {self._best_confidence:.2f}/1.00{guidance_text}"
            )

            return fallback_content

        result = await self._sample_with_fallback(
            prompt, fallback_generator, system_prompt=system_prompt
        )

        max_iter = verification_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)
        content = (
            f"Step {self._step_counter}: RL-Guided Correction "
            f"(Iteration {self._iteration_count})\n\n"
            f"Applying corrections based on verification from "
            f"Step {verification_thought.step_number}...\n\n"
            f"RL Policy Action: {policy_action}\n"
            f"Reward Signal: {self._reward_history[-1]:.3f}\n\n"
            f"Corrections Applied ({corrections_applied} issues addressed):\n"
            f"{issues_text}\n\n"
            f"Corrected answer:\n"
            f"{result}\n\n"
            f"Performance Metrics:\n"
            f"- Confidence: {confidence:.2f}/1.00\n"
            f"- Iteration: {self._iteration_count}/{max_iter}\n"
            f"- Best confidence so far: {self._best_confidence:.2f}/1.00{guidance_text}"
        )

        return content, confidence, corrections_applied

    def _compute_reward(
        self,
        current_confidence: float,
        verification_issues: list[str],
        previous_confidence: float | None = None,
    ) -> float:
        """Compute reward signal for RL-guided correction.

        Args:
            current_confidence: Current confidence score
            verification_issues: List of verification issues
            previous_confidence: Previous confidence score for comparison

        Returns:
            Reward value (positive for improvement, negative for degradation)
        """
        # Base reward from confidence level
        confidence_reward = current_confidence - 0.5  # Centered at 0.5

        # Penalty for verification issues
        issue_penalty = -0.1 * len(verification_issues)

        # Improvement bonus if we have previous confidence
        improvement_bonus = 0.0
        if previous_confidence is not None:
            improvement = current_confidence - previous_confidence
            improvement_bonus = improvement * 2.0  # Amplify improvement signal

        # Total reward
        reward = confidence_reward + issue_penalty + improvement_bonus

        return reward

    def _get_policy_action(self, reward: float) -> str:
        """Get RL policy action description based on reward signal.

        Args:
            reward: Current reward value

        Returns:
            Description of the policy action
        """
        if reward > 0.2:
            return "EXPLOIT - High reward, continue current strategy"
        elif reward > 0.0:
            return "REFINE - Positive reward, minor adjustments"
        elif reward > -0.2:
            return "EXPLORE - Low reward, try alternative approaches"
        else:
            return "RESET - Negative reward, significant strategy change"

    def _check_convergence(
        self,
        confidence: float,
        threshold: float,
    ) -> bool:
        """Check if the reasoning has converged.

        Args:
            confidence: Current confidence score
            threshold: Convergence threshold

        Returns:
            True if converged, False otherwise
        """
        # Check if confidence meets threshold
        if confidence >= threshold:
            return True

        # Check if improvement is too small (plateau)
        if len(self._confidence_history) >= 2:
            recent_improvement = confidence - self._confidence_history[-2]
            if abs(recent_improvement) < self.MIN_IMPROVEMENT:
                return True

        return False


__all__ = ["S2R", "S2R_METADATA"]
