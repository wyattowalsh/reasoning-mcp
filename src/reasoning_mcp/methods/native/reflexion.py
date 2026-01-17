"""Reflexion reasoning method.

This module implements Reflexion (Shinn et al. 2023), an advanced reasoning method
that combines self-reflection with episodic memory for learning from past attempts.
The method tracks episodes with memory persistence between attempts, allowing it to
learn from failures and improve iteratively.

Reflexion differs from Self-Reflection by maintaining episodic memory that persists
across multiple attempts, enabling learning from past mistakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_confirmation,
    elicit_feedback,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from fastmcp.server import Context

    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for Reflexion method
REFLEXION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.REFLEXION,
    name="Reflexion",
    description="Self-reflection with episodic memory for learning from past attempts. "
    "Tracks episodes with memory persistence between attempts, enabling iterative "
    "improvement through attempt → evaluate → reflect → retry cycles.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "episodic-memory",
            "self-reflection",
            "learning",
            "iterative",
            "memory-persistence",
            "self-improvement",
            "failure-recovery",
            "adaptive",
        }
    ),
    complexity=7,  # Advanced complexity - episodic memory and learning
    supports_branching=False,  # Linear episode progression
    supports_revision=True,  # Core feature - revising through memory
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: attempt + evaluate + reflect + retry
    max_thoughts=25,  # Multiple episodes with 4 phases each
    avg_tokens_per_thought=450,  # Moderate - includes memory and reflection
    best_for=(
        "complex problem solving",
        "learning from failures",
        "iterative optimization",
        "trial-and-error tasks",
        "adaptive systems",
        "self-improving agents",
        "error recovery",
        "multi-attempt problems",
    ),
    not_recommended_for=(
        "simple one-shot queries",
        "time-critical decisions",
        "problems requiring external feedback",
        "tasks with clear solutions",
    ),
)


class Reflexion(ReasoningMethodBase):
    """Reflexion reasoning method implementation.

    This class implements the Reflexion pattern (Shinn et al. 2023) where the system
    maintains episodic memory across multiple attempts at solving a problem. Each
    episode involves:
    1. Attempt: Try to solve the problem
    2. Evaluate: Assess the quality of the attempt
    3. Reflect: Analyze what went wrong and extract lessons
    4. Retry: Apply learned lessons to a new attempt

    Key characteristics:
    - Episodic memory persistence
    - Learning from failures
    - Iterative improvement across episodes
    - Memory-guided retry attempts
    - Advanced complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = Reflexion()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Solve this coding problem: reverse a linked list"
        ... )
        >>> print(result.content)  # Initial attempt

        Continue with evaluation:
        >>> evaluation = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Evaluate the solution"
        ... )
        >>> print(evaluation.type)  # ThoughtType.VERIFICATION

        Continue with reflection:
        >>> reflection = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=evaluation,
        ...     guidance="Reflect on mistakes"
        ... )
        >>> print(reflection.type)  # ThoughtType.INSIGHT

        Retry with learned lessons:
        >>> retry = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=reflection,
        ...     guidance="Apply lessons learned"
        ... )
        >>> print(retry.type)  # ThoughtType.REVISION
    """

    # Quality threshold for success
    QUALITY_THRESHOLD = 0.85
    # Maximum episodes to prevent infinite loops
    MAX_EPISODES = 3

    # Enable LLM sampling for generating attempts, evaluations, and reflections
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Reflexion method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._episode_number = 0
        self._current_phase: str = "attempt"  # attempt, evaluate, reflect, retry
        self._episodic_memory: list[dict[str, Any]] = []  # Persistent memory across episodes
        self.enable_elicitation = enable_elicitation
        self._ctx: Context | None = None
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.REFLEXION

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return REFLEXION_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return REFLEXION_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Reflexion method for execution.
        Resets counters, state, and episodic memory for a fresh reasoning session.

        Examples:
            >>> method = Reflexion()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._episode_number == 0
            >>> assert len(method._episodic_memory) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._episode_number = 0
        self._current_phase = "attempt"
        self._episodic_memory = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Reflexion method.

        This method creates the initial attempt at solving the problem.
        It starts the first episode with an attempt phase.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include quality_threshold)
            execution_context: Optional ExecutionContext for elicitation

        Returns:
            A ThoughtNode representing the initial attempt

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Reflexion()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How to implement binary search?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.REFLEXION
            >>> assert "episode_number" in thought.metadata
            >>> assert thought.metadata["episode_number"] == 1
        """
        if not self._initialized:
            raise RuntimeError("Reflexion method must be initialized before execution")

        # Store execution context for LLM sampling and elicitation
        self._execution_context = execution_context
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        # Reset for new execution
        self._step_counter = 1
        self._episode_number = 1
        self._current_phase = "attempt"
        self._episodic_memory = []

        # Extract quality threshold from context if provided
        quality_threshold = self.QUALITY_THRESHOLD
        if context and "quality_threshold" in context:
            quality_threshold = min(max(context["quality_threshold"], 0.0), 1.0)

        # Generate initial attempt using LLM sampling if available
        if self._execution_context and self._execution_context.can_sample:
            try:
                content = await self._sample_attempt(input_text, context)
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="execute",
                    error=str(e),
                )
                # Fall back to heuristic implementation if sampling fails
                content = self._generate_attempt(input_text, context)
        else:
            # Use heuristic implementation
            content = self._generate_attempt(input_text, context)

        # Initial quality score (moderate - room for improvement)
        initial_quality = 0.5

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.REFLEXION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,  # Initial confidence - will improve with episodes
            quality_score=initial_quality,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "reflexion",
                "phase": self._current_phase,
                "episode_number": self._episode_number,
                "quality_threshold": quality_threshold,
                "needs_improvement": initial_quality < quality_threshold,
                "episodic_memory_size": len(self._episodic_memory),
                "lessons_learned": [],
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.REFLEXION

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

        This method implements the Reflexion episode cycle logic:
        - If previous was attempt: generate evaluation
        - If previous was evaluate: generate reflection
        - If previous was reflect: generate retry (new attempt with memory)
        - If previous was retry: generate evaluation (continue cycle)
        - Continues until quality threshold met or max episodes reached

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for elicitation

        Returns:
            A new ThoughtNode continuing the Reflexion process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Reflexion()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Solve problem X")
            >>> evaluate = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert evaluate.type == ThoughtType.VERIFICATION
            >>> assert evaluate.metadata["phase"] == "evaluate"
            >>>
            >>> reflect = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=evaluate
            ... )
            >>> assert reflect.type == ThoughtType.INSIGHT
            >>> assert reflect.metadata["phase"] == "reflect"
        """
        if not self._initialized:
            raise RuntimeError("Reflexion method must be initialized before continuation")

        # Store execution context for LLM sampling and elicitation
        if execution_context:
            self._execution_context = execution_context
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        # Increment step counter
        self._step_counter += 1

        # Get quality threshold from previous thought's metadata
        quality_threshold = previous_thought.metadata.get(
            "quality_threshold", self.QUALITY_THRESHOLD
        )

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "attempt")

        # Optional elicitation: get user feedback during reflection phases
        elicited_response = ""
        if self.enable_elicitation and self._ctx and not guidance:
            try:
                elicit_config = ElicitationConfig(
                    timeout=60, required=False, default_on_timeout=None
                )

                if prev_phase == "evaluate":
                    # Before reflecting, ask user for feedback on the evaluation
                    feedback = await elicit_feedback(
                        self._ctx,
                        f"Episode {self._episode_number} - The evaluation shows quality score "
                        f"{previous_thought.quality_score:.2f}. What are your thoughts on what "
                        f"went wrong and how to improve?",
                        config=elicit_config,
                    )
                    if feedback.feedback:
                        elicited_response = f"\n\n[User Feedback]: {feedback.feedback}"
                        session.metrics.elicitations_made += 1

                elif prev_phase == "reflect":
                    # After reflection, ask if user wants to continue iterating
                    needs_improvement = (previous_thought.quality_score or 0.5) < quality_threshold
                    if needs_improvement and self._episode_number < self.MAX_EPISODES:
                        quality_msg = (
                            f"Reflection complete for episode {self._episode_number}. "
                            f"Quality is {previous_thought.quality_score:.2f}/"
                            f"{quality_threshold:.2f}. Continue with another attempt?"
                        )
                        confirmation = await elicit_confirmation(
                            self._ctx,
                            quality_msg,
                            config=elicit_config,
                        )
                        if confirmation.confirmed:
                            elicited_response = "\n\n[User Choice]: Continuing with next attempt"
                        else:
                            elicited_response = "\n\n[User Choice]: Stopping iteration"
                            # User chose to stop, so we'll mark as conclusion
                            quality_threshold = 0.0  # Force early exit
                        session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="continue_reasoning",
                    error=str(e),
                )
                # Elicitation failed - continue without it

        # Update guidance with elicited response
        combined_guidance = guidance or ""
        if elicited_response:
            combined_guidance = (guidance or "") + elicited_response

        if prev_phase == "attempt":
            # Next: evaluate
            thought_type, content, quality_score, confidence = await self._transition_to_evaluate(
                previous_thought, combined_guidance, context, quality_threshold
            )

        elif prev_phase == "evaluate":
            # Next: reflect
            thought_type, content, quality_score, confidence = await self._transition_to_reflect(
                previous_thought, combined_guidance, context, quality_threshold
            )

        elif prev_phase == "reflect":
            # Next: retry (new attempt with learned lessons)
            thought_type, content, quality_score, confidence = await self._transition_to_retry(
                previous_thought, combined_guidance, context, quality_threshold
            )

        elif prev_phase == "retry":
            # Next: evaluate (continue cycle)
            thought_type, content, quality_score, confidence = await self._transition_to_evaluate(
                previous_thought, combined_guidance, context, quality_threshold
            )

        else:
            # Fallback to evaluate
            self._current_phase = "evaluate"
            thought_type = ThoughtType.VERIFICATION
            content = self._generate_evaluation(previous_thought, combined_guidance, context)
            quality_score = previous_thought.quality_score or 0.5
            confidence = 0.6

        # Check if we should continue or conclude
        should_continue = (
            quality_score < quality_threshold and self._episode_number < self.MAX_EPISODES
        )

        # If this is a retry that meets threshold, mark as conclusion
        if self._current_phase == "retry" and not should_continue:
            thought_type = ThoughtType.CONCLUSION

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.REFLEXION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "episode_number": self._episode_number,
                "quality_threshold": quality_threshold,
                "needs_improvement": should_continue,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "reflexion",
                "previous_quality": previous_thought.quality_score,
                "episodic_memory_size": len(self._episodic_memory),
                "lessons_learned": self._get_lessons_learned(),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Reflexion, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Reflexion()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _transition_to_evaluate(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
        quality_threshold: float,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to evaluation phase.

        Args:
            previous_thought: The attempt to evaluate
            guidance: Optional guidance
            context: Optional context
            quality_threshold: Quality threshold for success

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "evaluate"
        thought_type = ThoughtType.VERIFICATION

        # Generate evaluation using LLM sampling if available
        if self._execution_context and self._execution_context.can_sample:
            try:
                content, quality_score = await self._sample_evaluation(
                    previous_thought, guidance, context
                )
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="_transition_to_evaluate",
                    error=str(e),
                )
                # Fall back to heuristic implementation
                content = self._generate_evaluation(previous_thought, guidance, context)
                # Evaluate quality based on attempt
                base_quality = previous_thought.quality_score or 0.5
                # Quality improves slightly with each episode (learning effect)
                quality_score = min(base_quality + (0.05 * (self._episode_number - 1)), 0.95)
        else:
            content = self._generate_evaluation(previous_thought, guidance, context)
            base_quality = previous_thought.quality_score or 0.5
            quality_score = min(base_quality + (0.05 * (self._episode_number - 1)), 0.95)

        confidence = 0.7

        return thought_type, content, quality_score, confidence

    async def _transition_to_reflect(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
        quality_threshold: float,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to reflection phase.

        Args:
            previous_thought: The evaluation to reflect on
            guidance: Optional guidance
            context: Optional context
            quality_threshold: Quality threshold for success

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "reflect"
        thought_type = ThoughtType.INSIGHT

        # Generate reflection using LLM sampling if available
        if self._execution_context and self._execution_context.can_sample:
            try:
                content, lesson = await self._sample_reflection(previous_thought, guidance, context)
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="_transition_to_reflect",
                    error=str(e),
                )
                # Fall back to heuristic implementation
                content = self._generate_reflection(previous_thought, guidance, context)
                quality = previous_thought.quality_score or 0.5
                lesson = self._extract_lesson(previous_thought, quality)
        else:
            content = self._generate_reflection(previous_thought, guidance, context)
            quality = previous_thought.quality_score or 0.5
            lesson = self._extract_lesson(previous_thought, quality)

        # Store reflection in episodic memory
        quality = previous_thought.quality_score or 0.5
        self._episodic_memory.append(
            {
                "episode": self._episode_number,
                "quality": quality,
                "lesson": lesson,
                "phase": "reflect",
            }
        )

        # Reflection maintains quality from evaluation
        quality_score = previous_thought.quality_score or 0.5
        confidence = 0.75

        return thought_type, content, quality_score, confidence

    async def _transition_to_retry(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
        quality_threshold: float,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to retry phase (new attempt with memory).

        Args:
            previous_thought: The reflection to apply
            guidance: Optional guidance
            context: Optional context
            quality_threshold: Quality threshold for success

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "retry"
        # Only increment episode if under max limit
        if self._episode_number < self.MAX_EPISODES:
            self._episode_number += 1
        thought_type = ThoughtType.REVISION

        # Get the original input from metadata
        original_input = previous_thought.metadata.get("input", "")

        # Generate retry using LLM sampling if available
        if self._execution_context and self._execution_context.can_sample:
            try:
                content = await self._sample_retry(
                    original_input, previous_thought, guidance, context
                )
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="_transition_to_retry",
                    error=str(e),
                )
                # Fall back to heuristic implementation
                content = self._generate_retry(previous_thought, guidance, context)
        else:
            content = self._generate_retry(previous_thought, guidance, context)

        # Retry improves quality based on learned lessons
        base_quality = previous_thought.quality_score or 0.5
        improvement = 0.15 * self._episode_number  # More improvement with more episodes
        quality_score = min(base_quality + improvement, 1.0)
        confidence = min(0.5 + (0.15 * self._episode_number), 0.95)

        return thought_type, content, quality_score, confidence

    def _generate_attempt(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an attempt at solving the problem.

        This is a helper method that would typically call an LLM to generate
        the initial attempt at solving the problem.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the attempt

        Note:
            In a full implementation, this would use an LLM to generate
            the actual attempt. This is a placeholder that provides
            the structure.
        """
        memory_context = self._format_memory_context()

        return (
            f"Step {self._step_counter}: Attempt (Episode {self._episode_number})\n\n"
            f"Problem: {input_text}\n\n"
            f"{memory_context}"
            f"Let me attempt to solve this problem. I will then evaluate my solution, "
            f"reflect on any mistakes, and retry if needed.\n\n"
            f"Attempt:\n"
            f"[This would contain the LLM-generated attempt at solving the problem]\n\n"
            f"Note: This is episode {self._episode_number}. I will evaluate this attempt next."
        )

    def _generate_evaluation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an evaluation of the attempt.

        This is a helper method that would typically call an LLM to evaluate
        the quality and correctness of the attempt.

        Args:
            previous_thought: The attempt to evaluate
            guidance: Optional guidance for the evaluation
            context: Optional additional context

        Returns:
            The content for the evaluation

        Note:
            In a full implementation, this would use an LLM to generate
            the actual evaluation. This is a placeholder that provides
            the structure.
        """
        episode = self._episode_number
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Evaluation (Episode {episode})\n\n"
            f"Evaluating the attempt from Step {previous_thought.step_number}...\n\n"
            f"Correctness Assessment:\n"
            f"[LLM would evaluate if the solution is correct]\n\n"
            f"Quality Assessment:\n"
            f"[LLM would evaluate the quality of the approach]\n\n"
            f"Issues Identified:\n"
            f"[LLM would list specific problems or errors]\n\n"
            f"Overall Score: {previous_thought.quality_score:.2f}/1.00{guidance_text}"
        )

    def _generate_reflection(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a reflection on what went wrong and what to learn.

        This is a helper method that would typically call an LLM to analyze
        the evaluation and extract actionable lessons.

        Args:
            previous_thought: The evaluation to reflect on
            guidance: Optional guidance for the reflection
            context: Optional additional context

        Returns:
            The content for the reflection

        Note:
            In a full implementation, this would use an LLM to generate
            the actual reflection. This is a placeholder that provides
            the structure.
        """
        episode = self._episode_number
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        memory_summary = self._summarize_memory()

        return (
            f"Step {self._step_counter}: Reflection (Episode {episode})\n\n"
            f"Reflecting on the evaluation from Step {previous_thought.step_number}...\n\n"
            f"{memory_summary}"
            f"What went wrong:\n"
            f"[LLM would analyze the root causes of failures]\n\n"
            f"Key insights:\n"
            f"[LLM would extract key learnings]\n\n"
            f"Lesson learned:\n"
            f"[LLM would formulate a specific, actionable lesson]\n\n"
            f"This lesson will be stored in episodic memory for the next attempt.{guidance_text}"
        )

    def _generate_retry(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a retry attempt incorporating learned lessons.

        This is a helper method that would typically call an LLM to generate
        a new attempt that applies the lessons from episodic memory.

        Args:
            previous_thought: The reflection to apply
            guidance: Optional guidance for the retry
            context: Optional additional context

        Returns:
            The content for the retry

        Note:
            In a full implementation, this would use an LLM to generate
            the actual retry attempt. This is a placeholder that provides
            the structure.
        """
        episode = self._episode_number
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        lessons = self._format_lessons_learned()
        threshold = previous_thought.metadata.get("quality_threshold", 0.85)

        return (
            f"Step {self._step_counter}: Retry (Episode {episode})\n\n"
            f"Based on the reflection from Step {previous_thought.step_number}, "
            f"I will now retry with learned lessons applied.\n\n"
            f"{lessons}"
            f"New attempt:\n"
            f"[LLM would generate improved attempt incorporating all lessons]\n\n"
            f"Improvements applied:\n"
            f"[LLM would describe how lessons were applied]\n\n"
            f"Episode {episode} quality: {previous_thought.quality_score:.2f}/1.00 → "
            f"Target: {threshold:.2f}/1.00{guidance_text}"
        )

    def _extract_lesson(
        self,
        evaluation_thought: ThoughtNode,
        quality: float,
    ) -> str:
        """Extract a lesson learned from an evaluation.

        Args:
            evaluation_thought: The evaluation thought
            quality: The quality score of the attempt

        Returns:
            A lesson learned string
        """
        episode = self._episode_number
        if quality < 0.3:
            return f"Episode {episode}: Fundamental approach issue - need complete redesign"
        elif quality < 0.6:
            return f"Episode {episode}: Major gaps - need significant improvements"
        elif quality < 0.85:
            return f"Episode {episode}: Close but needs refinement in details"
        else:
            return f"Episode {episode}: Good quality - minor polish needed"

    def _get_lessons_learned(self) -> list[str]:
        """Get all lessons learned from episodic memory.

        Returns:
            List of lesson strings
        """
        return [entry["lesson"] for entry in self._episodic_memory]

    def _format_memory_context(self) -> str:
        """Format episodic memory as context for attempts.

        Returns:
            Formatted memory context string
        """
        if not self._episodic_memory:
            return ""

        lessons = self._get_lessons_learned()
        if not lessons:
            return ""

        return (
            "Previous attempts and lessons learned:\n"
            + "\n".join(f"- {lesson}" for lesson in lessons)
            + "\n\n"
        )

    def _format_lessons_learned(self) -> str:
        """Format lessons learned for retry attempts.

        Returns:
            Formatted lessons string
        """
        lessons = self._get_lessons_learned()
        if not lessons:
            return "No previous lessons (first episode).\n\n"

        return (
            "Lessons from previous episodes:\n"
            + "\n".join(f"{i + 1}. {lesson}" for i, lesson in enumerate(lessons))
            + "\n\n"
        )

    def _summarize_memory(self) -> str:
        """Summarize episodic memory for reflection.

        Returns:
            Memory summary string
        """
        if not self._episodic_memory:
            return "No previous episodes to reference.\n\n"

        count = len(self._episodic_memory)
        avg_quality = sum(e["quality"] for e in self._episodic_memory) / count

        return (
            f"Episodic memory: {count} previous episode(s), average quality: {avg_quality:.2f}\n\n"
        )

    async def _sample_attempt(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate attempt using LLM sampling.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled attempt content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_attempt")

        memory_context = self._format_memory_context()
        memory_text = f"\n\nPrevious lessons learned:\n{memory_context}" if memory_context else ""

        system_prompt = """You are a reasoning assistant using the Reflexion methodology.
Generate an attempt at solving the given problem. This will be evaluated and refined through
self-reflection with episodic memory.

Your attempt should:
1. Address the core problem requirements
2. Show your reasoning process
3. Be complete enough to evaluate
4. Apply any lessons from previous attempts (if provided)"""

        user_prompt = f"""Problem: {input_text}

Episode: {self._episode_number}{memory_text}

Generate a thorough attempt at solving this problem:"""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1200,
            )
            content = str(result) if not isinstance(result, str) else result
            step_header = f"Step {self._step_counter}: Attempt (Episode {self._episode_number})"
            return f"{step_header}\n\n{content}"
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_attempt",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_attempt(input_text, context)

    async def _sample_evaluation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """Generate evaluation using LLM sampling.

        Args:
            previous_thought: The attempt to evaluate
            guidance: Optional guidance for the evaluation
            context: Optional additional context

        Returns:
            A tuple of (evaluation content, quality_score)

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_evaluation")

        episode = self._episode_number
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using the Reflexion methodology.
Evaluate the quality and correctness of the previous attempt.

Your evaluation should:
1. Assess correctness of the solution
2. Identify specific issues or errors
3. Evaluate the quality of reasoning
4. Provide a quality score between 0.0 and 1.0

Format your response as:
- Correctness assessment
- Quality assessment
- Issues identified (list)
- Overall quality score (0.0-1.0)"""

        user_prompt = f"""Previous attempt (step {previous_thought.step_number}):
{previous_thought.content}

Evaluate this attempt for episode {episode}.{guidance_text}

Provide a thorough evaluation with a quality score (0.0-1.0):"""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            content_text = str(result) if not isinstance(result, str) else result

            # Try to extract quality score from the response
            quality_score = 0.5  # Default
            import re

            # Look for patterns like "score: 0.75", "quality: 0.8", "0.85/1.0", etc.
            score_patterns = [
                r"score[:\s]+([0-9]*\.?[0-9]+)",
                r"quality[:\s]+([0-9]*\.?[0-9]+)",
                r"([0-9]*\.?[0-9]+)/1\.0",
                r"([0-9]*\.?[0-9]+)\s*/\s*1",
            ]
            for pattern in score_patterns:
                match = re.search(pattern, content_text.lower())
                if match:
                    try:
                        extracted_score = float(match.group(1))
                        if 0.0 <= extracted_score <= 1.0:
                            quality_score = extracted_score
                            break
                    except (ValueError, IndexError):
                        continue

            # Apply learning effect - quality improves with episodes
            base_quality = previous_thought.quality_score or quality_score
            quality_score = min(base_quality + (0.05 * (self._episode_number - 1)), 0.95)

            content = (
                f"Step {self._step_counter}: Evaluation (Episode {episode})\n\n"
                f"{content_text}\n\n"
                f"Overall Score: {quality_score:.2f}/1.00"
            )

            return content, quality_score
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_evaluation",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            content = self._generate_evaluation(previous_thought, guidance, context)
            base_quality = previous_thought.quality_score or 0.5
            quality_score = min(base_quality + (0.05 * (self._episode_number - 1)), 0.95)
            return content, quality_score

    async def _sample_reflection(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate reflection using LLM sampling.

        Args:
            previous_thought: The evaluation to reflect on
            guidance: Optional guidance for the reflection
            context: Optional additional context

        Returns:
            A tuple of (reflection content, lesson learned)

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_reflection")

        episode = self._episode_number
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        memory_summary = self._summarize_memory()

        system_prompt = """You are a reasoning assistant using the Reflexion methodology.
Reflect on the evaluation to extract actionable lessons for improvement.

Your reflection should:
1. Analyze what went wrong and why
2. Extract key insights from the failures
3. Formulate a specific, actionable lesson
4. Consider episodic memory from previous attempts

End with a clear "Lesson learned:" statement."""

        user_prompt = f"""Previous evaluation (step {previous_thought.step_number}):
{previous_thought.content}

{memory_summary}Reflect on this evaluation for episode {episode}.{guidance_text}

What went wrong, what insights can be gained, and what lesson should be learned?"""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            content_text = str(result) if not isinstance(result, str) else result

            # Try to extract the lesson learned
            lesson = f"Episode {episode}: Insights from reflection"
            if "lesson learned:" in content_text.lower():
                parts = content_text.lower().split("lesson learned:")
                if len(parts) > 1:
                    lesson_text = parts[1].strip().split("\n")[0].strip()
                    if lesson_text:
                        lesson = f"Episode {episode}: {lesson_text}"

            content = (
                f"Step {self._step_counter}: Reflection (Episode {episode})\n\n"
                f"{content_text}\n\n"
                f"This lesson will be stored in episodic memory."
            )

            return content, lesson
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_reflection",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            content = self._generate_reflection(previous_thought, guidance, context)
            quality = previous_thought.quality_score or 0.5
            lesson = self._extract_lesson(previous_thought, quality)
            return content, lesson

    async def _sample_retry(
        self,
        original_input: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate retry attempt using LLM sampling.

        Args:
            original_input: The original problem/question
            previous_thought: The reflection to apply
            guidance: Optional guidance for the retry
            context: Optional additional context

        Returns:
            The sampled retry content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_retry")

        episode = self._episode_number
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        lessons = self._format_lessons_learned()

        system_prompt = """You are a reasoning assistant using the Reflexion methodology.
Generate an improved attempt that incorporates all lessons learned from episodic memory.

Your retry should:
1. Apply all lessons from previous episodes
2. Avoid previous mistakes
3. Show clear improvements over earlier attempts
4. Demonstrate learning from failures"""

        user_prompt = f"""Original problem: {original_input}

Previous reflection (step {previous_thought.step_number}):
{previous_thought.content}

{lessons}Generate a retry attempt for episode {episode} that applies all learned lessons.
{guidance_text}

Show how you're incorporating the lessons into an improved solution:"""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1200,
            )
            content_text = str(result) if not isinstance(result, str) else result

            quality_threshold = previous_thought.metadata.get("quality_threshold", 0.85)
            content = (
                f"Step {self._step_counter}: Retry (Episode {episode})\n\n"
                f"{content_text}\n\n"
                f"Lessons applied from {len(self._episodic_memory)} previous episode(s)\n"
                f"Target quality: {quality_threshold:.2f}/1.00"
            )

            return content
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_retry",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_retry(previous_thought, guidance, context)
