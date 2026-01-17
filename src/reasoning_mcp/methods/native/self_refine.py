"""Self-Refine reasoning method.

This module implements an iterative self-improvement reasoning method based on
the Self-Refine approach (Madaan et al. 2023). The method generates an initial
output, then iteratively provides feedback and refines until satisfied or max
iterations reached.

The key difference from Self-Reflection:
- Self-Refine: Simple Generate → Feedback → Refine loop
- Self-Reflection: Has Initial → Critique → Improve with quality scores

Self-Refine focuses on concrete feedback items to address in each refinement cycle.
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


# Metadata for Self-Refine method
SELF_REFINE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_REFINE,
    name="Self-Refine",
    description="Iterative self-improvement through generate-feedback-refine cycles. "
    "Generates initial output, provides concrete feedback, then refines based on "
    "feedback items until satisfied or max iterations reached.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "iterative",
            "self-improvement",
            "feedback",
            "refinement",
            "incremental",
            "quality-improvement",
            "self-evaluation",
        }
    ),
    complexity=4,  # Medium complexity - iterative refinement
    supports_branching=False,  # Linear refinement path
    supports_revision=True,  # Core feature - refining through feedback
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: generate + feedback + refine
    max_thoughts=12,  # Max 4 iterations × 3 thoughts per iteration
    avg_tokens_per_thought=350,  # Moderate - includes feedback items
    best_for=(
        "output quality improvement",
        "iterative refinement",
        "self-correction",
        "quality-sensitive tasks",
        "creative writing",
        "code refinement",
        "answer improvement",
    ),
    not_recommended_for=(
        "simple factual queries",
        "time-critical decisions",
        "tasks requiring external validation",
        "fixed-format outputs",
    ),
)

logger = structlog.get_logger(__name__)


class SelfRefine(ReasoningMethodBase):
    """Self-Refine reasoning method implementation.

    This class implements an iterative self-improvement pattern where the system
    generates an initial output, then iteratively provides feedback and refines
    until satisfied or maximum iterations reached. Each cycle involves:
    1. Generating/reviewing current output
    2. Providing concrete feedback items
    3. Refining based on feedback
    4. Repeating until satisfied or max iterations

    Key characteristics:
    - Simpler than Self-Reflection
    - Concrete feedback items to address
    - Iterative refinement
    - No quality scoring (unlike Self-Reflection)
    - Revision-based improvement
    - Medium complexity (4)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SelfRefine()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Write an introduction to machine learning"
        ... )
        >>> print(result.content)  # Initial generation

        Continue with feedback:
        >>> feedback = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Provide feedback"
        ... )
        >>> print(feedback.type)  # ThoughtType.VERIFICATION (feedback phase)

        Continue with refinement:
        >>> refined = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=feedback,
        ...     guidance="Apply refinements"
        ... )
        >>> print(refined.type)  # ThoughtType.REVISION (refine phase)
    """

    # Maximum refinement iterations to prevent infinite loops
    MAX_ITERATIONS = 4

    # Enable LLM sampling for generating outputs and feedback
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Self-Refine method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase: str = "generate"  # generate, feedback, refine
        self.enable_elicitation = enable_elicitation
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SELF_REFINE

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SELF_REFINE_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SELF_REFINE_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Self-Refine method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = SelfRefine()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._iteration_count == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase = "generate"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Self-Refine method.

        This method creates the initial generation that will be iteratively
        refined through feedback cycles. It generates a first attempt at
        producing the desired output.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include max_iterations)
            execution_context: Optional ExecutionContext for LLM sampling and elicitation

        Returns:
            A ThoughtNode representing the initial generation

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfRefine()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Write a clear explanation of recursion"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SELF_REFINE
            >>> assert "iteration_count" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("Self-Refine method must be initialized before execution")

        # Store execution context for LLM sampling and elicitation
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._iteration_count = 0
        self._current_phase = "generate"

        # Extract max iterations from context if provided
        max_iterations = self.MAX_ITERATIONS
        if context and "max_iterations" in context:
            max_iterations = max(1, min(context["max_iterations"], 10))

        # Generate initial output using LLM sampling with fallback
        content = await self._sample_initial_output(input_text, context)

        # Initial confidence - moderate (will improve with refinement)
        initial_confidence = 0.6

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_REFINE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=initial_confidence,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "self_refine",
                "phase": self._current_phase,
                "iteration_count": self._iteration_count,
                "max_iterations": max_iterations,
                "feedback_items": [],  # No feedback yet
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SELF_REFINE

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

        This method implements the refinement cycle logic:
        - If previous was generate/refine: generate feedback
        - If previous was feedback: generate refinement
        - Continues until max iterations reached

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the self-refine process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfRefine()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Write about AI")
            >>> feedback = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert feedback.type == ThoughtType.VERIFICATION
            >>> assert feedback.metadata["phase"] == "feedback"
            >>>
            >>> refinement = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=feedback
            ... )
            >>> assert refinement.type == ThoughtType.REVISION
            >>> assert refinement.metadata["phase"] == "refine"
        """
        if not self._initialized:
            raise RuntimeError("Self-Refine method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Get max iterations from previous thought's metadata
        max_iterations = previous_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "generate")

        # Optional elicitation: ask user for refinement preferences
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
            and prev_phase in ("generate", "refine")  # Only elicit during feedback generation
        ):
            try:
                # Ask user what aspect to focus on during refinement
                options = [
                    {
                        "id": "clarity",
                        "label": "Clarity - Make explanations clearer and easier to understand",
                    },
                    {
                        "id": "accuracy",
                        "label": "Accuracy - Improve factual correctness and precision",
                    },
                    {
                        "id": "completeness",
                        "label": "Completeness - Add missing information and details",
                    },
                    {
                        "id": "conciseness",
                        "label": "Conciseness - Make output more concise and to-the-point",
                    },
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                iteration_msg = (
                    f"Self-Refine is generating feedback for iteration "
                    f"{self._iteration_count + 1}. What aspect should refinement focus on?"
                )
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    iteration_msg,
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    # Store the selection in context for use in feedback generation
                    if context is None:
                        context = {}
                    context["refinement_focus"] = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_expected_error",
                    method="continue_reasoning",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Elicitation failed or timed out - continue without it
                pass
            except Exception as e:
                # Log unexpected exceptions and re-raise to avoid masking bugs
                logger.error(
                    "elicitation_unexpected_error",
                    method="continue_reasoning",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        if prev_phase in ("generate", "refine"):
            # Next: feedback
            self._current_phase = "feedback"
            thought_type = ThoughtType.VERIFICATION

            # Generate feedback using LLM sampling with fallback
            content, feedback_items = await self._sample_feedback(
                previous_thought, guidance, context
            )

            # Confidence moderate during feedback
            confidence = 0.7

        elif prev_phase == "feedback":
            # Next: refinement
            self._current_phase = "refine"
            # Only increment if not exceeding max_iterations
            self._iteration_count = min(self._iteration_count + 1, max_iterations)
            thought_type = ThoughtType.REVISION

            # Get feedback items from previous thought
            feedback_items = previous_thought.metadata.get("feedback_items", [])

            # Generate refinement using LLM sampling with fallback
            content, addressed_count = await self._sample_refinement(
                previous_thought, feedback_items, guidance, context
            )

            # Confidence increases with each refinement iteration
            confidence = min(0.6 + (0.1 * self._iteration_count), 0.95)

        else:
            # Fallback to feedback
            self._current_phase = "feedback"
            thought_type = ThoughtType.VERIFICATION

            # Generate feedback using LLM sampling with fallback
            content, feedback_items = await self._sample_feedback(
                previous_thought, guidance, context
            )
            confidence = 0.7

        # Check if we should continue or conclude
        should_continue = self._iteration_count < max_iterations

        # If this is a refinement that has reached max iterations, mark as conclusion
        if self._current_phase == "refine" and not should_continue:
            thought_type = ThoughtType.CONCLUSION

        # Build metadata
        metadata: dict[str, Any] = {
            "phase": self._current_phase,
            "iteration_count": self._iteration_count,
            "max_iterations": max_iterations,
            "should_continue": should_continue,
            "guidance": guidance or "",
            "context": context or {},
            "reasoning_type": "self_refine",
        }

        # Add phase-specific metadata
        if self._current_phase == "feedback":
            metadata["feedback_items"] = feedback_items
            metadata["feedback_count"] = len(feedback_items)
        elif self._current_phase == "refine":
            metadata["feedback_items"] = feedback_items
            metadata["addressed_items"] = addressed_count
            metadata["refinement_iteration"] = self._iteration_count

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SELF_REFINE,
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

        For Self-Refine, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SelfRefine()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_output(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial output to be refined.

        This is a helper method that would typically call an LLM to generate
        the initial attempt at producing the desired output.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial output

        Note:
            In a full implementation, this would use an LLM to generate
            the actual output. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: Initial Generation (Iteration 0)\n\n"
            f"Task: {input_text}\n\n"
            f"I will generate an initial output for this task. "
            f"This will then be iteratively refined through feedback cycles "
            f"to improve quality and address any issues.\n\n"
            f"Initial output:\n"
            f"[This would contain the LLM-generated initial output]\n\n"
            f"Note: This is the first version. I will now receive feedback "
            f"to identify areas for improvement."
        )

    def _generate_feedback(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate feedback on the previous output.

        This is a helper method that would typically call an LLM to analyze
        the previous output and identify concrete feedback items to address.

        Args:
            previous_thought: The output to provide feedback on
            guidance: Optional guidance for the feedback
            context: Optional additional context

        Returns:
            A tuple of (feedback content, list of feedback items)

        Note:
            In a full implementation, this would use an LLM to generate
            the actual feedback. This is a placeholder that provides
            the structure.
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Example feedback items (would be LLM-generated)
        feedback_items = [
            "Add more concrete examples",
            "Clarify technical terminology",
            "Improve flow and transitions",
            "Add conclusion summary",
        ]

        content = (
            f"Step {self._step_counter}: Feedback Generation (Iteration {iteration})\n\n"
            f"Evaluating output from Step {previous_thought.step_number}...\n\n"
            f"Feedback items to address:\n"
        )

        for i, item in enumerate(feedback_items, 1):
            content += f"{i}. {item}\n"

        content += (
            f"\nTotal feedback items: {len(feedback_items)}\n"
            f"These items should be addressed in the next refinement.{guidance_text}"
        )

        return content, feedback_items

    def _generate_refinement(
        self,
        feedback_thought: ThoughtNode,
        feedback_items: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Generate a refined output based on feedback.

        This is a helper method that would typically call an LLM to generate
        an improved version addressing the feedback items.

        Args:
            feedback_thought: The feedback to address
            feedback_items: List of concrete feedback items
            guidance: Optional guidance for the refinement
            context: Optional additional context

        Returns:
            A tuple of (refined content, number of items addressed)

        Note:
            In a full implementation, this would use an LLM to generate
            the actual refined output. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        addressed_count = len(feedback_items)  # In practice, might not address all

        content = (
            f"Step {self._step_counter}: Refined Output (Iteration {self._iteration_count})\n\n"
            f"Based on feedback from Step {feedback_thought.step_number}, "
            f"I will now produce a refined version addressing the identified issues.\n\n"
            f"Addressing {addressed_count}/{len(feedback_items)} feedback items:\n"
        )

        for i, item in enumerate(feedback_items, 1):
            content += f"{i}. {item} - [addressed]\n"

        max_iter = feedback_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)
        content += (
            f"\nRefined output:\n"
            f"[This would contain the LLM-generated refined output]\n\n"
            f"Improvements applied: {addressed_count} items addressed\n"
            f"Iteration: {self._iteration_count}/{max_iter}{guidance_text}"
        )

        return content, addressed_count

    async def _sample_initial_output(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial output using LLM sampling with fallback.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled initial output content
        """
        system_prompt = """You are a reasoning assistant using the Self-Refine methodology.
Generate an initial output for the given task. This will be iteratively refined based on feedback.

Your initial output should:
1. Address the core task requirements
2. Be complete enough to evaluate
3. Have room for improvement through refinement
4. Show your reasoning process"""

        user_prompt = f"""Task: {input_text}

Generate an initial output for this task. Remember, this is iteration 0 and will be
refined based on feedback."""

        def fallback() -> str:
            return self._generate_initial_output(input_text, context)

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Check if we got a sampled result (not the fallback)
        if "Step " not in result or "[This would contain" not in result:
            return f"Step {self._step_counter}: Initial Generation (Iteration 0)\n\n{result}"
        return result

    async def _sample_feedback(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate feedback using LLM sampling with fallback.

        Args:
            previous_thought: The output to provide feedback on
            guidance: Optional guidance for the feedback
            context: Optional additional context (may include refinement_focus)

        Returns:
            A tuple of (feedback content, list of feedback items)
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        focus_text = ""
        if context and "refinement_focus" in context:
            focus = context["refinement_focus"]
            focus_text = f"\n\nUser has requested focus on: {focus}"

        system_prompt = """You are a reasoning assistant using the Self-Refine methodology.
Provide concrete, actionable feedback on the previous output.

Your feedback should:
1. Identify specific areas for improvement
2. Be concrete and actionable
3. Focus on the most impactful changes
4. List 3-5 distinct feedback items

Format your response as a numbered list of feedback items."""

        user_prompt = f"""Previous output (step {previous_thought.step_number}):
{previous_thought.content}

Provide feedback for refinement iteration {iteration + 1}.{focus_text}{guidance_text}

List 3-5 concrete, actionable feedback items:"""

        # Use a sentinel to detect fallback
        fallback_sentinel = "__FALLBACK_USED__"

        def fallback() -> str:
            return fallback_sentinel

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # If fallback was triggered, use the heuristic generator
        if result == fallback_sentinel:
            return self._generate_feedback(previous_thought, guidance, context)

        content_text = result

        # Parse feedback items from numbered list
        feedback_items = []
        for line in content_text.split("\n"):
            line = line.strip()
            # Match numbered items like "1. ", "2.", "1)", etc.
            if line and (
                line[0].isdigit()
                or (len(line) > 2 and line[0:2].replace(".", "").replace(")", "").isdigit())
            ):
                # Remove the number prefix
                item = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                if item:
                    feedback_items.append(item)

        # Fallback if no items parsed
        if not feedback_items:
            feedback_items = [
                "Add more concrete examples",
                "Clarify technical terminology",
                "Improve flow and transitions",
            ]

        content = (
            f"Step {self._step_counter}: Feedback Generation (Iteration {iteration})\n\n"
            f"{content_text}\n\n"
            f"Total feedback items: {len(feedback_items)}"
        )

        return content, feedback_items

    async def _sample_refinement(
        self,
        feedback_thought: ThoughtNode,
        feedback_items: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, int]:
        """Generate refinement using LLM sampling with fallback.

        Args:
            feedback_thought: The feedback to address
            feedback_items: List of concrete feedback items
            guidance: Optional guidance for the refinement
            context: Optional additional context

        Returns:
            A tuple of (refined content, number of items addressed)
        """
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        # Format feedback items as a numbered list
        feedback_list = "\n".join(f"{i}. {item}" for i, item in enumerate(feedback_items, 1))

        system_prompt = """You are a reasoning assistant using the Self-Refine methodology.
Generate a refined output that addresses the feedback items.

Your refined output should:
1. Incorporate all feedback items where possible
2. Improve upon the previous version
3. Maintain the original intent and correctness
4. Show clear improvements in quality"""

        user_prompt = f"""Previous feedback (step {feedback_thought.step_number}):
{feedback_thought.content}

Feedback items to address:
{feedback_list}

Generate a refined output for iteration {self._iteration_count} that addresses these
feedback items.{guidance_text}"""

        def fallback() -> str:
            content, _ = self._generate_refinement(
                feedback_thought, feedback_items, guidance, context
            )
            return content

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        content_text = result

        # Assume all items were addressed (in practice, might analyze the output)
        addressed_count = len(feedback_items)

        content = (
            f"Step {self._step_counter}: Refined Output (Iteration {self._iteration_count})\n\n"
            f"{content_text}\n\n"
            f"Improvements applied: {addressed_count}/{len(feedback_items)} items addressed"
        )

        return content, addressed_count
