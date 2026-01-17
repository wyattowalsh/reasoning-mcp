"""Socratic Self-Refine (SSR) reasoning method.

This module implements SSR (Socratic Self-Refine), a method that combines Socratic
questioning with iterative self-refinement. Unlike standard self-refine which uses
general feedback, SSR uses probing Socratic questions to drive deeper improvements.

Reference: 2025
Key Idea: Socratic questioning to drive iterative self-refinement

The method progresses through cycles of:
1. Generate - Initial answer generation
2. Socratic Question - Ask probing questions (Why? What if? How do we know? etc.)
3. Refine - Improve answer based on question responses
4. Validate - Check if refinement satisfies the questions
5. Conclude - Final answer after sufficient refinement

Each cycle uses Socratic questioning to challenge assumptions, expose contradictions,
and guide toward deeper understanding and better outputs.
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


# Metadata for SSR method
SSR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SSR,
    name="Socratic Self-Refine",
    description="Socratic questioning to drive iterative self-refinement. "
    "Uses probing questions (Why? What if? How do we know? What are the implications?) "
    "to challenge assumptions and guide deeper improvements through multiple refinement cycles.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "socratic",
            "self-refine",
            "iterative",
            "questioning",
            "critical-thinking",
            "refinement",
            "self-improvement",
            "assumptions",
            "validation",
        }
    ),
    complexity=5,  # Medium-high complexity - combines two techniques
    supports_branching=False,  # Linear refinement path
    supports_revision=True,  # Core feature - refining through questions
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: generate + question + refine + validate + conclude
    max_thoughts=20,  # Max 4 iterations × 5 thoughts per iteration
    avg_tokens_per_thought=400,  # Higher - includes questions and reasoning
    best_for=(
        "output quality improvement",
        "critical thinking development",
        "assumption challenging",
        "iterative refinement",
        "self-correction",
        "deep understanding",
        "quality-sensitive tasks",
        "complex problem solving",
    ),
    not_recommended_for=(
        "simple factual queries",
        "time-critical decisions",
        "routine problem solving",
        "when direct answers are needed",
        "simple calculations",
    ),
)


class SSR(ReasoningMethodBase):
    """Socratic Self-Refine (SSR) reasoning method implementation.

    This class implements an iterative self-improvement pattern where Socratic
    questioning drives the refinement process. Each cycle involves:
    1. Generating/reviewing current output
    2. Asking probing Socratic questions
    3. Refining based on question responses
    4. Validating that questions have been addressed
    5. Repeating until satisfied or max iterations

    Key characteristics:
    - Socratic question-driven refinement
    - Deeper than standard self-refine
    - Challenges assumptions systematically
    - Multiple question types (Why, What if, How, Implications)
    - Track questions asked and improvements made
    - Medium-high complexity (5)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SSR()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain why democracy is important"
        ... )
        >>> print(result.content)  # Initial generation

        Continue with Socratic questions:
        >>> questions = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Ask probing questions"
        ... )
        >>> print(questions.type)  # ThoughtType.HYPOTHESIS (question phase)

        Continue with refinement:
        >>> refined = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=questions,
        ...     guidance="Apply refinements"
        ... )
        >>> print(refined.type)  # ThoughtType.REVISION (refine phase)
    """

    # Maximum refinement iterations to prevent infinite loops
    MAX_ITERATIONS = 4

    # Socratic question types
    QUESTION_WHY = "why"  # Challenge reasoning and assumptions
    QUESTION_WHAT_IF = "what_if"  # Explore alternatives and edge cases
    QUESTION_HOW = "how"  # Verify mechanisms and evidence
    QUESTION_IMPLICATIONS = "implications"  # Examine consequences

    # Enable LLM sampling for generating outputs, questions, and refinements
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the SSR method."""
        self._initialized = False
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase: str = "generate"  # generate, question, refine, validate
        self._questions_asked: list[dict[str, str]] = []
        self._improvements_made: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SSR

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SSR_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SSR_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the SSR method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = SSR()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._iteration_count == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._iteration_count = 0
        self._current_phase = "generate"
        self._questions_asked = []
        self._improvements_made = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the SSR method.

        This method creates the initial generation that will be iteratively
        refined through Socratic questioning cycles. It generates a first
        attempt at producing the desired output.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include max_iterations)
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the initial generation

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SSR()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Write about the importance of critical thinking"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SSR
            >>> assert "iteration_count" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("SSR method must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._iteration_count = 0
        self._current_phase = "generate"
        self._questions_asked = []
        self._improvements_made = []

        # Extract max iterations from context if provided
        max_iterations = self.MAX_ITERATIONS
        if context and "max_iterations" in context:
            max_iterations = max(1, min(context["max_iterations"], 10))

        # Generate initial output using LLM sampling if available
        content = await self._sample_initial_output(input_text, context)

        # Initial confidence - moderate (will improve with refinement)
        initial_confidence = 0.6

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SSR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=initial_confidence,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "socratic_self_refine",
                "phase": self._current_phase,
                "iteration_count": self._iteration_count,
                "max_iterations": max_iterations,
                "questions_asked": [],
                "improvements_made": [],
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SSR

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

        This method implements the SSR refinement cycle logic:
        - If previous was generate/refine: generate Socratic questions
        - If previous was questions: generate refinement addressing questions
        - If previous was refine: validate improvements
        - If previous was validate: either conclude or ask new questions
        - Continues until max iterations reached

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the SSR process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SSR()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Explain AI ethics")
            >>> questions = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert questions.type == ThoughtType.HYPOTHESIS
            >>> assert questions.metadata["phase"] == "socratic_question"
            >>>
            >>> refinement = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=questions
            ... )
            >>> assert refinement.type == ThoughtType.REVISION
            >>> assert refinement.metadata["phase"] == "refine"
        """
        if not self._initialized:
            raise RuntimeError("SSR method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Get max iterations from previous thought's metadata
        max_iterations = previous_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            # Next: Socratic questions
            self._current_phase = "socratic_question"
            thought_type = ThoughtType.HYPOTHESIS

            # Generate Socratic questions using LLM sampling if available
            content, questions = await self._sample_socratic_questions(
                previous_thought, guidance, context
            )
            confidence = 0.7

        elif prev_phase == "socratic_question":
            # Next: refinement
            self._current_phase = "refine"
            thought_type = ThoughtType.REVISION

            # Get questions from previous thought
            questions_data = previous_thought.metadata.get("questions", [])

            # Generate refinement using LLM sampling if available
            content, improvements = await self._sample_refinement(
                previous_thought, questions_data, guidance, context
            )

            # Confidence increases with each refinement iteration
            confidence = min(0.6 + (0.1 * self._iteration_count), 0.95)

        elif prev_phase == "refine":
            # Next: validation
            self._current_phase = "validate"
            thought_type = ThoughtType.VERIFICATION

            # Validate refinement using LLM sampling if available
            content, validation_passed = await self._sample_validation(
                previous_thought, guidance, context
            )
            confidence = 0.8

        elif prev_phase == "validate":
            # Next: either conclude or continue with new questions
            # Note: iteration_count was already incremented during refine phase
            if self._iteration_count >= max_iterations:
                # Conclude
                self._current_phase = "conclude"
                thought_type = ThoughtType.CONCLUSION

                # Generate conclusion using LLM sampling if available
                content = await self._sample_conclusion(previous_thought, guidance, context)
                confidence = 0.9
            else:
                # Continue with new Socratic questions
                self._current_phase = "socratic_question"
                thought_type = ThoughtType.HYPOTHESIS

                # Generate Socratic questions using LLM sampling if available
                content, questions = await self._sample_socratic_questions(
                    previous_thought, guidance, context
                )
                confidence = 0.75

        else:
            # Fallback to Socratic questions
            self._current_phase = "socratic_question"
            thought_type = ThoughtType.HYPOTHESIS

            # Generate Socratic questions using LLM sampling if available
            content, questions = await self._sample_socratic_questions(
                previous_thought, guidance, context
            )
            confidence = 0.7

        # Check if we should continue or conclude
        should_continue = self._iteration_count < max_iterations

        # Build metadata
        metadata: dict[str, Any] = {
            "phase": self._current_phase,
            "iteration_count": self._iteration_count,
            "max_iterations": max_iterations,
            "should_continue": should_continue,
            "guidance": guidance or "",
            "context": context or {},
            "reasoning_type": "socratic_self_refine",
            "questions_asked": len(self._questions_asked),
            "improvements_made": len(self._improvements_made),
        }

        # Add phase-specific metadata
        if self._current_phase == "socratic_question":
            metadata["questions"] = questions
            metadata["question_count"] = len(questions)
        elif self._current_phase == "refine":
            metadata["improvements"] = improvements
            metadata["improvement_count"] = len(improvements)
            metadata["refinement_iteration"] = self._iteration_count
        elif self._current_phase == "validate":
            metadata["validation_passed"] = validation_passed

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SSR,
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

        For SSR, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SSR()
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
            f"I will generate an initial response. This will then be refined through "
            f"Socratic questioning cycles - using probing questions to challenge "
            f"assumptions, expose weaknesses, and guide toward deeper understanding.\n\n"
            f"Initial output:\n"
            f"[This would contain the LLM-generated initial output]\n\n"
            f"Note: This is the first version. I will now receive Socratic questions "
            f"to identify areas requiring deeper thought and improvement."
        )

    def _generate_socratic_questions(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Generate Socratic questions to probe the previous output.

        This is a helper method that would typically call an LLM to generate
        probing Socratic questions that challenge assumptions and guide improvement.

        Args:
            previous_thought: The output to question
            guidance: Optional guidance for the questions
            context: Optional additional context

        Returns:
            A tuple of (question content, list of question dictionaries)

        Note:
            In a full implementation, this would use an LLM to generate
            contextually appropriate Socratic questions. This is a placeholder
            that provides the structure.
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Generate different types of Socratic questions
        questions = [
            {
                "type": self.QUESTION_WHY,
                "question": "Why is this reasoning valid? What assumptions underlie it?",
            },
            {
                "type": self.QUESTION_WHAT_IF,
                "question": "What if these assumptions are false? Are there counterexamples?",
            },
            {
                "type": self.QUESTION_HOW,
                "question": "How do we know this is true? What evidence supports it?",
            },
            {
                "type": self.QUESTION_IMPLICATIONS,
                "question": "What are the implications if we accept this? What follows from it?",
            },
        ]

        # Track questions
        self._questions_asked.extend(questions)

        content = (
            f"Step {self._step_counter}: Socratic Questioning (Iteration {iteration})\n\n"
            f"Examining output from Step {previous_thought.step_number} through "
            f"probing questions...\n\n"
            f"Socratic Questions:\n\n"
        )

        for i, q in enumerate(questions, 1):
            content += f"{i}. [{q['type'].upper()}] {q['question']}\n"

        content += (
            f"\nTotal questions: {len(questions)}\n"
            f"These questions challenge assumptions, explore alternatives, verify "
            f"evidence, and examine implications.{guidance_text}\n\n"
            f"The next step will refine the output to address these questions."
        )

        return content, questions

    def _generate_refinement(
        self,
        question_thought: ThoughtNode,
        questions: list[dict[str, str]],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate a refined output based on Socratic questions.

        This is a helper method that would typically call an LLM to generate
        an improved version that addresses the Socratic questions.

        Args:
            question_thought: The Socratic questions to address
            questions: List of question dictionaries
            guidance: Optional guidance for the refinement
            context: Optional additional context

        Returns:
            A tuple of (refined content, list of improvements made)

        Note:
            In a full implementation, this would use an LLM to generate
            the actual refined output. This is a placeholder that provides
            the structure.
        """
        self._iteration_count = min(self._iteration_count + 1, self.MAX_ITERATIONS)
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Example improvements (would be LLM-generated)
        improvements = [
            "Clarified underlying assumptions and their validity",
            "Addressed counterexamples and alternative perspectives",
            "Added evidence and justification for claims",
            "Explored implications and consequences",
        ]

        # Track improvements
        self._improvements_made.extend(improvements)

        content = (
            f"Step {self._step_counter}: Refined Output (Iteration {self._iteration_count})\n\n"
            f"Based on Socratic questions from Step {question_thought.step_number}, "
            f"I will now produce a refined version.\n\n"
            f"Addressing {len(questions)} Socratic questions:\n"
        )

        for i, q in enumerate(questions, 1):
            content += f"{i}. {q.get('type', 'unknown').upper()}: {q.get('question', '')}\n"
            content += f"   → Improvement: {improvements[min(i - 1, len(improvements) - 1)]}\n\n"

        max_iter = question_thought.metadata.get("max_iterations", self.MAX_ITERATIONS)
        content += (
            f"Refined output:\n"
            f"[This would contain the LLM-generated refined output addressing "
            f"all Socratic questions]\n\n"
            f"Improvements applied: {len(improvements)} areas addressed\n"
            f"Iteration: {self._iteration_count}/{max_iter}{guidance_text}"
        )

        return content, improvements

    def _validate_refinement(
        self,
        refinement_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, bool]:
        """Validate that the refinement adequately addressed the questions.

        This is a helper method that would typically call an LLM to verify
        that the Socratic questions have been satisfactorily addressed.

        Args:
            refinement_thought: The refined output to validate
            guidance: Optional guidance for validation
            context: Optional additional context

        Returns:
            A tuple of (validation content, whether validation passed)

        Note:
            In a full implementation, this would use an LLM to perform
            actual validation. This is a placeholder that provides
            the structure.
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Simulate validation (would be LLM-generated)
        validation_passed = iteration >= 2  # Pass after 2+ iterations

        check = "✓" if validation_passed else "✗"
        result = "PASSED" if validation_passed else "NEEDS MORE REFINEMENT"
        content = (
            f"Step {self._step_counter}: Validation (Iteration {iteration})\n\n"
            f"Validating refinement from Step {refinement_thought.step_number}...\n\n"
            f"Checking if Socratic questions have been adequately addressed:\n"
            f"- WHY questions: Assumptions clarified? {check}\n"
            f"- WHAT IF questions: Alternatives explored? {check}\n"
            f"- HOW questions: Evidence provided? {check}\n"
            f"- IMPLICATIONS questions: Consequences examined? {check}\n\n"
            f"Validation result: {result}\n"
            f"Total questions addressed: {len(self._questions_asked)}\n"
            f"Total improvements made: {len(self._improvements_made)}{guidance_text}"
        )

        return content, validation_passed

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion after all refinement cycles.

        This is a helper method that would typically call an LLM to synthesize
        the final answer incorporating all improvements.

        Args:
            previous_thought: The previous thought to build on
            guidance: Optional guidance for the conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion

        Note:
            In a full implementation, this would use an LLM to generate
            the actual conclusion. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Final Conclusion\n\n"
            f"After {self._iteration_count} iterations of Socratic self-refinement, "
            f"here is the final answer:\n\n"
            f"[This would contain the LLM-generated final answer incorporating "
            f"all insights from the Socratic questioning and refinement process]\n\n"
            f"Summary:\n"
            f"- Total Socratic questions asked: {len(self._questions_asked)}\n"
            f"- Total improvements made: {len(self._improvements_made)}\n"
            f"- Refinement iterations: {self._iteration_count}\n\n"
            f"Through systematic Socratic questioning and iterative refinement, "
            f"we have challenged assumptions, explored alternatives, verified evidence, "
            f"and examined implications to produce a deeper, more robust answer.{guidance_text}"
        )

    async def _sample_initial_output(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial output using LLM sampling.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The sampled initial output content
        """
        system_prompt = (
            "You are a reasoning assistant using the Socratic Self-Refine (SSR) methodology.\n"
            "Generate an initial output for the given task. This will be iteratively refined "
            "through Socratic questioning cycles.\n\n"
            "Your initial output should:\n"
            "1. Address the core task requirements\n"
            "2. Be complete enough to evaluate\n"
            "3. Have room for improvement through Socratic questioning\n"
            "4. Show your reasoning process"
        )

        user_prompt = (
            f"Task: {input_text}\n\n"
            f"Generate an initial output for this task. Remember, this is iteration 0 and will "
            f"be refined based on Socratic questions that challenge assumptions, explore "
            f"alternatives, verify evidence, and examine implications."
        )

        step_counter = self._step_counter  # Capture for closure

        def fallback() -> str:
            return self._generate_initial_output(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Add step prefix if content doesn't already have it (i.e., it was sampled)
        if not content.startswith(f"Step {step_counter}"):
            return f"Step {step_counter}: Initial Generation (Iteration 0)\n\n{content}"
        return content

    async def _sample_socratic_questions(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Generate Socratic questions using LLM sampling.

        Args:
            previous_thought: The output to question
            guidance: Optional guidance for the questions
            context: Optional additional context

        Returns:
            A tuple of (question content, list of question dictionaries)
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = (
            "You are a reasoning assistant using the Socratic Self-Refine (SSR) methodology.\n"
            "Generate probing Socratic questions to challenge and improve the previous output.\n\n"
            "Your questions should:\n"
            "1. Challenge underlying assumptions (WHY questions)\n"
            "2. Explore alternatives and edge cases (WHAT IF questions)\n"
            "3. Verify mechanisms and evidence (HOW questions)\n"
            "4. Examine consequences and implications (IMPLICATIONS questions)\n\n"
            "Generate 4 distinct Socratic questions, one for each type."
        )

        user_prompt = (
            f"Previous output (step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Generate Socratic questions for refinement iteration {iteration}.{guidance_text}\n\n"
            f"Format your response as:\n"
            f"WHY: [your why question]\n"
            f"WHAT IF: [your what if question]\n"
            f"HOW: [your how question]\n"
            f"IMPLICATIONS: [your implications question]"
        )

        def fallback() -> str:
            # Return marker that indicates fallback was used
            return "__FALLBACK__"

        content_text = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # If fallback was used, delegate to the heuristic method
        if content_text == "__FALLBACK__":
            return self._generate_socratic_questions(previous_thought, guidance, context)

        # Parse questions from the response
        questions: list[dict[str, str]] = []
        lines = content_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHY:"):
                questions.append({"type": self.QUESTION_WHY, "question": line[4:].strip()})
            elif line.upper().startswith("WHAT IF:"):
                questions.append({"type": self.QUESTION_WHAT_IF, "question": line[8:].strip()})
            elif line.upper().startswith("HOW:"):
                questions.append({"type": self.QUESTION_HOW, "question": line[4:].strip()})
            elif line.upper().startswith("IMPLICATIONS:"):
                questions.append(
                    {"type": self.QUESTION_IMPLICATIONS, "question": line[13:].strip()}
                )

        # Fallback if parsing failed
        if not questions:
            questions = [
                {
                    "type": self.QUESTION_WHY,
                    "question": "Why is this reasoning valid? What assumptions underlie it?",
                },
                {
                    "type": self.QUESTION_WHAT_IF,
                    "question": ("What if these assumptions are false? Are there counterexamples?"),
                },
                {
                    "type": self.QUESTION_HOW,
                    "question": "How do we know this is true? What evidence supports it?",
                },
                {
                    "type": self.QUESTION_IMPLICATIONS,
                    "question": (
                        "What are the implications if we accept this? What follows from it?"
                    ),
                },
            ]

        # Track questions
        self._questions_asked.extend(questions)

        content = (
            f"Step {self._step_counter}: Socratic Questioning (Iteration {iteration})\n\n"
            f"{content_text}\n\n"
            f"Total questions: {len(questions)}"
        )

        return content, questions

    async def _sample_refinement(
        self,
        question_thought: ThoughtNode,
        questions: list[dict[str, str]],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate refinement using LLM sampling.

        Args:
            question_thought: The Socratic questions to address
            questions: List of question dictionaries
            guidance: Optional guidance for the refinement
            context: Optional additional context

        Returns:
            A tuple of (refined content, list of improvements made)
        """
        self._iteration_count = min(self._iteration_count + 1, self.MAX_ITERATIONS)
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        # Format questions as a list
        questions_list = "\n".join(
            f"{i}. [{q.get('type', 'unknown').upper()}] {q.get('question', '')}"
            for i, q in enumerate(questions, 1)
        )

        system_prompt = (
            "You are a reasoning assistant using the Socratic Self-Refine (SSR) methodology.\n"
            "Generate a refined output that addresses the Socratic questions.\n\n"
            "Your refined output should:\n"
            "1. Address each Socratic question thoughtfully\n"
            "2. Clarify assumptions and provide justification\n"
            "3. Explore alternatives and counterexamples\n"
            "4. Add evidence and verify claims\n"
            "5. Examine implications and consequences"
        )

        user_prompt = (
            f"Previous Socratic questions (step {question_thought.step_number}):\n"
            f"{questions_list}\n\n"
            f"Generate a refined output for iteration {self._iteration_count} that addresses "
            f"these Socratic questions.{guidance_text}\n\n"
            f"Produce an improved version that incorporates insights from the questions."
        )

        def fallback() -> str:
            # Return marker that indicates fallback was used
            return "__FALLBACK__"

        content_text = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        # If fallback was used, delegate to the heuristic method
        if content_text == "__FALLBACK__":
            return self._generate_refinement(question_thought, questions, guidance, context)

        # Generate improvements list based on questions
        improvements: list[str] = []
        for q in questions:
            q_type = q.get("type", "unknown")
            if q_type == self.QUESTION_WHY:
                improvements.append("Clarified underlying assumptions and their validity")
            elif q_type == self.QUESTION_WHAT_IF:
                improvements.append("Addressed counterexamples and alternative perspectives")
            elif q_type == self.QUESTION_HOW:
                improvements.append("Added evidence and justification for claims")
            elif q_type == self.QUESTION_IMPLICATIONS:
                improvements.append("Explored implications and consequences")

        # Track improvements
        self._improvements_made.extend(improvements)

        content = (
            f"Step {self._step_counter}: Refined Output (Iteration {self._iteration_count})\n\n"
            f"{content_text}\n\n"
            f"Improvements applied: {len(improvements)} areas addressed"
        )

        return content, improvements

    async def _sample_validation(
        self,
        refinement_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, bool]:
        """Validate refinement using LLM sampling.

        Args:
            refinement_thought: The refined output to validate
            guidance: Optional guidance for validation
            context: Optional additional context

        Returns:
            A tuple of (validation content, whether validation passed)
        """
        iteration = self._iteration_count
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = (
            "You are a reasoning assistant using the Socratic Self-Refine (SSR) methodology.\n"
            "Validate whether the refinement has adequately addressed the Socratic questions.\n\n"
            "Evaluate:\n"
            "1. WHY questions: Are assumptions clarified and justified?\n"
            "2. WHAT IF questions: Are alternatives and counterexamples explored?\n"
            "3. HOW questions: Is evidence provided and claims verified?\n"
            "4. IMPLICATIONS questions: Are consequences examined?\n\n"
            "Respond with:\n"
            "VALIDATION: PASSED or NEEDS MORE REFINEMENT\n"
            "Then explain your assessment."
        )

        user_prompt = (
            f"Refined output to validate (step {refinement_thought.step_number}):\n"
            f"{refinement_thought.content}\n\n"
            f"Validate whether iteration {iteration} has adequately addressed the Socratic "
            f"questions.{guidance_text}"
        )

        def fallback() -> str:
            # Return marker that indicates fallback was used
            return "__FALLBACK__"

        content_text = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

        # If fallback was used, delegate to the heuristic method
        if content_text == "__FALLBACK__":
            return self._validate_refinement(refinement_thought, guidance, context)

        # Parse validation result
        validation_passed = "PASSED" in content_text.upper()

        content = f"Step {self._step_counter}: Validation (Iteration {iteration})\n\n{content_text}"

        return content, validation_passed

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Args:
            previous_thought: The previous thought to build on
            guidance: Optional guidance for the conclusion
            context: Optional additional context

        Returns:
            The sampled conclusion content
        """
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = (
            "You are a reasoning assistant using the Socratic Self-Refine (SSR) methodology.\n"
            "Generate a final conclusion that synthesizes all insights from the Socratic "
            "questioning and refinement process.\n\n"
            "Your conclusion should:\n"
            "1. Incorporate all improvements from the refinement cycles\n"
            "2. Demonstrate deeper understanding than the initial output\n"
            "3. Show how assumptions were challenged and addressed\n"
            "4. Reflect evidence-based reasoning\n"
            "5. Acknowledge implications and consequences"
        )

        user_prompt = (
            f"After {self._iteration_count} iterations of Socratic self-refinement, "
            f"generate a final conclusion.\n\n"
            f"Summary:\n"
            f"- Total Socratic questions asked: {len(self._questions_asked)}\n"
            f"- Total improvements made: {len(self._improvements_made)}\n"
            f"- Refinement iterations: {self._iteration_count}\n\n"
            f"Generate a final, comprehensive answer that incorporates all insights from the "
            f"Socratic questioning and refinement process.{guidance_text}"
        )

        step_counter = self._step_counter  # Capture for closure

        def fallback() -> str:
            return self._generate_conclusion(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1500,
        )

        # Add step prefix if content doesn't already have it (i.e., it was sampled)
        if not content.startswith(f"Step {step_counter}"):
            return f"Step {step_counter}: Final Conclusion\n\n{content}"
        return content


__all__ = ["SSR", "SSR_METADATA"]
