"""Self-Ask reasoning method.

This module implements the Self-Ask reasoning method, which uses recursive follow-up
questioning to decompose complex problems. The method asks "What do I need to know to
answer this?" and generates sub-questions, answering each one to progressively build
toward the main answer.
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
    from reasoning_mcp.models import Session


# Metadata for Self-Ask method
SELF_ASK_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_ASK,
    name="Self-Ask",
    description="Decompose questions into subquestions and answer them iteratively. "
    "Uses recursive follow-up questioning to build understanding progressively.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "self-ask",
            "decomposition",
            "subquestions",
            "recursive",
            "progressive",
            "question-driven",
        }
    ),
    complexity=4,  # Medium complexity - requires question decomposition
    supports_branching=False,  # Linear decomposition
    supports_revision=True,  # Can revise questions and answers
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At least: main question, one subquestion, one answer
    max_thoughts=0,  # No limit - depends on question complexity
    avg_tokens_per_thought=400,  # Moderate - questions + answers
    best_for=(
        "complex questions requiring background knowledge",
        "multi-step reasoning problems",
        "information synthesis tasks",
        "questions with dependencies",
        "educational contexts",
        "knowledge-intensive queries",
    ),
    not_recommended_for=(
        "simple factual questions",
        "creative tasks without clear sub-problems",
        "problems requiring parallel exploration",
        "highly abstract or philosophical questions",
    ),
)


class SelfAsk(ReasoningMethodBase):
    """Self-Ask reasoning method implementation.

    This class implements the Self-Ask reasoning pattern, which decomposes complex
    questions into simpler sub-questions. It recursively asks "What do I need to know
    to answer this?" and systematically answers each sub-question to build toward
    the final answer.

    Key characteristics:
    - Question decomposition into sub-questions
    - Recursive follow-up questioning
    - Progressive answer building
    - Systematic knowledge accumulation
    - Dependency-aware reasoning

    The reasoning flow typically follows this pattern:
    1. Analyze the main question
    2. Identify what needs to be known
    3. Generate sub-questions
    4. Answer each sub-question (may generate further sub-questions)
    5. Synthesize answers into the main answer

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SelfAsk()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How does photosynthesis work in plants?"
        ... )
        >>> print(result.content)  # Initial question analysis

        Continue with sub-questions:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Generate first sub-question"
        ... )
        >>> print(next_thought.type)  # ThoughtType.HYPOTHESIS (sub-question)
    """

    def __init__(self) -> None:
        """Initialize the Self-Ask method."""
        self._initialized = False
        self._step_counter = 0
        self._question_stack: list[str] = []  # Track pending sub-questions
        self._answered_questions: dict[str, str] = {}  # Track Q&A pairs
        self._use_sampling = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SELF_ASK

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SELF_ASK_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SELF_ASK_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Self-Ask method for execution.
        It resets the question stack and answered questions tracking.

        Examples:
            >>> method = SelfAsk()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert len(method._question_stack) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._question_stack = []
        self._answered_questions = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Self-Ask method.

        This method creates the initial thought that analyzes the main question
        and begins the decomposition process.

        Args:
            session: The current reasoning session
            input_text: The main question or problem to reason about
            context: Optional additional context

        Returns:
            A ThoughtNode representing the initial question analysis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfAsk()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Why is the sky blue?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.SELF_ASK
        """
        if not self._initialized:
            raise RuntimeError("Self-Ask method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._question_stack = []
        self._answered_questions = {}

        # Create the initial thought analyzing the main question
        if self._use_sampling:
            content = await self._sample_initial_analysis(input_text, context)
        else:
            content = self._generate_initial_analysis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_ASK,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Moderate initial confidence
            metadata={
                "main_question": input_text,
                "context": context or {},
                "reasoning_type": "self_ask",
                "phase": "initial_analysis",
                "pending_questions": len(self._question_stack),
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SELF_ASK

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

        This method generates the next step in the Self-Ask process, which may be:
        - A sub-question (HYPOTHESIS type)
        - An answer to a sub-question (VERIFICATION type)
        - A synthesis of answers (SYNTHESIS type)
        - The final conclusion (CONCLUSION type)

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the self-ask reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SelfAsk()
            >>> await method.initialize()
            >>> first = await method.execute(session, "How does rain form?")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Generate first sub-question"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError("Self-Ask method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine the next step based on the current phase
        phase = self._determine_next_phase(previous_thought, guidance)

        # Generate content based on phase (use sampling if available)
        if phase == "generate_subquestion":
            if self._use_sampling:
                content, thought_type = await self._sample_subquestion(
                    previous_thought, guidance, context
                )
            else:
                content, thought_type = self._generate_subquestion(
                    previous_thought, guidance, context
                )
        elif phase == "answer_subquestion":
            if self._use_sampling:
                content, thought_type = await self._sample_answer(
                    previous_thought, guidance, context
                )
            else:
                content, thought_type = self._answer_subquestion(
                    previous_thought, guidance, context
                )
        elif phase == "synthesize":
            if self._use_sampling:
                content, thought_type = await self._sample_synthesis(previous_thought, context)
            else:
                content, thought_type = self._synthesize_answers(previous_thought, context)
        else:  # conclude
            if self._use_sampling:
                content, thought_type = await self._sample_conclusion(previous_thought, context)
            else:
                content, thought_type = self._generate_conclusion(previous_thought, context)

        # Calculate confidence based on how many questions have been answered
        confidence = self._calculate_confidence()

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SELF_ASK,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "phase": phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "self_ask",
                "pending_questions": len(self._question_stack),
                "answered_count": len(self._answered_questions),
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Self-Ask, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SelfAsk()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial question analysis.

        This helper method analyzes the main question and identifies what needs
        to be known to answer it.

        Args:
            input_text: The main question to analyze
            context: Optional additional context

        Returns:
            The content for the initial thought

        Note:
            In a full implementation, this would use an LLM to generate
            the actual analysis and identify necessary sub-questions.
        """
        return (
            f"Step {self._step_counter}: Initial Question Analysis\n\n"
            f"Main Question: {input_text}\n\n"
            f"To answer this question comprehensively, I need to decompose it "
            f"into smaller, more manageable sub-questions. Let me identify what "
            f"background knowledge and intermediate answers are needed.\n\n"
            f"This will involve asking myself: 'What do I need to know to answer this?'"
        )

    def _determine_next_phase(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> str:
        """Determine what the next reasoning phase should be.

        Args:
            previous_thought: The previous thought in the chain
            guidance: Optional guidance from the user

        Returns:
            The next phase: generate_subquestion, answer_subquestion,
            synthesize, or conclude
        """
        previous_phase = previous_thought.metadata.get("phase", "initial_analysis")

        # If guidance explicitly requests something, honor it
        # Note: Check more specific keywords first (e.g., "final" before "answer")
        if guidance:
            guidance_lower = guidance.lower()
            if "conclude" in guidance_lower or "final" in guidance_lower:
                return "conclude"
            if "synthesize" in guidance_lower or "combine" in guidance_lower:
                return "synthesize"
            if "question" in guidance_lower or "ask" in guidance_lower:
                return "generate_subquestion"
            if "answer" in guidance_lower:
                return "answer_subquestion"

        # Default progression based on previous phase
        if previous_phase in ("initial_analysis", "answer_subquestion"):
            # After analysis or answering, check if more questions needed
            if len(self._question_stack) > 0:
                return "answer_subquestion"
            elif len(self._answered_questions) < 2:
                return "generate_subquestion"
            else:
                return "synthesize"
        elif previous_phase == "generate_subquestion":
            return "answer_subquestion"
        elif previous_phase == "synthesize":
            return "conclude"

        return "generate_subquestion"

    def _generate_subquestion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Generate a sub-question.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        # In a real implementation, this would use an LLM to generate
        # contextually appropriate sub-questions
        main_question = previous_thought.metadata.get("main_question", "the main question")

        # Generate a sub-question based on what we know so far
        sub_question = (
            f"What specific aspect or prerequisite knowledge do we need "
            f"to understand {main_question}?"
        )

        # Add to question stack
        self._question_stack.append(sub_question)

        content = (
            f"Step {self._step_counter}: Sub-Question Generation\n\n"
            f"Follow-up question: {sub_question}\n\n"
            f"This sub-question will help build the necessary understanding "
            f"to answer the main question. I'll need to answer this before "
            f"proceeding further."
        )

        return content, ThoughtType.HYPOTHESIS

    def _answer_subquestion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Answer a sub-question.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        # Get the most recent sub-question
        if self._question_stack:
            current_question = self._question_stack.pop()
        else:
            current_question = "the current sub-question"

        # In a real implementation, this would use an LLM to generate
        # the actual answer
        answer = (
            "Based on available knowledge and previous insights, here is "
            "the answer to this sub-question."
        )

        # Store the Q&A pair
        self._answered_questions[current_question] = answer

        content = (
            f"Step {self._step_counter}: Sub-Question Answer\n\n"
            f"Question: {current_question}\n\n"
            f"Answer: {answer}\n\n"
            f"This answer contributes to building our understanding toward "
            f"the main question."
        )

        return content, ThoughtType.VERIFICATION

    def _synthesize_answers(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Synthesize answers from sub-questions.

        Args:
            previous_thought: The previous thought
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        # In a real implementation, this would use an LLM to synthesize
        # all the Q&A pairs into a coherent understanding

        qa_summary = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in self._answered_questions.items())

        content = (
            f"Step {self._step_counter}: Synthesis of Sub-Answers\n\n"
            f"I've answered {len(self._answered_questions)} sub-questions. "
            f"Let me synthesize these answers to build a comprehensive response.\n\n"
            f"Sub-Questions and Answers:\n{qa_summary}\n\n"
            f"By combining these insights, I can now form a well-informed "
            f"answer to the main question."
        )

        return content, ThoughtType.SYNTHESIS

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Generate the final conclusion.

        Args:
            previous_thought: The previous thought
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        main_question = previous_thought.metadata.get("main_question", "the main question")

        content = (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Main Question: {main_question}\n\n"
            f"Final Answer: Based on the {len(self._answered_questions)} "
            f"sub-questions I've answered and synthesized, here is my "
            f"comprehensive answer to the main question.\n\n"
            f"This answer is built upon systematic decomposition and "
            f"progressive understanding of the necessary components."
        )

        return content, ThoughtType.CONCLUSION

    def _calculate_confidence(self) -> float:
        """Calculate confidence based on the current state.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Increase confidence with each answered question
        confidence += min(0.4, len(self._answered_questions) * 0.1)

        # Decrease slightly if there are many pending questions
        confidence -= min(0.2, len(self._question_stack) * 0.05)

        # Ensure within valid range
        return max(0.3, min(0.95, confidence))

    # ========================
    # Sampling Methods
    # ========================

    async def _sample_initial_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial question analysis using LLM sampling.

        Args:
            input_text: The main question to analyze
            context: Optional additional context

        Returns:
            The content for the initial thought
        """
        system_prompt = """You are a reasoning assistant using the Self-Ask methodology.
Analyze the given question and identify what sub-questions need to be answered first.
Think about what background knowledge or intermediate understanding is needed.
Your analysis should set up the question decomposition process."""

        user_prompt = f"""Main Question: {input_text}

Analyze this question using Self-Ask methodology:
1. What is the core question asking?
2. What do I need to know to answer this comprehensively?
3. What are the key concepts or dependencies involved?

Begin your initial analysis to set up the questioning process."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_initial_analysis(input_text, context),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=600,
        )

    async def _sample_subquestion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Generate a sub-question using LLM sampling.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        main_question = previous_thought.metadata.get("main_question", "the main question")
        answered_so_far = list(self._answered_questions.keys())

        system_prompt = """You are a reasoning assistant using Self-Ask methodology.
Generate a specific, targeted sub-question that will help answer the main question.
The sub-question should be answerable and contribute to building understanding."""

        user_prompt = f"""Main Question: {main_question}

Previously answered sub-questions: {answered_so_far if answered_so_far else "None yet"}

{f"Guidance: {guidance}" if guidance else ""}

Generate a follow-up sub-question that will help answer the main question.
The sub-question should:
1. Be specific and answerable
2. Not duplicate previous questions
3. Build toward understanding the main question"""

        def fallback() -> str:
            content, _ = self._generate_subquestion(previous_thought, guidance, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=400,
        )

        # Extract the sub-question for the stack
        sub_question = content.split("?")[0] + "?" if "?" in content else content[:100]
        self._question_stack.append(sub_question)

        formatted_content = f"Step {self._step_counter}: Sub-Question Generation\n\n{content}"
        return formatted_content, ThoughtType.HYPOTHESIS

    async def _sample_answer(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Answer a sub-question using LLM sampling.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        # Get the current sub-question
        if self._question_stack:
            current_question = self._question_stack.pop()
        else:
            current_question = "the current sub-question"

        main_question = previous_thought.metadata.get("main_question", "the main question")

        system_prompt = """You are a reasoning assistant using Self-Ask methodology.
Provide a clear, informative answer to the sub-question.
Your answer should be accurate and contribute to answering the main question."""

        user_prompt = f"""Main Question: {main_question}

Sub-Question to Answer: {current_question}

Previously answered questions and answers:
{chr(10).join(f"Q: {q}{chr(10)}A: {a}" for q, a in self._answered_questions.items()) if self._answered_questions else "None yet"}

Provide a clear, comprehensive answer to the sub-question.
Explain how this answer helps toward the main question."""

        def fallback() -> str:
            # We need to re-add the question to the stack for the fallback method
            self._question_stack.append(current_question)
            content, _ = self._answer_subquestion(previous_thought, guidance, context)
            return content

        answer = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=600,
        )

        # Store the Q&A pair
        self._answered_questions[current_question] = answer

        formatted_content = (
            f"Step {self._step_counter}: Sub-Question Answer\n\n"
            f"Question: {current_question}\n\n"
            f"Answer: {answer}"
        )
        return formatted_content, ThoughtType.VERIFICATION

    async def _sample_synthesis(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Synthesize answers using LLM sampling.

        Args:
            previous_thought: The previous thought
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        main_question = previous_thought.metadata.get("main_question", "the main question")

        system_prompt = """You are a reasoning assistant using Self-Ask methodology.
Synthesize all the answered sub-questions into a coherent understanding.
Show how the pieces fit together to address the main question."""

        qa_pairs = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in self._answered_questions.items())

        user_prompt = f"""Main Question: {main_question}

Answered Sub-Questions:
{qa_pairs}

Synthesize these answers into a coherent understanding.
Show how these pieces connect and build toward answering the main question."""

        def fallback() -> str:
            content, _ = self._synthesize_answers(previous_thought, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800,
        )

        formatted_content = f"Step {self._step_counter}: Synthesis of Sub-Answers\n\n{content}"
        return formatted_content, ThoughtType.SYNTHESIS

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> tuple[str, ThoughtType]:
        """Generate the final conclusion using LLM sampling.

        Args:
            previous_thought: The previous thought
            context: Optional context

        Returns:
            Tuple of (content, thought_type)
        """
        main_question = previous_thought.metadata.get("main_question", "the main question")

        system_prompt = """You are a reasoning assistant using Self-Ask methodology.
Provide the final, comprehensive answer to the main question.
Your answer should integrate all the insights from the sub-questions."""

        qa_pairs = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in self._answered_questions.items())

        user_prompt = f"""Main Question: {main_question}

All Sub-Questions and Answers:
{qa_pairs}

Provide the final, comprehensive answer to the main question.
Integrate all insights from the self-questioning process.
Be clear, complete, and well-organized."""

        def fallback() -> str:
            content, _ = self._generate_conclusion(previous_thought, context)
            return content

        content = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=1000,
        )

        formatted_content = (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Main Question: {main_question}\n\n"
            f"Final Answer: {content}"
        )
        return formatted_content, ThoughtType.CONCLUSION
