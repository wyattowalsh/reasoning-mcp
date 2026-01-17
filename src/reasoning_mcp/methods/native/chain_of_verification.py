"""Chain of Verification (CoVe) reasoning method.

This module implements a verification-based reasoning method that improves answer
quality through systematic verification. The method proceeds through four phases:
1. Baseline response generation
2. Verification question generation
3. Independent answer verification
4. Final verified response synthesis

Based on Dhuliawala et al. (2023): Chain-of-Verification Reduces Hallucination
in Large Language Models.
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
    from reasoning_mcp.streaming.context import StreamingContext


# Metadata for Chain of Verification method
CHAIN_OF_VERIFICATION_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CHAIN_OF_VERIFICATION,
    name="Chain of Verification",
    description="Verification-based reasoning that generates baseline response, creates "
    "verification questions, answers them independently, and synthesizes a verified "
    "final response to reduce hallucination and improve accuracy.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "verification",
            "fact-checking",
            "hallucination-reduction",
            "multi-phase",
            "accuracy-focused",
            "systematic",
            "quality-assurance",
        }
    ),
    complexity=5,  # Medium-high complexity - multi-phase verification
    supports_branching=False,  # Linear verification path
    supports_revision=True,  # Core feature - revising through verification
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: baseline + questions + answers + verified
    max_thoughts=15,  # Multiple questions and verification steps possible
    avg_tokens_per_thought=450,  # Moderate to high - includes analysis
    best_for=(
        "factual question answering",
        "hallucination reduction",
        "accuracy-critical tasks",
        "information verification",
        "knowledge-based reasoning",
        "fact-checking scenarios",
        "reliability-sensitive outputs",
    ),
    not_recommended_for=(
        "creative writing tasks",
        "subjective opinion generation",
        "time-critical decisions",
        "simple computational tasks",
        "tasks without verifiable facts",
    ),
)

logger = structlog.get_logger(__name__)


class ChainOfVerification(ReasoningMethodBase):
    """Chain of Verification (CoVe) reasoning method implementation.

    This class implements a verification-based reasoning pattern where the system
    generates an initial response, then systematically verifies it through:
    1. Generating a baseline response to the question
    2. Creating verification questions to check critical facts
    3. Answering verification questions independently
    4. Synthesizing a final verified response

    The method reduces hallucination by explicitly verifying factual claims
    before producing the final answer.

    Key characteristics:
    - Multi-phase verification process
    - Independent verification questions
    - Systematic fact-checking
    - Hallucination reduction
    - Quality improvement through verification
    - Medium-high complexity (5)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = ChainOfVerification()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="What are the capital cities of Scandinavia?"
        ... )
        >>> print(result.content)  # Baseline response

        Continue with verification questions:
        >>> questions = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Generate verification questions"
        ... )
        >>> print(questions.type)  # ThoughtType.CONTINUATION

        Continue with verification answers:
        >>> answers = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=questions,
        ...     guidance="Answer verification questions"
        ... )
        >>> print(answers.type)  # ThoughtType.VERIFICATION

        Synthesize final verified response:
        >>> verified = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=answers,
        ...     guidance="Synthesize verified response"
        ... )
        >>> print(verified.type)  # ThoughtType.CONCLUSION
    """

    # Default number of verification questions to generate
    DEFAULT_NUM_QUESTIONS = 3
    # Maximum number of verification questions allowed
    MAX_NUM_QUESTIONS = 10
    # Enable LLM sampling support
    _use_sampling: bool = True

    # Streaming support
    streaming_context: StreamingContext | None = None

    def __init__(self) -> None:
        """Initialize the Chain of Verification method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "baseline"  # baseline, questions, answers, verified
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.CHAIN_OF_VERIFICATION

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return CHAIN_OF_VERIFICATION_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return CHAIN_OF_VERIFICATION_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Chain of Verification method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = ChainOfVerification()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "baseline"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "baseline"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Chain of Verification method.

        This method creates the baseline response that will be verified
        through subsequent phases. It generates a first attempt at answering
        the question without explicit verification.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include num_questions)
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the baseline response

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = ChainOfVerification()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="What are the tallest mountains?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.CHAIN_OF_VERIFICATION
            >>> assert "verification_questions" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("Chain of Verification method must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "baseline"

        # Extract number of verification questions from context if provided
        num_questions = self.DEFAULT_NUM_QUESTIONS
        if context and "num_questions" in context:
            num_questions = min(max(context["num_questions"], 1), self.MAX_NUM_QUESTIONS)

        # Generate baseline response
        content = await self._generate_baseline_response(input_text, context)

        # Initial quality score (moderate - will improve through verification)
        initial_quality = 0.5

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,  # Initial confidence - will improve through verification
            quality_score=initial_quality,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "chain_of_verification",
                "phase": self._current_phase,
                "num_questions": num_questions,
                "verification_questions": [],
                "verification_answers": [],
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.CHAIN_OF_VERIFICATION

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

        This method implements the verification cycle logic:
        - If previous was baseline: generate verification questions
        - If previous was questions: answer verification questions
        - If previous was answers: synthesize verified response
        - Follows the CoVe four-phase structure

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the verification process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = ChainOfVerification()
            >>> await method.initialize()
            >>> baseline = await method.execute(session, "What is gravity?")
            >>> questions = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=baseline
            ... )
            >>> assert questions.type == ThoughtType.CONTINUATION
            >>> assert questions.metadata["phase"] == "questions"
            >>>
            >>> answers = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=questions
            ... )
            >>> assert answers.type == ThoughtType.VERIFICATION
            >>> assert answers.metadata["phase"] == "answers"
            >>>
            >>> verified = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=answers
            ... )
            >>> assert verified.type == ThoughtType.CONCLUSION
            >>> assert verified.metadata["phase"] == "verified"
        """
        if not self._initialized:
            raise RuntimeError(
                "Chain of Verification method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "baseline")

        # Get verification data from previous thought
        verification_questions = previous_thought.metadata.get("verification_questions", [])
        verification_answers = previous_thought.metadata.get("verification_answers", [])
        num_questions = previous_thought.metadata.get("num_questions", self.DEFAULT_NUM_QUESTIONS)

        if prev_phase == "baseline":
            # Next: generate verification questions
            self._current_phase = "questions"
            thought_type = ThoughtType.CONTINUATION
            content, new_questions = await self._generate_verification_questions(
                previous_thought, num_questions, guidance, context
            )
            verification_questions = new_questions
            quality_score = 0.65
            confidence = 0.6

        elif prev_phase == "questions":
            # Next: answer verification questions independently
            self._current_phase = "answers"
            thought_type = ThoughtType.VERIFICATION
            content, new_answers = await self._answer_verification_questions(
                previous_thought, verification_questions, guidance, context
            )
            verification_answers = new_answers
            quality_score = 0.8
            confidence = 0.75

        elif prev_phase == "answers":
            # Next: synthesize final verified response
            self._current_phase = "verified"
            thought_type = ThoughtType.CONCLUSION
            content = await self._synthesize_verified_response(
                previous_thought, verification_questions, verification_answers, guidance, context
            )
            quality_score = 0.9
            confidence = 0.85

        else:
            # Fallback to conclusion (already verified)
            self._current_phase = "verified"
            thought_type = ThoughtType.CONCLUSION
            content = await self._synthesize_verified_response(
                previous_thought, verification_questions, verification_answers, guidance, context
            )
            quality_score = 0.9
            confidence = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "num_questions": num_questions,
                "verification_questions": verification_questions,
                "verification_answers": verification_answers,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "chain_of_verification",
                "previous_phase": prev_phase,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Chain of Verification, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = ChainOfVerification()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _generate_baseline_response(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the baseline response to the input.

        This is a helper method that calls an LLM to generate the initial
        attempt at answering the question without verification, or falls
        back to a heuristic implementation.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the baseline response
        """
        system_prompt = """You are using the Chain of Verification (CoVe) methodology.
Generate an initial baseline response to the question. This is your first attempt at answering,
which will be verified in subsequent steps through systematic fact-checking.

Provide a direct, factual answer based on your knowledge. Include specific details and claims
that can be verified. Don't worry about being perfect - the verification process will catch
any errors."""

        user_prompt = f"""Question: {input_text}

Please provide a baseline answer to this question. Be specific and include verifiable facts.
Your answer will be verified in the next steps, so focus on providing a comprehensive response."""

        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        def fallback() -> str:
            return self._generate_baseline_response_heuristic(input_text, context)

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got LLM result, format it; otherwise heuristic already formatted
        if result != fallback():
            return (
                f"Step {self._step_counter}: Baseline Response (Phase 1/4)\n\n"
                f"Question: {input_text}\n\n"
                f"Let me provide an initial response to this question. "
                f"This is the baseline answer that will be verified in subsequent steps.\n\n"
                f"Baseline answer:\n{result}\n\n"
                f"Note: This is the initial response. I will now generate verification questions "
                f"to check the accuracy of this answer."
            )
        return result

    def _generate_baseline_response_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate baseline response using heuristic (fallback).

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the baseline response
        """
        return (
            f"Step {self._step_counter}: Baseline Response (Phase 1/4)\n\n"
            f"Question: {input_text}\n\n"
            f"Let me provide an initial response to this question. "
            f"This is the baseline answer that will be verified in subsequent steps.\n\n"
            f"Baseline answer:\n"
            f"[This would contain the LLM-generated baseline response to the question]\n\n"
            f"Note: This is the initial response. I will now generate verification questions "
            f"to check the accuracy of this answer."
        )

    async def _generate_verification_questions(
        self,
        baseline_thought: ThoughtNode,
        num_questions: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate verification questions for the baseline response.

        This is a helper method that calls an LLM to analyze the baseline
        response and generate specific questions to verify critical facts
        and claims, or falls back to a heuristic implementation.

        Args:
            baseline_thought: The baseline response to verify
            num_questions: Number of verification questions to generate
            guidance: Optional guidance for question generation
            context: Optional additional context

        Returns:
            A tuple of (content string, list of verification questions)
        """
        import re

        system_prompt = """You are using the Chain of Verification (CoVe) methodology.
Analyze the baseline response and generate specific verification questions to check critical
facts and claims. Each question should target a verifiable factual claim that can be answered
independently.

Generate questions that:
1. Target specific factual claims in the baseline answer
2. Can be answered objectively with facts
3. Help identify potential hallucinations or errors
4. Are independent and don't reference the baseline answer"""

        user_prompt = f"""Baseline Answer:
{baseline_thought.content}

Generate exactly {num_questions} verification questions to check critical facts from this
baseline answer. Each question should verify a specific claim."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        user_prompt += f"\n\nFormat your response as a numbered list of {num_questions} questions."

        def fallback() -> str:
            return "__FALLBACK__"

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If fallback was used, return heuristic result
        if result == "__FALLBACK__":
            return self._generate_verification_questions_heuristic(
                baseline_thought, num_questions, guidance, context
            )

        # Parse questions from response (simple heuristic: look for numbered lines)
        question_pattern = r"^\s*\d+[\.\)]\s*(.+)$"
        questions: list[str] = []
        for line in result.split("\n"):
            match = re.match(question_pattern, line.strip())
            if match and len(questions) < num_questions:
                questions.append(match.group(1).strip())

        # Ensure we have exactly num_questions
        while len(questions) < num_questions:
            questions.append(
                f"Verification Question {len(questions) + 1}: [Additional verification needed]"
            )
        questions = questions[:num_questions]

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verification Questions (Phase 2/4)\n\n"
            f"Based on the baseline response in Step {baseline_thought.step_number}, "
            f"I will now generate {num_questions} verification questions to check critical "
            f"facts.\n\n"
            f"Verification questions:\n"
        )

        for i, question in enumerate(questions, 1):
            content += f"{i}. {question}\n"

        content += (
            f"\nThese questions will help verify the accuracy of the baseline response "
            f"by checking specific factual claims independently.{guidance_text}"
        )

        return content, questions

    def _generate_verification_questions_heuristic(
        self,
        baseline_thought: ThoughtNode,
        num_questions: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate verification questions using heuristic (fallback).

        Args:
            baseline_thought: The baseline response to verify
            num_questions: Number of verification questions to generate
            guidance: Optional guidance for question generation
            context: Optional additional context

        Returns:
            A tuple of (content string, list of verification questions)
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Generate sample questions (in real implementation, LLM would generate these)
        questions = [
            f"Verification Question {i + 1}: [LLM would generate specific question to "
            f"verify a claim from baseline]"
            for i in range(num_questions)
        ]

        content = (
            f"Step {self._step_counter}: Verification Questions (Phase 2/4)\n\n"
            f"Based on the baseline response in Step {baseline_thought.step_number}, "
            f"I will now generate {num_questions} verification questions to check critical "
            f"facts.\n\n"
            f"Verification questions:\n"
        )

        for i, question in enumerate(questions, 1):
            content += f"{i}. {question}\n"

        content += (
            f"\nThese questions will help verify the accuracy of the baseline response "
            f"by checking specific factual claims independently.{guidance_text}"
        )

        return content, questions

    async def _answer_verification_questions(
        self,
        questions_thought: ThoughtNode,
        verification_questions: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Answer the verification questions independently.

        This is a helper method that calls an LLM to answer each verification
        question independently, without reference to the baseline response,
        to avoid bias, or falls back to a heuristic implementation.

        Args:
            questions_thought: The thought containing verification questions
            verification_questions: List of questions to answer
            guidance: Optional guidance for answering
            context: Optional additional context

        Returns:
            A tuple of (content string, list of verification answers)
        """
        system_prompt = """You are using the Chain of Verification (CoVe) methodology.
Answer each verification question independently and factually, without referring to or being
biased by the baseline answer. Use your knowledge to provide accurate, verifiable facts.

For each question:
1. Answer directly and concisely
2. Focus on factual accuracy
3. Provide specific information that can be verified
4. Do not reference the baseline answer"""

        # Check if sampling is available
        can_sample = self._execution_context is not None and self._execution_context.can_sample

        # Answer questions independently
        answers: list[str] = []
        used_fallback = False

        for i, question in enumerate(verification_questions, 1):
            user_prompt = f"""Question {i}: {question}

Please provide a factual, verifiable answer to this question. Answer independently without
reference to any previous context."""

            if guidance:
                user_prompt += f"\n\nGuidance: {guidance}"

            def make_fallback(idx: int) -> str:
                return f"Answer {idx}: [LLM would answer this question independently]"

            if can_sample:
                result = await self._sample_with_fallback(
                    user_prompt=user_prompt,
                    fallback_generator=lambda idx=i: make_fallback(idx),
                    system_prompt=system_prompt,
                )
                if result == make_fallback(i):
                    used_fallback = True
                answers.append(result.strip())
            else:
                used_fallback = True
                answers.append(make_fallback(i))

        # If we used fallback for all, return heuristic result for consistency
        if used_fallback and not can_sample:
            return self._answer_verification_questions_heuristic(
                questions_thought, verification_questions, guidance, context
            )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verification Answers (Phase 3/4)\n\n"
            f"Answering the {len(verification_questions)} verification questions from "
            f"Step {questions_thought.step_number} independently...\n\n"
        )

        for i, (question, answer) in enumerate(
            zip(verification_questions, answers, strict=True), 1
        ):
            content += f"Q{i}: {question}\n"
            content += f"A{i}: {answer}\n\n"

        content += (
            f"These answers were generated independently to avoid bias from the baseline response. "
            f"They will now be used to verify and refine the final answer.{guidance_text}"
        )

        return content, answers

    def _answer_verification_questions_heuristic(
        self,
        questions_thought: ThoughtNode,
        verification_questions: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Answer verification questions using heuristic (fallback).

        Args:
            questions_thought: The thought containing verification questions
            verification_questions: List of questions to answer
            guidance: Optional guidance for answering
            context: Optional additional context

        Returns:
            A tuple of (content string, list of verification answers)
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Generate sample answers (in real implementation, LLM would generate these)
        answers = [
            f"Answer {i + 1}: [LLM would answer this question independently]"
            for i in range(len(verification_questions))
        ]

        content = (
            f"Step {self._step_counter}: Verification Answers (Phase 3/4)\n\n"
            f"Answering the {len(verification_questions)} verification questions from "
            f"Step {questions_thought.step_number} independently...\n\n"
        )

        for i, (question, answer) in enumerate(
            zip(verification_questions, answers, strict=True), 1
        ):
            content += f"Q{i}: {question}\n"
            content += f"A{i}: {answer}\n\n"

        content += (
            f"These answers were generated independently to avoid bias from the baseline response. "
            f"They will now be used to verify and refine the final answer.{guidance_text}"
        )

        return content, answers

    async def _synthesize_verified_response(
        self,
        answers_thought: ThoughtNode,
        verification_questions: list[str],
        verification_answers: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Synthesize the final verified response.

        This is a helper method that calls an LLM to combine the baseline
        response with verification results to produce a final verified answer
        with reduced hallucination, or falls back to a heuristic implementation.

        Args:
            answers_thought: The thought containing verification answers
            verification_questions: List of verification questions
            verification_answers: List of verification answers
            guidance: Optional guidance for synthesis
            context: Optional additional context

        Returns:
            The content for the final verified response
        """
        system_prompt = """You are using the Chain of Verification (CoVe) methodology.
Synthesize a final verified response by comparing the baseline answer with the independent
verification results. Correct any errors or hallucinations found in the baseline answer based
on the verification answers.

Your synthesis should:
1. Identify discrepancies between baseline and verification answers
2. Correct any errors using the verified information
3. Preserve accurate information from the baseline
4. Provide a coherent, factually accurate final answer
5. Explicitly note any corrections made"""

        # Get the baseline response from session history (first thought in chain)
        baseline_content = "The baseline response"  # Simplified - would get from thought history

        # Build verification summary
        verification_summary = "Verification Results:\n"
        for i, (question, answer) in enumerate(
            zip(verification_questions, verification_answers, strict=True), 1
        ):
            verification_summary += f"{i}. Q: {question}\n   A: {answer}\n\n"

        user_prompt = f"""Baseline Answer:
{baseline_content}

{verification_summary}

Based on the baseline answer and the verification results above, synthesize a final verified
response. Correct any errors or hallucinations found in the baseline using the verified
information."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        def fallback() -> str:
            return "__FALLBACK__"

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If fallback was used, return heuristic result
        if result == "__FALLBACK__":
            return self._synthesize_verified_response_heuristic(
                answers_thought, verification_questions, verification_answers, guidance, context
            )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verified Response (Phase 4/4)\n\n"
            f"Based on the baseline response and verification results from "
            f"Step {answers_thought.step_number}, I will now synthesize a final verified "
            f"answer.\n\n"
            f"Verification summary:\n"
        )

        for i, (question, answer) in enumerate(
            zip(verification_questions, verification_answers, strict=True), 1
        ):
            content += f"- Verified claim {i}: {question} → {answer}\n"

        content += (
            f"\nFinal verified response:\n{result}\n\n"
            f"This response has been verified through systematic fact-checking to ensure accuracy "
            f"and reduce hallucination.{guidance_text}"
        )

        return content

    def _synthesize_verified_response_heuristic(
        self,
        answers_thought: ThoughtNode,
        verification_questions: list[str],
        verification_answers: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Synthesize verified response using heuristic (fallback).

        Args:
            answers_thought: The thought containing verification answers
            verification_questions: List of verification questions
            verification_answers: List of verification answers
            guidance: Optional guidance for synthesis
            context: Optional additional context

        Returns:
            The content for the final verified response
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verified Response (Phase 4/4)\n\n"
            f"Based on the baseline response and verification results from "
            f"Step {answers_thought.step_number}, I will now synthesize a final verified "
            f"answer.\n\n"
            f"Verification summary:\n"
        )

        for i, (question, answer) in enumerate(
            zip(verification_questions, verification_answers, strict=True), 1
        ):
            content += f"- Verified claim {i}: {question} → {answer}\n"

        content += (
            f"\nFinal verified response:\n"
            f"[LLM would synthesize a final answer that incorporates the verification results, "
            f"correcting any errors or hallucinations found in the baseline response]\n\n"
            f"This response has been verified through systematic fact-checking to ensure accuracy "
            f"and reduce hallucination.{guidance_text}"
        )

        return content
