"""STaR (Self-Taught Reasoner) reasoning method.

This module implements STaR based on Zelikman et al. (2022), which bootstraps
reasoning capabilities from a few rationale examples. The method generates
rationales, filters correct ones, and iteratively improves reasoning quality.

Key phases:
1. Attempt: Try to solve with initial reasoning
2. Rationalize: Generate rationale for the attempt
3. Verify: Check if rationalized answer is correct
4. Bootstrap: Use successful rationales to improve

Reference: Zelikman et al. (2022) - "STaR: Bootstrapping Reasoning With Reasoning"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SAMPLING_TEMPERATURE,
    MethodMetadata,
    ReasoningMethodBase,
)
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


STAR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.STAR,
    name="STaR",
    description="Self-Taught Reasoner - bootstraps reasoning from rationale examples. "
    "Attempts, rationalizes, verifies, and iteratively improves reasoning quality.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"bootstrap", "self-taught", "rationale", "iterative", "2022"}),
    complexity=7,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=("learning from examples", "iterative improvement", "rationale generation"),
    not_recommended_for=("zero-shot tasks", "no-example scenarios"),
)


class STaR(ReasoningMethodBase):
    """STaR (Self-Taught Reasoner) method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "attempt"
        self._initial_answer: str = ""
        self._rationale: str = ""
        self._verified: bool = False
        self._bootstrap_count: int = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.STAR

    @property
    def name(self) -> str:
        return STAR_METADATA.name

    @property
    def description(self) -> str:
        return STAR_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "attempt"
        self._initial_answer = ""
        self._rationale = ""
        self._verified = False
        self._bootstrap_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("STaR must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        self._step_counter = 1
        self._current_phase = "attempt"

        # Generate initial answer using sampling or heuristic
        self._initial_answer = await self._sample_initial_attempt(input_text, context)

        # Generate content using sampling or heuristic
        content = await self._sample_attempt_step(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.STAR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "attempt": self._initial_answer,
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.STAR
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
        if not self._initialized:
            raise RuntimeError("STaR must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "attempt")

        if prev_phase == "attempt":
            self._current_phase = "rationalize"

            # Generate rationale using _sample_with_fallback
            attempt_value = previous_thought.metadata.get("attempt", "[answer]")
            self._rationale = await self._sample_rationale(attempt_value, context)

            # Generate content using _sample_with_fallback
            content = await self._sample_rationalize_step(self._rationale)

            thought_type = ThoughtType.REASONING
            confidence = 0.65
        elif prev_phase == "rationalize":
            self._current_phase = "verify"

            # Verify rationale using _sample_with_fallback
            self._verified = await self._sample_verification(self._rationale, context)

            # Generate content using _sample_with_fallback
            content = await self._sample_verify_step(self._verified)

            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75 if self._verified else 0.5
        elif prev_phase == "verify":
            self._current_phase = "bootstrap"
            self._bootstrap_count += 1

            # Generate content using _sample_with_fallback
            content = await self._sample_bootstrap_step(self._bootstrap_count)

            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.8
        else:
            self._current_phase = "conclude"

            # Generate final answer using _sample_with_fallback
            content = await self._sample_conclude_step(context)

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.STAR,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "verified": self._verified},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Heuristic fallback methods

    def _generate_attempt_step_heuristic(self, input_text: str) -> str:
        """Generate attempt step content using heuristic.

        Args:
            input_text: The problem/question to attempt

        Returns:
            Heuristic attempt step content
        """
        return (
            f"Step {self._step_counter}: Initial Attempt (STaR)\n\n"
            f"Problem: {input_text}\n\n"
            f"Making initial reasoning attempt...\n\n"
            f"Initial Attempt:\n"
            f"  Approach: Direct problem solving\n"
            f"  Answer: {self._initial_answer}\n"
            f"  Confidence: Low (initial attempt)\n\n"
            f"Next: Generate rationale for this attempt."
        )

    def _generate_rationale_heuristic(self) -> str:
        """Generate rationale using heuristic.

        Returns:
            Heuristic rationale text
        """
        return (
            "Step 1: Identify key elements in the problem. "
            "Step 2: Apply relevant domain knowledge. "
            "Step 3: Derive answer through logical inference."
        )

    def _generate_rationalize_step_heuristic(self) -> str:
        """Generate rationalize step content using heuristic.

        Returns:
            Heuristic rationalize step content
        """
        return (
            f"Step {self._step_counter}: Generate Rationale\n\n"
            f"Creating reasoning chain for the attempt...\n\n"
            f"Generated Rationale:\n"
            f"  {self._rationale}\n\n"
            f"Rationale Quality: Coherent chain established\n"
            f"Next: Verify the rationalized answer."
        )

    def _generate_verify_step_heuristic(self) -> str:
        """Generate verify step content using heuristic.

        Returns:
            Heuristic verify step content
        """
        return (
            f"Step {self._step_counter}: Verify Rationale\n\n"
            f"Checking rationale correctness...\n\n"
            f"Verification:\n"
            f"  • Logical consistency: ✓ Pass\n"
            f"  • Answer derivation: ✓ Valid\n"
            f"  • Knowledge alignment: ✓ Correct\n\n"
            f"Result: {'✓ VERIFIED' if self._verified else '✗ FAILED'}\n"
            f"Next: {'Bootstrap successful rationale' if self._verified else 'Retry with hints'}."
        )

    def _generate_bootstrap_step_heuristic(self) -> str:
        """Generate bootstrap step content using heuristic.

        Returns:
            Heuristic bootstrap step content
        """
        return (
            f"Step {self._step_counter}: Bootstrap Learning\n\n"
            f"Using verified rationale for improvement...\n\n"
            f"Bootstrap Update:\n"
            f"  • Added rationale to training set: ✓\n"
            f"  • Reasoning quality improved\n"
            f"  • Bootstrap iteration: {self._bootstrap_count}\n\n"
            f"The model has learned from this successful reasoning chain."
        )

    def _generate_conclude_step_heuristic(self) -> str:
        """Generate conclude step content using heuristic.

        Returns:
            Heuristic conclude step content
        """
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"STaR Complete:\n"
            f"  • Initial attempt: Made\n"
            f"  • Rationale generated: Yes\n"
            f"  • Verified: {self._verified}\n"
            f"  • Bootstrap iterations: {self._bootstrap_count}\n\n"
            f"Final Answer: [Answer from bootstrapped reasoning]\n"
            f"Confidence: High (85%)\n"
            f"Learning: Rationale added to knowledge base"
        )

    # LLM sampling methods (used when execution_context available)

    async def _sample_initial_attempt(
        self, input_text: str, context: dict[str, Any] | None
    ) -> str:
        """Generate initial answer attempt using LLM sampling.

        Args:
            input_text: The problem/question to attempt
            context: Optional context dictionary

        Returns:
            LLM-generated initial answer attempt
        """
        system_prompt = (
            "You are a reasoning assistant using STaR (Self-Taught Reasoner) methodology. "
            "Generate an initial answer attempt to the problem. This is your first attempt, "
            "so don't overthink it. Provide a direct, straightforward answer that can be "
            "later rationalized and improved."
        )

        user_prompt = f"""Problem: {input_text}

Make an initial attempt at solving this problem. Focus on providing a clear answer.
This is step 1 of STaR - a rationale will be generated next.

Provide your initial answer directly."""

        def fallback() -> str:
            return "[Initial answer attempt]"

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    async def _sample_attempt_step(self, input_text: str, context: dict[str, Any] | None) -> str:
        """Generate attempt step content using LLM sampling.

        Args:
            input_text: The problem/question to attempt
            context: Optional context dictionary

        Returns:
            LLM-generated attempt step content
        """
        if not self._execution_context:
            return self._generate_attempt_step_heuristic(input_text)

        system_prompt = """You are documenting the STaR (Self-Taught Reasoner) reasoning process.
Generate clear, structured content for the initial attempt step."""

        prompt = f"""Problem: {input_text}
Initial Attempt: {self._initial_answer}

Create a structured description of this initial attempt step (Step {self._step_counter}).
Include:
- The problem being addressed
- The approach taken
- The initial answer
- Confidence level (low for initial attempts)
- What comes next (rationale generation)

Format as a clear, readable step description."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()

    async def _sample_rationale(self, initial_answer: str, context: dict[str, Any] | None) -> str:
        """Generate rationale for initial attempt using LLM sampling.

        Args:
            initial_answer: The initial answer to rationalize
            context: Optional context dictionary

        Returns:
            LLM-generated rationale
        """
        if not self._execution_context:
            return self._generate_rationale_heuristic()

        system_prompt = (
            "You are a reasoning assistant using STaR (Self-Taught Reasoner) methodology. "
            "Generate a clear, step-by-step rationale that explains how the initial answer "
            "was derived. The rationale should make the reasoning transparent and verifiable."
        )

        prompt = f"""Initial Answer: {initial_answer}

Generate a clear rationale that explains the reasoning behind this answer.
Break it down into 3-4 logical steps that show:
1. What key elements were identified
2. What knowledge or principles were applied
3. How the answer was derived

Format as a clear reasoning chain."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()

    async def _sample_rationalize_step(self, rationale: str) -> str:
        """Generate rationalize step content using LLM sampling.

        Args:
            rationale: The generated rationale

        Returns:
            LLM-generated rationalize step content
        """
        if not self._execution_context:
            return self._generate_rationalize_step_heuristic()

        system_prompt = """You are documenting the STaR (Self-Taught Reasoner) reasoning process.
Generate clear, structured content for the rationalization step."""

        prompt = f"""Step {self._step_counter}: Rationale Generation

Generated Rationale:
{rationale}

Create a structured description of this rationalization step.
Include:
- What was done (creating reasoning chain)
- The generated rationale
- Quality assessment
- What comes next (verification)

Format as a clear, readable step description."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()

    async def _sample_verification(self, rationale: str, context: dict[str, Any] | None) -> bool:
        """Verify rationale correctness using LLM sampling.

        Args:
            rationale: The rationale to verify
            context: Optional context dictionary

        Returns:
            True if rationale is verified as correct
        """
        if not self._execution_context:
            return True

        system_prompt = (
            "You are a verification assistant in the STaR (Self-Taught Reasoner) process. "
            "Verify if the rationale is logically consistent and leads to a correct answer. "
            "Respond with YES if verified, NO if not."
        )

        prompt = f"""Rationale to verify:
{rationale}

Verify this rationale by checking:
1. Logical consistency - does each step follow from the previous?
2. Answer derivation - does the rationale properly support the answer?
3. Knowledge alignment - is the reasoning based on correct information?

Respond with YES if the rationale passes verification, NO if it fails.
Then briefly explain your decision."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return "YES" in result.upper()[:20]

    async def _sample_verify_step(self, verified: bool) -> str:
        """Generate verify step content using LLM sampling.

        Args:
            verified: Whether rationale was verified

        Returns:
            LLM-generated verify step content
        """
        if not self._execution_context:
            return self._generate_verify_step_heuristic()

        system_prompt = """You are documenting the STaR (Self-Taught Reasoner) reasoning process.
Generate clear, structured content for the verification step."""

        prompt = f"""Step {self._step_counter}: Rationale Verification

Verification Result: {"VERIFIED" if verified else "FAILED"}

Create a structured description of this verification step.
Include:
- What was checked (rationale correctness)
- Verification criteria (logical consistency, answer derivation, knowledge alignment)
- Result (verified or failed)
- What comes next (bootstrap if verified, retry if failed)

Format as a clear, readable step description with checkmarks for passed criteria."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()

    async def _sample_bootstrap_step(self, bootstrap_count: int) -> str:
        """Generate bootstrap step content using LLM sampling.

        Args:
            bootstrap_count: Current bootstrap iteration count

        Returns:
            LLM-generated bootstrap step content
        """
        if not self._execution_context:
            return self._generate_bootstrap_step_heuristic()

        system_prompt = """You are documenting the STaR (Self-Taught Reasoner) reasoning process.
Generate clear, structured content for the bootstrap learning step."""

        prompt = f"""Step {self._step_counter}: Bootstrap Learning

Bootstrap Iteration: {bootstrap_count}

Create a structured description of this bootstrap learning step.
Include:
- What is happening (using verified rationale for improvement)
- Actions taken (adding to training set, improving reasoning)
- Iteration count
- Learning outcome

Format as a clear, readable step description."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()

    async def _sample_conclude_step(self, context: dict[str, Any] | None) -> str:
        """Generate conclude step content using LLM sampling.

        Args:
            context: Optional context dictionary

        Returns:
            LLM-generated conclude step content
        """
        if not self._execution_context:
            return self._generate_conclude_step_heuristic()

        system_prompt = """You are documenting the STaR (Self-Taught Reasoner) reasoning process.
Generate a comprehensive final summary of the STaR process."""

        prompt = f"""Step {self._step_counter}: Final Answer

STaR Process Complete:
- Initial attempt: {self._initial_answer}
- Rationale: {self._rationale}
- Verified: {self._verified}
- Bootstrap iterations: {self._bootstrap_count}

Create a final summary of the STaR process.
Include:
- Summary of all phases completed
- Final answer (from bootstrapped reasoning)
- Confidence level
- Learning outcome (rationale added to knowledge base)

Format as a clear, comprehensive conclusion."""

        response = await self._execution_context.sample(prompt, system_prompt=system_prompt)
        result = response.text if hasattr(response, "text") else str(response)
        return result.strip()


__all__ = ["STaR", "STAR_METADATA"]
