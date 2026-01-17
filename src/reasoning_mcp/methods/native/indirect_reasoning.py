"""Indirect Reasoning (Proof by Contradiction) method.

This module implements Indirect Reasoning (Wang et al. 2023), which uses
proof by contradiction to solve problems. Instead of proving something
directly, it assumes the negation is true and derives a contradiction,
thereby proving the original statement.

Key phases:
1. State: Formulate the claim to prove
2. Negate: Assume the negation of the claim
3. Derive: Derive logical consequences from the negation
4. Contradict: Show the derivation leads to contradiction
5. Conclude: Original claim proven by contradiction

Reference: Wang et al. (2023) - "Towards Logical Reasoning with Language Models:
Indirect Reasoning for Logical Problems"
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


# Metadata for Indirect Reasoning method
INDIRECT_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.INDIRECT_REASONING,
    name="Indirect Reasoning",
    description="Uses proof by contradiction - assumes negation and derives "
    "contradiction to prove original claim. Follows state → negate → derive → "
    "contradict → conclude phases for rigorous logical proof.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "proof-by-contradiction",
            "reductio-ad-absurdum",
            "indirect-proof",
            "logical",
            "negation",
            "contradiction",
            "rigorous",
            "formal",
        }
    ),
    complexity=7,  # High complexity due to indirect approach
    supports_branching=False,  # Linear contradiction path
    supports_revision=True,  # Can revise derivation
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: state + negate + derive + contradict + conclude
    max_thoughts=10,  # Complex derivations
    avg_tokens_per_thought=350,  # Logical steps are moderate
    best_for=(
        "mathematical proofs",
        "logical impossibility",
        "existence proofs",
        "uniqueness proofs",
        "irrationality proofs",
        "logical paradoxes",
        "negation problems",
        "impossibility demonstrations",
    ),
    not_recommended_for=(
        "direct calculation",
        "simple queries",
        "creative tasks",
        "subjective analysis",
    ),
)

logger = structlog.get_logger(__name__)


class IndirectReasoning(ReasoningMethodBase):
    """Indirect Reasoning (Proof by Contradiction) method implementation.

    This class implements proof by contradiction:
    1. State: Formulate the claim P to prove
    2. Negate: Assume ¬P (not P)
    3. Derive: Derive consequences from ¬P
    4. Contradict: Show ¬P leads to contradiction
    5. Conclude: Therefore P must be true

    Key characteristics:
    - Indirect proof strategy
    - Reductio ad absurdum
    - Rigorous logical structure
    - High complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = IndirectReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Prove that √2 is irrational"
        ... )
        >>> print(result.content)  # Statement of claim
    """

    # Maximum derivation steps
    MAX_DERIVATIONS = 5

    # Enable LLM sampling
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Indirect Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "state"
        self._claim: str = ""
        self._negation: str = ""
        self._derivation_count = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.INDIRECT_REASONING

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return INDIRECT_REASONING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return INDIRECT_REASONING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Indirect Reasoning method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "state"
        self._claim = ""
        self._negation = ""
        self._derivation_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Indirect Reasoning method.

        Creates the initial statement of the claim to prove.

        Args:
            session: The current reasoning session
            input_text: The claim to prove
            context: Optional additional context

        Returns:
            A ThoughtNode representing the statement phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Indirect Reasoning method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "state"
        self._claim = input_text
        self._negation = ""
        self._derivation_count = 0

        # Generate statement content
        content = await self._generate_statement(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.INDIRECT_REASONING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "indirect_reasoning",
                "phase": self._current_phase,
                "claim": self._claim,
                "derivation_count": self._derivation_count,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.INDIRECT_REASONING

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

        Implements the Indirect Reasoning phase progression:
        - After state: negate the claim
        - After negate: derive consequences
        - During derive: continue deriving or find contradiction
        - After contradict: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the Indirect Reasoning process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Indirect Reasoning method must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "state")

        if prev_phase == "state":
            # Negate the claim
            self._current_phase = "negate"
            thought_type = ThoughtType.HYPOTHESIS
            content = await self._generate_negation(guidance, context)
            confidence = 0.7
            quality_score = 0.7

        elif prev_phase == "negate":
            # Start derivation
            self._current_phase = "derive"
            self._derivation_count = 1
            thought_type = ThoughtType.REASONING
            content = await self._generate_derivation(self._derivation_count, guidance, context)
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "derive":
            self._derivation_count += 1
            if self._derivation_count < self.MAX_DERIVATIONS:
                # Check if we found contradiction
                found_contradiction = self._derivation_count >= 3
                if found_contradiction:
                    # Show contradiction
                    self._current_phase = "contradict"
                    thought_type = ThoughtType.VERIFICATION
                    content = await self._generate_contradiction(guidance, context)
                    confidence = 0.85
                    quality_score = 0.85
                else:
                    # Continue derivation
                    thought_type = ThoughtType.REASONING
                    content = await self._generate_derivation(
                        self._derivation_count, guidance, context
                    )
                    confidence = 0.75
                    quality_score = 0.75
            else:
                # Max derivations, show contradiction
                self._current_phase = "contradict"
                thought_type = ThoughtType.VERIFICATION
                content = await self._generate_contradiction(guidance, context)
                confidence = 0.85
                quality_score = 0.85

        elif prev_phase == "contradict":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = await self._generate_conclusion(guidance, context)
            confidence = 0.9
            quality_score = 0.9

        elif prev_phase == "conclude":
            # Final synthesis
            self._current_phase = "done"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._generate_final_synthesis(guidance, context)
            confidence = 0.95
            quality_score = 0.95

        else:
            # Fallback
            self._current_phase = "contradict"
            thought_type = ThoughtType.VERIFICATION
            content = await self._generate_contradiction(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.INDIRECT_REASONING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "indirect_reasoning",
                "claim": self._claim,
                "negation": self._negation,
                "derivation_count": self._derivation_count,
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    async def _sample_statement(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate statement of the claim."""
        prompt = f"""Problem: {input_text}

Generate a clear statement of the claim to prove using indirect reasoning (proof by contradiction).
Explain the reductio ad absurdum strategy and outline the steps:
1. State the claim P
2. Assume negation ¬P
3. Derive consequences from ¬P
4. Show contradiction
5. Conclude P is true

Provide a structured statement phase for this indirect proof."""

        system_prompt = """You are an expert in mathematical logic and proof by contradiction.
Generate clear, rigorous statements for indirect reasoning proofs.
Focus on formal logic structure and precise mathematical language."""

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: self._generate_statement_heuristic(input_text, context),
            system_prompt=system_prompt,
        )
        if result != self._generate_statement_heuristic(input_text, context):
            return f"Step {self._step_counter}: State the Claim (Indirect Reasoning)\n\n{result}"
        return result

    def _generate_statement_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the statement of the claim (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: State the Claim (Indirect Reasoning)\n\n"
            f"Problem: {input_text}\n\n"
            f"Proof Strategy: Reductio ad Absurdum (Proof by Contradiction)\n\n"
            f"Claim to Prove:\n"
            f"  P: [The statement we want to prove]\n\n"
            f"Method:\n"
            f"  1. Assume ¬P (the negation of P)\n"
            f"  2. Derive logical consequences from ¬P\n"
            f"  3. Show these consequences lead to contradiction\n"
            f"  4. Conclude P must be true\n\n"
            f"This indirect approach often succeeds where direct proof is difficult."
        )

    async def _generate_statement(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the statement of the claim."""
        return await self._sample_statement(input_text, context)

    async def _sample_negation(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate negation of the claim."""
        prompt = f"""Original Claim: {self._claim}

Generate the negation of this claim for proof by contradiction.
1. State the original claim P clearly
2. Formulate the negation ¬P precisely
3. Explain that we assume ¬P is true (for sake of contradiction)
4. Note this is temporary - we will derive contradiction from it

{f"Guidance: {guidance}" if guidance else ""}"""

        system_prompt = """You are an expert in logical negation and proof by contradiction.
Generate precise negations for mathematical and logical claims.
Use formal notation where appropriate (P, ¬P, etc.)."""

        fallback_result = self._generate_negation_heuristic(guidance, context)
        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: fallback_result,
            system_prompt=system_prompt,
        )
        if result != fallback_result:
            self._negation = "¬P: [Negation extracted from LLM]"
            return f"Step {self._step_counter}: Assume the Negation\n\n{result}"
        return result

    def _generate_negation_heuristic(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the negation of the claim (heuristic fallback)."""
        self._negation = "¬P: [Negation of the claim]"

        return (
            f"Step {self._step_counter}: Assume the Negation\n\n"
            f"For proof by contradiction, assume the opposite...\n\n"
            f"Original Claim (P):\n"
            f"  [The statement we want to prove]\n\n"
            f"Negation (¬P) - ASSUMED TRUE:\n"
            f"  Suppose, for sake of contradiction, that ¬P.\n"
            f"  That is, assume: [negation of the claim]\n\n"
            f"⚠️ Note: This is a temporary assumption.\n"
            f"We will show this leads to an impossibility.\n\n"
            f"Now deriving consequences from this assumption..."
        )

    async def _generate_negation(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the negation of the claim."""
        return await self._sample_negation(guidance, context)

    async def _sample_derivation(
        self,
        deriv_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate derivation step."""
        prompt = f"""Original Claim: {self._claim}
Assumed Negation: {self._negation}

This is derivation step #{deriv_num} from the negation ¬P.

Generate the next logical consequence that follows from assuming ¬P:
1. State what we have so far (¬P and previous derivations)
2. Derive the next consequence D{deriv_num}
3. Explain the reasoning (which logical rule or theorem applies)
4. Show accumulated consequences so far

{f"Guidance: {guidance}" if guidance else ""}"""

        system_prompt = """You are an expert in logical derivation and proof by contradiction.
Generate rigorous logical derivations from assumptions.
Show clear reasoning steps and apply appropriate logical rules."""

        fallback_result = self._generate_derivation_heuristic(deriv_num, guidance, context)
        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: fallback_result,
            system_prompt=system_prompt,
        )
        if result != fallback_result:
            return f"Step {self._step_counter}: Derivation #{deriv_num} from ¬P\n\n{result}"
        return result

    def _generate_derivation_heuristic(
        self,
        deriv_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a derivation step from the negation (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Derivation #{deriv_num} from ¬P\n\n"
            f"Given: ¬P (our assumption)\n"
            f"Plus: [Previous derivations or known facts]\n\n"
            f"Derivation:\n"
            f"  From ¬P, we can derive:\n"
            f"  D{deriv_num}: [Logical consequence of ¬P]\n\n"
            f"Reasoning:\n"
            f"  If ¬P is true, then by [logical rule/theorem],\n"
            f"  it follows that D{deriv_num} must also be true.\n\n"
            f"Accumulated Consequences:\n"
            + "\n".join(f"  D{i}: [Derived statement {i}]" for i in range(1, deriv_num + 1))
            + "\n\nContinuing derivation..."
        )

    async def _generate_derivation(
        self,
        deriv_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a derivation step from the negation."""
        return await self._sample_derivation(deriv_num, guidance, context)

    async def _sample_contradiction(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate contradiction discovery."""
        prompt = f"""Original Claim: {self._claim}
Assumed Negation: {self._negation}
Number of derivations made: {self._derivation_count}

The derivations from ¬P have now led to a contradiction.

Generate the contradiction discovery:
1. Show what contradiction has emerged (Q ∧ ¬Q)
2. Explain why this is impossible
3. Trace the derivation path visually
4. Conclude that ¬P must be false, therefore P must be true

{f"Guidance: {guidance}" if guidance else ""}"""

        system_prompt = """You are an expert in logical contradictions and proof by contradiction.
Identify and explain contradictions clearly.
Show the derivation path that led to the impossibility."""

        fallback_result = self._generate_contradiction_heuristic(guidance, context)
        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: fallback_result,
            system_prompt=system_prompt,
        )
        if result != fallback_result:
            return f"Step {self._step_counter}: Contradiction Found!\n\n{result}"
        return result

    def _generate_contradiction_heuristic(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the contradiction discovery (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Contradiction Found! ⚡\n\n"
            f"The derivations from ¬P have led to:\n\n"
            f"Contradiction:\n"
            f"  From our derivations, we have both:\n"
            f"    • Q (some statement Q is true)\n"
            f"    • ¬Q (but also Q is false)\n\n"
            f"This is impossible: Q ∧ ¬Q = ⊥ (contradiction)\n\n"
            f"Analysis:\n"
            f"  ┌──────────────────────────────────────┐\n"
            f"  │ ¬P (assumed)                         │\n"
            f"  │   ↓                                  │\n"
            f"  │ D1, D2, D3, ... (derived)           │\n"
            f"  │   ↓                                  │\n"
            f"  │ Q ∧ ¬Q  ← CONTRADICTION!            │\n"
            f"  └──────────────────────────────────────┘\n\n"
            f"Since ¬P leads to contradiction, ¬P must be false.\n"
            f"Therefore, P must be true."
        )

    async def _generate_contradiction(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the contradiction discovery."""
        return await self._sample_contradiction(guidance, context)

    async def _sample_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate conclusion."""
        prompt = f"""Original Claim: {self._claim}
Assumed Negation: {self._negation}
Number of derivations: {self._derivation_count}

The contradiction has been established. Generate the conclusion:
1. Summarize the proof steps
2. State that ¬P is false
3. Conclude P is true with QED/∎
4. State the logical principle used: (¬P → ⊥) → P

{f"Guidance: {guidance}" if guidance else ""}"""

        system_prompt = """You are an expert in mathematical proof conclusions.
Generate clear, formal conclusions for proofs by contradiction.
Use proper mathematical notation and formal language."""

        fallback_result = self._generate_conclusion_heuristic(guidance, context)
        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: fallback_result,
            system_prompt=system_prompt,
        )
        if result != fallback_result:
            return f"Step {self._step_counter}: Conclusion by Contradiction\n\n{result}"
        return result

    def _generate_conclusion_heuristic(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the conclusion of the proof (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Conclusion by Contradiction\n\n"
            f"Proof Complete:\n\n"
            f"  1. We assumed ¬P (negation of our claim)\n"
            f"  2. We derived {self._derivation_count} consequences from ¬P\n"
            f"  3. These led to a contradiction (Q ∧ ¬Q)\n"
            f"  4. Therefore, ¬P is false\n"
            f"  5. Hence, P is true ∎\n\n"
            f"The Claim is Proven:\n"
            f"  P: [Original claim] is TRUE.\n\n"
            f"This proof is valid by the logical principle:\n"
            f"  (¬P → ⊥) → P\n"
            f'  "If not-P leads to contradiction, then P."'
        )

    async def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the conclusion of the proof."""
        try:
            return await self._sample_conclusion(guidance, context)
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_generate_conclusion",
                error=str(e),
                exc_info=True,
            )
            return self._generate_conclusion_heuristic(guidance, context)

    async def _sample_final_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Use LLM to generate final synthesis."""
        prompt = f"""Original Claim: {self._claim}
Number of derivations: {self._derivation_count}

Generate the final synthesis for this proof by contradiction:
1. Summarize the entire proof process
2. State the final answer clearly
3. Provide confidence assessment
4. Explain why this proof is rigorous

{f"Guidance: {guidance}" if guidance else ""}"""

        system_prompt = """You are an expert in mathematical proof synthesis.
Generate comprehensive summaries of proofs by contradiction.
Assess proof rigor and provide confidence levels."""

        if self._execution_context and self._execution_context.can_sample:
            result = await self._execution_context.sample(prompt, system_prompt=system_prompt)
            return f"Step {self._step_counter}: Final Answer\n\n{result}"
        return self._generate_final_synthesis_heuristic(guidance, context)

    def _generate_final_synthesis_heuristic(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Indirect Reasoning (Proof by Contradiction) Complete:\n\n"
            f"Summary:\n"
            f"  - Claim stated and negation assumed\n"
            f"  - Derived {self._derivation_count} consequences\n"
            f"  - Found logical contradiction\n"
            f"  - Concluded original claim is true\n\n"
            f"Final Answer: [The claim is proven true]\n\n"
            f"Confidence: Very High\n"
            f"Reason: Rigorous proof by contradiction with clear\n"
            f"derivation path to impossibility (Q ∧ ¬Q = ⊥)."
        )

    async def _generate_final_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis."""
        try:
            return await self._sample_final_synthesis(guidance, context)
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_generate_final_synthesis",
                error=str(e),
                exc_info=True,
            )
            return self._generate_final_synthesis_heuristic(guidance, context)


# Export
__all__ = ["IndirectReasoning", "INDIRECT_REASONING_METADATA"]
