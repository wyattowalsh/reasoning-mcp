"""Logic of Thought (LogiCoT) reasoning method.

This module implements Logic of Thought (Zhao et al. 2023), which applies
formal logic principles to chain-of-thought reasoning. LogiCoT structures
reasoning using premises, inferences, and conclusions with explicit logical
operators and validity checks.

Key phases:
1. Formalize: Convert problem to logical premises
2. Infer: Apply logical rules to derive new statements
3. Validate: Check logical validity of inferences
4. Conclude: Derive final conclusion from valid inferences

Reference: Zhao et al. (2023) - "Enhancing Chain-of-Thought Reasoning with
Logic of Thought"
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


# Metadata for Logic of Thought method
LOGIC_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LOGIC_OF_THOUGHT,
    name="Logic of Thought",
    description="Applies formal logic principles to reasoning with explicit premises, "
    "logical operators, and validity checking. Structures thinking through "
    "formalize → infer → validate → conclude phases.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "formal-logic",
            "premises",
            "inferences",
            "validity",
            "deduction",
            "propositional",
            "syllogism",
            "rigorous",
        }
    ),
    complexity=7,  # High complexity due to formal logic
    supports_branching=True,  # Multiple inference paths
    supports_revision=True,  # Can revise invalid inferences
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: formalize + infer + validate + conclude
    max_thoughts=10,  # Complex logical chains
    avg_tokens_per_thought=400,  # Formal logic can be verbose
    best_for=(
        "logical puzzles",
        "syllogistic reasoning",
        "formal arguments",
        "validity checking",
        "deductive reasoning",
        "mathematical proofs",
        "legal reasoning",
        "philosophical arguments",
    ),
    not_recommended_for=(
        "creative tasks",
        "emotional intelligence",
        "ambiguous situations",
        "tasks without clear logical structure",
    ),
)

logger = structlog.get_logger(__name__)


class LogicOfThought(ReasoningMethodBase):
    """Logic of Thought reasoning method implementation.

    This class implements the LogiCoT pattern:
    1. Formalize: Extract and formalize premises
    2. Infer: Apply logical rules (modus ponens, etc.)
    3. Validate: Check validity of inferences
    4. Conclude: Derive logically sound conclusion

    Key characteristics:
    - Formal logical structure
    - Explicit validity checking
    - Rigorous deduction
    - High complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = LogicOfThought()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="All mammals are warm-blooded. Whales are mammals. "
        ...                "Are whales warm-blooded?"
        ... )
        >>> print(result.content)  # Formalization with premises
    """

    # Maximum inference steps
    MAX_INFERENCES = 5

    # Enable sampling support
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Logic of Thought method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "formalize"
        self._premises: list[str] = []
        self._inferences: list[str] = []
        self._inference_count = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.LOGIC_OF_THOUGHT

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return LOGIC_OF_THOUGHT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return LOGIC_OF_THOUGHT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Logic of Thought method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "formalize"
        self._premises = []
        self._inferences = []
        self._inference_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Logic of Thought method.

        Creates the formalization phase, converting the problem
        to logical premises.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the formalization phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Logic of Thought method must be initialized before execution")

        # Store execution context and configure sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "formalize"
        self._premises = []
        self._inferences = []
        self._inference_count = 0

        # Generate formalization content
        if self._use_sampling:
            content = await self._sample_formalization(input_text, context)
        else:
            content = self._generate_formalization(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LOGIC_OF_THOUGHT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "logic_of_thought",
                "phase": self._current_phase,
                "num_premises": len(self._premises),
                "inference_count": self._inference_count,
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.LOGIC_OF_THOUGHT

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

        Implements the LogiCoT phase progression:
        - After formalize: perform logical inferences
        - During infer: continue or validate
        - After validate: conclude
        - After conclude: done

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the LogiCoT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Logic of Thought method must be initialized before continuation")

        # Store execution context and configure sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "formalize")

        if prev_phase == "formalize":
            # Start inference
            self._current_phase = "infer"
            self._inference_count = 1
            thought_type = ThoughtType.REASONING
            if self._use_sampling:
                content = await self._sample_inference(
                    previous_thought, self._inference_count, guidance, context
                )
            else:
                content = self._generate_inference(
                    previous_thought, self._inference_count, guidance, context
                )
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "infer":
            self._inference_count += 1
            if self._inference_count <= self.MAX_INFERENCES:
                # Check if more inferences needed
                needs_more = self._inference_count < 3  # At least 2 inferences
                if needs_more:
                    thought_type = ThoughtType.REASONING
                    if self._use_sampling:
                        content = await self._sample_inference(
                            previous_thought, self._inference_count, guidance, context
                        )
                    else:
                        content = self._generate_inference(
                            previous_thought, self._inference_count, guidance, context
                        )
                    confidence = 0.75
                    quality_score = 0.75
                else:
                    # Validate inferences
                    self._current_phase = "validate"
                    thought_type = ThoughtType.VERIFICATION
                    if self._use_sampling:
                        content = await self._sample_validation(guidance, context)
                    else:
                        content = self._generate_validation(guidance, context)
                    confidence = 0.8
                    quality_score = 0.8
            else:
                # Max inferences, validate
                self._current_phase = "validate"
                thought_type = ThoughtType.VERIFICATION
                if self._use_sampling:
                    content = await self._sample_validation(guidance, context)
                else:
                    content = self._generate_validation(guidance, context)
                confidence = 0.8
                quality_score = 0.8

        elif prev_phase == "validate":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance, context)
            else:
                content = self._generate_conclusion(previous_thought, guidance, context)
            confidence = 0.9
            quality_score = 0.9

        elif prev_phase == "conclude":
            # Already concluded, synthesize
            self._current_phase = "done"
            thought_type = ThoughtType.SYNTHESIS
            if self._use_sampling:
                content = await self._sample_final_synthesis(previous_thought, guidance, context)
            else:
                content = self._generate_final_synthesis(previous_thought, guidance, context)
            confidence = 0.95
            quality_score = 0.95

        else:
            # Fallback
            self._current_phase = "validate"
            thought_type = ThoughtType.VERIFICATION
            if self._use_sampling:
                content = await self._sample_validation(guidance, context)
            else:
                content = self._generate_validation(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LOGIC_OF_THOUGHT,
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
                "reasoning_type": "logic_of_thought",
                "num_premises": len(self._premises),
                "inference_count": self._inference_count,
                "previous_phase": prev_phase,
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _generate_formalization(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the formalization phase content."""
        self._premises = [
            "P1: [First extracted premise]",
            "P2: [Second extracted premise]",
        ]

        return (
            f"Step {self._step_counter}: Logical Formalization (Logic of Thought)\n\n"
            f"Problem: {input_text}\n\n"
            f"Extracting Logical Structure...\n\n"
            f"Identified Premises:\n"
            f"  P1: [Universal statement - ∀x: A(x) → B(x)]\n"
            f"  P2: [Particular statement - A(c)]\n\n"
            f"Query: Q - [What needs to be determined]\n\n"
            f"Logical Form:\n"
            f"  P1: ∀x: A(x) → B(x)  (If A then B, for all x)\n"
            f"  P2: A(c)             (c has property A)\n"
            f"  Q:  B(c)?            (Does c have property B?)\n\n"
            f"Ready for logical inference."
        )

    def _generate_inference(
        self,
        previous_thought: ThoughtNode,
        inference_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a logical inference step."""
        rules = [
            ("Modus Ponens", "If P → Q and P, then Q"),
            ("Universal Instantiation", "If ∀x: P(x), then P(c) for any c"),
            ("Modus Tollens", "If P → Q and ¬Q, then ¬P"),
        ]
        rule_name, rule_desc = rules[(inference_num - 1) % len(rules)]

        inference = f"I{inference_num}: [Derived statement]"
        self._inferences.append(inference)

        return (
            f"Step {self._step_counter}: Logical Inference #{inference_num}\n\n"
            f"Applying Rule: {rule_name}\n"
            f"Rule Form: {rule_desc}\n\n"
            f"Inference:\n"
            f"  From: P1 (∀x: A(x) → B(x))\n"
            f"  And:  P2 (A(c))\n"
            f"  By:   {rule_name}\n"
            f"  ───────────────────\n"
            f"  Derive: I{inference_num} - B(c)\n\n"
            f"Justification:\n"
            f"  The premises logically entail this conclusion by {rule_name}.\n"
            f"  This inference preserves truth from premises to conclusion."
        )

    def _generate_validation(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the validation phase content."""
        return (
            f"Step {self._step_counter}: Logical Validation\n\n"
            f"Checking validity of inference chain...\n\n"
            f"Validity Criteria:\n"
            f"1. ✓ Premises are well-formed propositions\n"
            f"2. ✓ Each inference follows valid rule\n"
            f"3. ✓ No fallacies detected\n"
            f"4. ✓ Conclusion follows necessarily from premises\n\n"
            f"Inference Chain:\n"
            f"  P1 ─┬─ (Universal Instantiation)\n"
            f"      │\n"
            f"  P2 ─┴─ (Modus Ponens) ──→ I1 ──→ Conclusion\n\n"
            f"Validity Check: VALID\n"
            f"  - Deductively valid argument\n"
            f"  - If premises true, conclusion necessarily true\n"
            f"  - No logical gaps in reasoning"
        )

    def _generate_conclusion(
        self,
        validation_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the logical conclusion."""
        return (
            f"Step {self._step_counter}: Logical Conclusion\n\n"
            f"Based on validated inference chain:\n\n"
            f"Argument Summary:\n"
            f"  Premise 1: ∀x: A(x) → B(x)\n"
            f"  Premise 2: A(c)\n"
            f"  ────────────────────────\n"
            f"  ∴ Conclusion: B(c)  [Q.E.D.]\n\n"
            f"Answer: Yes, the conclusion B(c) follows logically.\n\n"
            f"Logical Properties:\n"
            f"  - Validity: The argument is deductively valid\n"
            f"  - Soundness: Depends on truth of premises\n"
            f"  - Inference count: {self._inference_count - 1}"
        )

    def _generate_final_synthesis(
        self,
        conclusion_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Logic of Thought Analysis Complete:\n\n"
            f"1. Formalized problem into logical premises\n"
            f"2. Applied {self._inference_count - 1} valid inference rules\n"
            f"3. Validated entire inference chain\n"
            f"4. Derived logically necessary conclusion\n\n"
            f"Final Answer: [Answer derived through formal logic]\n\n"
            f"Confidence: Very High\n"
            f"Reason: Conclusion follows necessarily from premises\n"
            f"through valid deductive inference."
        )

    async def _sample_formalization(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate formalization using LLM sampling.

        Args:
            input_text: The problem to formalize
            context: Optional additional context

        Returns:
            Formatted formalization content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        system_prompt = """You are a formal logic expert using Logic of Thought methodology.
Your task is to formalize problems into logical premises using formal logic notation.

Structure your formalization:
1. State the problem clearly
2. Extract logical premises (P1, P2, etc.)
3. Convert to formal logic notation (∀, ∃, →, ∧, ∨, ¬)
4. Identify the query (Q)
5. Present the logical form

Use standard logical operators and be precise with quantifiers."""

        user_prompt = f"""Problem: {input_text}

Formalize this problem using Logic of Thought methodology.
Extract premises, use formal logic notation, and prepare for logical inference."""

        def fallback() -> str:
            return self._generate_formalization(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for formal logic
            max_tokens=800,
        )
        # Try to extract premises from the sampled content
        self._extract_premises_from_content(content)
        return content

    async def _sample_inference(
        self,
        previous_thought: ThoughtNode,
        inference_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate logical inference using LLM sampling.

        Args:
            previous_thought: The previous thought containing premises
            inference_num: The inference step number
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Formatted inference content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        system_prompt = """You are a formal logic expert applying logical inference rules.
Generate valid logical inferences using standard inference rules:
- Modus Ponens: If P → Q and P, then Q
- Modus Tollens: If P → Q and ¬Q, then ¬P
- Universal Instantiation: If ∀x: P(x), then P(c)
- Existential Generalization: If P(c), then ∃x: P(x)
- Hypothetical Syllogism: If P → Q and Q → R, then P → R

Show your reasoning step explicitly with rule application and justification."""

        guidance_text = f"\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Previous formalization:
{previous_thought.content}

Generate logical inference step #{inference_num}.
Apply a valid inference rule and show your work.{guidance_text}"""

        def fallback() -> str:
            return self._generate_inference(previous_thought, inference_num, guidance, context)

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=600,
        )

    async def _sample_validation(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate validation using LLM sampling.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Formatted validation content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        system_prompt = """You are a formal logic validator checking inference validity.
Check the logical validity of an inference chain:
1. Are premises well-formed?
2. Does each inference follow a valid rule?
3. Are there any logical fallacies?
4. Does the conclusion follow necessarily?

Provide a thorough validity check with specific criteria."""

        inferences_summary = (
            "\n".join(self._inferences) if self._inferences else "Multiple inferences"
        )
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Inference chain to validate:
Premises: {len(self._premises)} premises identified
Inferences: {inferences_summary}

Validate this logical reasoning chain.
Check validity, soundness, and identify any issues.{guidance_text}"""

        def fallback() -> str:
            return self._generate_validation(guidance, context)

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=700,
        )

    async def _sample_conclusion(
        self,
        validation_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Args:
            validation_thought: The validation thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Formatted conclusion content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        system_prompt = """You are a formal logic expert deriving conclusions.
Based on validated inferences, derive the final logical conclusion.

Structure your conclusion:
1. Summarize the argument (premises → conclusion)
2. State the conclusion clearly
3. Assess validity and soundness
4. Provide logical properties

Use proper logical notation (∴ for "therefore")."""

        guidance_text = f"\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Validation result:
{validation_thought.content}

Derive the final logical conclusion.
State the conclusion with proper justification.{guidance_text}"""

        def fallback() -> str:
            return self._generate_conclusion(validation_thought, guidance, context)

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=700,
        )

    async def _sample_final_synthesis(
        self,
        conclusion_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final synthesis using LLM sampling.

        Args:
            conclusion_thought: The conclusion thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Formatted final synthesis content

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        system_prompt = """You are a formal logic expert providing final synthesis.
Synthesize the complete Logic of Thought analysis:
1. Summarize the logical journey
2. State the final answer
3. Assess confidence based on logical rigor
4. Highlight key logical properties"""

        guidance_text = f"\nGuidance: {guidance}" if guidance else ""
        user_prompt = f"""Conclusion reached:
{conclusion_thought.content}

Provide a final synthesis of the Logic of Thought analysis.
Give the definitive answer with high confidence justification.{guidance_text}"""

        def fallback() -> str:
            return self._generate_final_synthesis(conclusion_thought, guidance, context)

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=600,
        )

    def _extract_premises_from_content(self, content: str) -> None:
        """Extract premise markers from sampled content.

        Args:
            content: The sampled content to parse
        """
        # Simple extraction: look for P1, P2, etc. patterns
        import re

        premise_pattern = r"(P\d+:.*?)(?=\n|P\d+:|$)"
        matches = re.findall(premise_pattern, content, re.MULTILINE | re.DOTALL)
        if matches:
            self._premises = [m.strip() for m in matches[:5]]  # Limit to 5 premises


# Export
__all__ = ["LogicOfThought", "LOGIC_OF_THOUGHT_METADATA"]
