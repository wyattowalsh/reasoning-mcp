"""Contrastive Chain-of-Thought reasoning method.

This module implements Contrastive Chain-of-Thought (Chia et al. 2023), which
enhances reasoning by contrasting correct and incorrect reasoning paths. By
explicitly showing what NOT to do alongside what TO do, the model learns to
avoid common pitfalls and produce more accurate reasoning.

Key phases:
1. Generate: Create initial reasoning path
2. Contrast: Generate plausible but incorrect alternatives
3. Analyze: Compare correct vs incorrect approaches
4. Refine: Strengthen reasoning based on contrast

Reference: Chia et al. (2023) - "Contrastive Chain-of-Thought Prompting"
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


# Metadata for Contrastive Chain-of-Thought method
CONTRASTIVE_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CONTRASTIVE_COT,
    name="Contrastive Chain-of-Thought",
    description="Enhances reasoning by contrasting correct and incorrect paths. "
    "Explicitly shows what NOT to do alongside correct reasoning through "
    "generate → contrast → analyze → refine phases for improved accuracy.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "contrastive",
            "error-analysis",
            "negative-examples",
            "reasoning-improvement",
            "pitfall-avoidance",
            "accuracy",
            "comparison",
            "learning",
        }
    ),
    complexity=6,  # Moderate-high complexity
    supports_branching=True,  # Contrasting branches
    supports_revision=True,  # Refines based on contrast
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: generate + contrast + analyze + refine
    max_thoughts=8,  # Including multiple contrasts
    avg_tokens_per_thought=350,  # Comparisons can be verbose
    best_for=(
        "error-prone reasoning tasks",
        "mathematical problem solving",
        "logical deduction",
        "classification tasks",
        "disambiguation",
        "avoiding common mistakes",
        "high-stakes reasoning",
        "educational explanations",
    ),
    not_recommended_for=(
        "creative tasks",
        "subjective analysis",
        "tasks without clear right/wrong",
        "open-ended exploration",
    ),
)

logger = structlog.get_logger(__name__)


class ContrastiveCoT(ReasoningMethodBase):
    """Contrastive Chain-of-Thought reasoning method implementation.

    This class implements the Contrastive CoT pattern:
    1. Generate: Create an initial reasoning path
    2. Contrast: Generate plausible incorrect alternatives
    3. Analyze: Compare and identify differences
    4. Refine: Strengthen reasoning based on contrast

    Key characteristics:
    - Explicit error demonstration
    - Learning from negative examples
    - Improved accuracy through contrast
    - Moderate-high complexity (6)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = ContrastiveCoT()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="What is 15% of 80?"
        ... )
        >>> print(result.content)  # Generate phase with correct approach
    """

    # Maximum incorrect paths to generate
    MAX_CONTRASTS = 2

    def __init__(self) -> None:
        """Initialize the Contrastive Chain-of-Thought method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._correct_path: str = ""
        self._incorrect_paths: list[str] = []
        self._contrast_count = 0
        self._use_sampling: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.CONTRASTIVE_COT

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return CONTRASTIVE_COT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return CONTRASTIVE_COT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Contrastive CoT method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._correct_path = ""
        self._incorrect_paths = []
        self._contrast_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Contrastive Chain-of-Thought method.

        Creates the initial correct reasoning path.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the generation phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Contrastive CoT method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "generate"
        self._correct_path = ""
        self._incorrect_paths = []
        self._contrast_count = 0

        # Generate correct reasoning content (use sampling if available)
        if self._use_sampling:
            content = await self._sample_correct_path(input_text, context)
        else:
            content = self._generate_correct_path(input_text, context)
        self._correct_path = content

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CONTRASTIVE_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "contrastive_cot",
                "phase": self._current_phase,
                "path_type": "correct",
                "contrast_count": self._contrast_count,
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.CONTRASTIVE_COT

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

        Implements the Contrastive CoT phase progression:
        - After generate: create incorrect contrasting paths
        - After contrast: analyze differences
        - After analyze: refine and strengthen
        - After refine: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the Contrastive CoT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Contrastive CoT method must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            # Generate first incorrect path
            self._current_phase = "contrast"
            self._contrast_count = 1
            thought_type = ThoughtType.HYPOTHESIS
            if self._use_sampling:
                content = await self._sample_incorrect_path(
                    previous_thought, self._contrast_count, guidance, context
                )
            else:
                content = self._generate_incorrect_path(
                    previous_thought, self._contrast_count, guidance, context
                )
            self._incorrect_paths.append(content)
            confidence = 0.6  # Lower for intentionally wrong path
            quality_score = 0.65

        elif prev_phase == "contrast":
            if self._contrast_count < self.MAX_CONTRASTS:
                # Generate another incorrect path
                self._contrast_count += 1
                thought_type = ThoughtType.HYPOTHESIS
                if self._use_sampling:
                    content = await self._sample_incorrect_path(
                        previous_thought, self._contrast_count, guidance, context
                    )
                else:
                    content = self._generate_incorrect_path(
                        previous_thought, self._contrast_count, guidance, context
                    )
                self._incorrect_paths.append(content)
                confidence = 0.6
                quality_score = 0.65
            else:
                # Move to analysis
                self._current_phase = "analyze"
                thought_type = ThoughtType.REASONING
                if self._use_sampling:
                    content = await self._sample_analysis(guidance, context)
                else:
                    content = self._generate_analysis(guidance, context)
                confidence = 0.8
                quality_score = 0.8

        elif prev_phase == "analyze":
            # Refine based on contrast
            self._current_phase = "refine"
            thought_type = ThoughtType.SYNTHESIS
            if self._use_sampling:
                content = await self._sample_refinement(previous_thought, guidance, context)
            else:
                content = self._generate_refinement(previous_thought, guidance, context)
            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "refine":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance, context)
            else:
                content = self._generate_conclusion(previous_thought, guidance, context)
            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback
            self._current_phase = "refine"
            thought_type = ThoughtType.SYNTHESIS
            if self._use_sampling:
                content = await self._sample_refinement(previous_thought, guidance, context)
            else:
                content = self._generate_refinement(previous_thought, guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CONTRASTIVE_COT,
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
                "reasoning_type": "contrastive_cot",
                "path_type": "incorrect" if self._current_phase == "contrast" else "analysis",
                "contrast_count": self._contrast_count,
                "previous_phase": prev_phase,
                "sampled": self._use_sampling,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _generate_correct_path(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the correct reasoning path."""
        return (
            f"Step {self._step_counter}: Correct Reasoning Path (Contrastive CoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"✓ CORRECT APPROACH:\n"
            f"Step 1: [Identify what the problem is asking]\n"
            f"Step 2: [Apply appropriate method/formula]\n"
            f"Step 3: [Execute calculation/reasoning correctly]\n"
            f"Step 4: [Verify result makes sense]\n\n"
            f"Preliminary Answer: [correct answer with reasoning]\n\n"
            f"Next: Generate contrasting incorrect paths to strengthen understanding."
        )

    def _generate_incorrect_path(
        self,
        correct_thought: ThoughtNode,
        contrast_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an incorrect reasoning path for contrast."""
        error_types = [
            ("Misinterpretation Error", "misunderstood what was being asked"),
            ("Calculation Error", "applied wrong operation or formula"),
            ("Logic Error", "flawed reasoning step"),
        ]
        error_type, description = error_types[(contrast_num - 1) % len(error_types)]

        return (
            f"Step {self._step_counter}: Incorrect Path #{contrast_num} (Contrast)\n\n"
            f"✗ INCORRECT APPROACH ({error_type}):\n"
            f"This path {description}.\n\n"
            f"Flawed Reasoning:\n"
            f"Step 1: [Incorrect interpretation or setup]\n"
            f"Step 2: [Wrong method applied]\n"
            f"Step 3: [Error propagates]\n\n"
            f"Wrong Answer: [incorrect result]\n\n"
            f"Why This Is Wrong:\n"
            f"- [Specific error in reasoning]\n"
            f"- [What was overlooked or misapplied]\n"
            f"- [Common pitfall this represents]"
        )

    def _generate_analysis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the contrastive analysis."""
        return (
            f"Step {self._step_counter}: Contrastive Analysis\n\n"
            f"Comparing {self._contrast_count + 1} reasoning paths...\n\n"
            f"Path Comparison:\n"
            f"┌─────────────────────────────────────────────┐\n"
            f"│ Correct Path        vs  Incorrect Paths     │\n"
            f"├─────────────────────────────────────────────┤\n"
            f"│ ✓ Correct setup     │  ✗ Misinterpretation  │\n"
            f"│ ✓ Right method      │  ✗ Wrong operation    │\n"
            f"│ ✓ Valid logic       │  ✗ Flawed reasoning   │\n"
            f"│ ✓ Verified result   │  ✗ Unchecked answer   │\n"
            f"└─────────────────────────────────────────────┘\n\n"
            f"Key Differentiators:\n"
            f"1. [Critical difference that makes correct path right]\n"
            f"2. [Common mistake to avoid]\n"
            f"3. [Verification step that catches errors]"
        )

    def _generate_refinement(
        self,
        analysis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the refined reasoning."""
        return (
            f"Step {self._step_counter}: Refined Reasoning\n\n"
            f"Based on contrastive analysis, strengthening the correct path...\n\n"
            f"Refined Approach:\n"
            f"1. [Original correct reasoning, now fortified]\n"
            f"2. [Explicit avoidance of identified pitfalls]\n"
            f"3. [Additional verification steps]\n\n"
            f"Pitfalls Explicitly Avoided:\n"
            f"- ✗ Avoided: [Incorrect approach 1]\n"
            f"- ✗ Avoided: [Incorrect approach 2]\n\n"
            f"Confidence Boost:\n"
            f"- Reasoning is stronger after seeing what NOT to do\n"
            f"- Common errors explicitly addressed"
        )

    def _generate_conclusion(
        self,
        refine_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Contrastive Chain-of-Thought Analysis Complete:\n\n"
            f"1. Generated correct reasoning path\n"
            f"2. Created {self._contrast_count} contrasting incorrect paths\n"
            f"3. Analyzed differences between correct and incorrect\n"
            f"4. Refined reasoning by learning from errors\n\n"
            f"Final Answer: [answer with high confidence]\n\n"
            f"Confidence: Very High\n"
            f"Reason: Explicitly considered and rejected {self._contrast_count} "
            f"incorrect approaches, strengthening the correct solution."
        )

    async def _sample_correct_path(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the correct reasoning path using LLM sampling.

        Uses the execution context's sampling capability to generate
        an actual correct reasoning path with proper step-by-step analysis.

        Args:
            input_text: The problem to solve
            context: Optional additional context

        Returns:
            A formatted string containing the sampled correct reasoning path
        """
        system_prompt = """You are a reasoning assistant using Contrastive CoT methodology.
Generate a CORRECT reasoning path with clear step-by-step analysis.

Structure your response with:
1. Problem identification
2. Step-by-step correct approach
3. Proper method/formula application
4. Correct execution and calculation
5. Verification that the result makes sense
6. Preliminary answer with reasoning

Label it clearly as "✓ CORRECT APPROACH" and show why this approach is right.
Be explicit and thorough in your reasoning."""

        user_prompt = f"""Problem: {input_text}

Generate a CORRECT reasoning path following Contrastive CoT methodology.
This is the initial correct approach that will later be contrasted with incorrect alternatives.
Show clear step-by-step reasoning with proper verification."""

        def fallback() -> str:
            return self._generate_correct_path(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

        header = f"Step {self._step_counter}: Correct Reasoning Path (Contrastive CoT)\n\n"
        footer = "\n\nNext: Generate contrasting incorrect paths to strengthen understanding."
        return header + content + footer

    async def _sample_incorrect_path(
        self,
        correct_thought: ThoughtNode,
        contrast_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an incorrect reasoning path using LLM sampling.

        Creates a plausible but incorrect alternative approach for contrast.

        Args:
            correct_thought: The correct thought to contrast against
            contrast_num: The number of this contrasting path
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the sampled incorrect reasoning path
        """
        error_types = [
            ("Misinterpretation Error", "misunderstood what was being asked"),
            ("Calculation Error", "applied wrong operation or formula"),
            ("Logic Error", "flawed reasoning step"),
        ]
        error_type, description = error_types[(contrast_num - 1) % len(error_types)]

        system_prompt = f"""You are a reasoning assistant using Contrastive CoT methodology.
Generate an INCORRECT reasoning path that demonstrates a common pitfall.

Your task is to create a plausible but WRONG approach showing a {error_type}.
The path should:
1. Show incorrect interpretation or setup
2. Apply the wrong method or make calculation errors
3. Demonstrate how errors propagate
4. Arrive at an incorrect result
5. Explain WHY this approach is wrong and what pitfall it represents

Label it clearly as "✗ INCORRECT APPROACH ({error_type})" and be explicit about the mistakes."""

        user_prompt = f"""Correct reasoning path:
{correct_thought.content}

Generate an INCORRECT reasoning path demonstrating a {error_type}.
The error type should show that the approach {description}.
Make it plausible but clearly wrong, explaining the specific errors and common pitfalls."""

        def fallback() -> str:
            return self._generate_incorrect_path(correct_thought, contrast_num, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.8,  # Slightly higher for variety in error patterns
            max_tokens=1200,
        )

        header = f"Step {self._step_counter}: Incorrect Path #{contrast_num} (Contrast)\n\n"
        return header + content

    async def _sample_analysis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate contrastive analysis using LLM sampling.

        Compares and analyzes the differences between correct and incorrect paths.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the sampled contrastive analysis
        """
        system_prompt = """You are a reasoning assistant using Contrastive CoT methodology.
Perform a detailed contrastive analysis comparing correct and incorrect reasoning paths.

Your analysis should:
1. Compare the approaches side-by-side
2. Identify key differentiators that make the correct path right
3. Highlight common mistakes to avoid
4. Explain verification steps that catch errors
5. Provide insights for strengthening reasoning

Be clear, structured, and educational in your comparison."""

        # Build comparison context
        incorrect_summaries = "\n".join(
            f"Incorrect Path {i + 1}: {path[:200]}..."
            for i, path in enumerate(self._incorrect_paths)
        )

        user_prompt = f"""Correct reasoning path:
{self._correct_path[:500]}...

Incorrect reasoning paths:
{incorrect_summaries}

Perform a contrastive analysis comparing these {self._contrast_count + 1} reasoning paths.
Identify what makes the correct path right and what pitfalls the incorrect paths demonstrate.
Provide key differentiators and lessons learned."""

        def fallback() -> str:
            return self._generate_analysis(guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

        return f"Step {self._step_counter}: Contrastive Analysis\n\n{content}"

    async def _sample_refinement(
        self,
        analysis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate refined reasoning using LLM sampling.

        Strengthens the correct path based on contrastive analysis.

        Args:
            analysis_thought: The analysis thought to build upon
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the sampled refined reasoning
        """
        system_prompt = """You are a reasoning assistant using Contrastive CoT methodology.
Refine and strengthen the correct reasoning based on contrastive analysis.

Your refinement should:
1. Restate the correct approach with reinforcement
2. Explicitly mention avoided pitfalls from the analysis
3. Add verification steps to prevent identified errors
4. Explain how the reasoning is stronger after seeing what NOT to do
5. Boost confidence through explicit error avoidance

Make the reasoning more robust by learning from the contrasts."""

        user_prompt = f"""Correct reasoning path:
{self._correct_path[:400]}...

Contrastive analysis:
{analysis_thought.content[:500]}...

Refine the correct reasoning based on the contrastive analysis.
Strengthen it by explicitly avoiding the {self._contrast_count} identified pitfalls.
Add verification steps and explain the confidence boost from learning what NOT to do."""

        def fallback() -> str:
            return self._generate_refinement(analysis_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

        return f"Step {self._step_counter}: Refined Reasoning\n\n{content}"

    async def _sample_conclusion(
        self,
        refine_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final conclusion using LLM sampling.

        Creates a comprehensive conclusion summarizing the contrastive analysis.

        Args:
            refine_thought: The refined reasoning thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the sampled conclusion
        """
        system_prompt = """You are a reasoning assistant using Contrastive CoT methodology.
Generate a final conclusion summarizing the complete contrastive analysis.

Your conclusion should:
1. Summarize the contrastive process (generate → contrast → analyze → refine)
2. State the final answer with high confidence
3. Explain why the confidence is high (considered and rejected alternatives)
4. Highlight key learnings from the contrastive approach
5. Emphasize the value of learning from both correct and incorrect paths

Be confident, clear, and conclusive."""

        user_prompt = f"""Refined reasoning:
{refine_thought.content[:500]}...

Process summary:
1. Generated correct reasoning path
2. Created {self._contrast_count} contrasting incorrect paths
3. Analyzed differences between correct and incorrect
4. Refined reasoning by learning from errors

Generate a final conclusion with the answer and high confidence explanation.
Emphasize that explicitly considering and rejecting {self._contrast_count} incorrect \
approaches strengthened the solution."""

        def fallback() -> str:
            return self._generate_conclusion(refine_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        return f"Step {self._step_counter}: Final Answer\n\n{content}"


# Export
__all__ = ["ContrastiveCoT", "CONTRASTIVE_COT_METADATA"]
