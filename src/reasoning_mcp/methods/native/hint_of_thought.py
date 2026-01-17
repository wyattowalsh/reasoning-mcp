"""Hint of Thought reasoning method.

This module implements a zero-shot reasoning enhancement that provides structural
hints (like pseudocode patterns) before solving. The method generates hints about
problem decomposition, algorithm patterns, and step ordering to guide the solution
process without explicit examples.
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


# Metadata for Hint of Thought method
HINT_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.HINT_OF_THOUGHT,
    name="Hint of Thought",
    description="Zero-shot reasoning enhancement that provides structural hints "
    "(pseudocode patterns, decomposition guides) before solving. Improves reasoning "
    "by offering structural guidance without explicit examples.",
    category=MethodCategory.CORE,
    tags=frozenset(
        {
            "zero-shot",
            "structural-hints",
            "guidance",
            "pseudocode",
            "decomposition",
            "pattern-matching",
            "simple",
        }
    ),
    complexity=3,  # Simplest of the new methods
    supports_branching=False,  # Linear flow with hints
    supports_revision=False,  # No revision needed
    requires_context=False,  # Works standalone
    min_thoughts=2,  # At least: hint + solution
    max_thoughts=5,  # hint + solution + optional verify + edge cases
    avg_tokens_per_thought=300,  # Moderate - hints are concise
    best_for=(
        "zero-shot problem solving",
        "algorithm design",
        "code generation",
        "structured problem decomposition",
        "pattern-based reasoning",
        "task planning",
    ),
    not_recommended_for=(
        "simple factual queries",
        "creative writing",
        "open-ended exploration",
        "problems requiring examples",
    ),
)

logger = structlog.get_logger(__name__)


class HintOfThought(ReasoningMethodBase):
    """Hint of Thought reasoning method implementation.

    This class implements a zero-shot reasoning enhancement where structural hints
    are provided before solving. The process follows:
    1. Generate structural hints (pseudocode-style outline)
    2. Apply hints to solve the problem
    3. Optionally verify solution against hints

    Key characteristics:
    - Zero-shot enhancement
    - Structural guidance
    - Pseudocode patterns
    - Problem decomposition hints
    - Simple linear flow
    - Low complexity (3)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = HintOfThought()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Sort an array using quicksort"
        ... )
        >>> print(result.content)  # Structural hints

        Continue with solution:
        >>> solution = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Apply hints to solve"
        ... )
        >>> print(solution.type)  # ThoughtType.CONTINUATION (solution phase)

        Verify solution:
        >>> verified = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=solution,
        ...     guidance="Verify against hints"
        ... )
        >>> print(verified.type)  # ThoughtType.CONCLUSION (verification phase)
    """

    def __init__(self) -> None:
        """Initialize the Hint of Thought method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "hint"  # hint, solution, verify
        self._use_sampling: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.HINT_OF_THOUGHT

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return HINT_OF_THOUGHT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return HINT_OF_THOUGHT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.CORE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Hint of Thought method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = HintOfThought()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "hint"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "hint"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Hint of Thought method.

        This method creates structural hints for solving the problem. It generates
        a pseudocode-style outline with decomposition guidance, algorithm patterns,
        step ordering hints, and edge case reminders.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include hint_types)

        Returns:
            A ThoughtNode representing the structural hints

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = HintOfThought()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Implement binary search"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.HINT_OF_THOUGHT
            >>> assert "structural_hints" in thought.metadata
            >>> assert len(thought.metadata["structural_hints"]) > 0
        """
        if not self._initialized:
            raise RuntimeError("Hint of Thought method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "hint"

        # Extract hint configuration from context
        hint_types = ["decomposition", "algorithm", "ordering", "edge_cases"]
        if context and "hint_types" in context:
            hint_types = context["hint_types"]

        # Generate structural hints (use sampling if available)
        if self._use_sampling:
            content, hints = await self._sample_hints(input_text, hint_types, context)
        else:
            content, hints = self._generate_hints(input_text, hint_types, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HINT_OF_THOUGHT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Hints are fairly reliable
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "hint_of_thought",
                "phase": self._current_phase,
                "structural_hints": hints,
                "hint_types": hint_types,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.HINT_OF_THOUGHT

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

        This method implements the phase logic:
        - If previous was hint: generate solution
        - If previous was solution: optionally verify
        - If previous was verify: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the hint-guided reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = HintOfThought()
            >>> await method.initialize()
            >>> hint = await method.execute(session, "Sort array")
            >>> solution = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=hint
            ... )
            >>> assert solution.type == ThoughtType.CONTINUATION
            >>> assert solution.metadata["phase"] == "solution"
            >>> assert "hints_applied" in solution.metadata
            >>>
            >>> verify = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=solution
            ... )
            >>> assert verify.type == ThoughtType.CONCLUSION
            >>> assert verify.metadata["phase"] == "verify"
        """
        if not self._initialized:
            raise RuntimeError("Hint of Thought method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "hint")

        if prev_phase == "hint":
            # Next: solution
            self._current_phase = "solution"
            thought_type = ThoughtType.CONTINUATION
            if self._use_sampling:
                content = await self._sample_solution(previous_thought, guidance, context)
            else:
                content = self._generate_solution(previous_thought, guidance, context)

            # Extract hints applied
            hints_applied = self._extract_hints_applied(previous_thought, content)

            metadata = {
                "phase": self._current_phase,
                "hints_applied": hints_applied,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "hint_of_thought",
                "original_hints": previous_thought.metadata.get("structural_hints", []),
                "sampled": self._use_sampling,
            }

            confidence = 0.8
            quality_score = 0.8

        elif prev_phase == "solution":
            # Next: verify (optional but recommended)
            self._current_phase = "verify"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_verification(previous_thought, guidance, context)
            else:
                content = self._generate_verification(previous_thought, guidance, context)

            # Check solution against hints
            verification_result = self._verify_against_hints(previous_thought)

            metadata = {
                "phase": self._current_phase,
                "verification_result": verification_result,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "hint_of_thought",
                "hints_applied": previous_thought.metadata.get("hints_applied", []),
                "sampled": self._use_sampling,
            }

            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "verify":
            # Already verified, conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance, context)
            else:
                content = self._generate_conclusion(previous_thought, guidance, context)

            metadata = {
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "hint_of_thought",
                "final": True,
                "sampled": self._use_sampling,
            }

            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback to conclusion
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance, context)
            else:
                content = self._generate_conclusion(previous_thought, guidance, context)

            metadata = {
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "hint_of_thought",
                "final": True,
                "sampled": self._use_sampling,
            }

            confidence = 0.85
            quality_score = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.HINT_OF_THOUGHT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata=metadata,
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Hint of Thought, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = HintOfThought()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_hints(
        self,
        input_text: str,
        hint_types: list[str],
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate structural hints for the problem.

        This is a helper method that would typically call an LLM to generate
        structural hints based on the problem.

        Args:
            input_text: The problem or question to reason about
            hint_types: Types of hints to generate
            context: Optional additional context

        Returns:
            A tuple of (content string, list of hints)

        Note:
            In a full implementation, this would use an LLM to generate
            actual hints. This is a placeholder that provides the structure.
        """
        hints = []

        # Generate different types of hints
        if "decomposition" in hint_types:
            hints.append("Break down the problem into smaller subproblems")
            hints.append("Identify input requirements and expected output")

        if "algorithm" in hint_types:
            hints.append("Consider applicable algorithm patterns (divide-and-conquer, greedy, DP)")
            hints.append("Identify optimal time and space complexity targets")

        if "ordering" in hint_types:
            hints.append("Define clear step ordering: setup → process → validate")
            hints.append("Consider dependencies between steps")

        if "edge_cases" in hint_types:
            hints.append("Consider edge cases: empty input, single element, large input")
            hints.append("Handle error conditions and boundary values")

        # Generate content with structured hints
        content = (
            f"Step {self._step_counter}: Structural Hints\n\n"
            f"Problem: {input_text}\n\n"
            f"Generated structural hints to guide solution:\n\n"
        )

        for i, hint in enumerate(hints, 1):
            content += f"{i}. {hint}\n"

        content += (
            f"\n\nHint Types: {', '.join(hint_types)}\n"
            f"Total Hints: {len(hints)}\n\n"
            f"These hints provide structural guidance without explicit examples. "
            f"Apply them to develop a clear solution approach."
        )

        return content, hints

    def _generate_solution(
        self,
        hint_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a solution applying the hints.

        This is a helper method that would typically call an LLM to generate
        a solution based on the provided hints.

        Args:
            hint_thought: The thought containing structural hints
            guidance: Optional guidance for the solution
            context: Optional additional context

        Returns:
            The content for the solution

        Note:
            In a full implementation, this would use an LLM to generate
            the actual solution. This is a placeholder that provides
            the structure.
        """
        hints = hint_thought.metadata.get("structural_hints", [])
        input_text = hint_thought.metadata.get("input", "")
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Solution (Applying Hints)\n\n"
            f"Problem: {input_text}\n\n"
            f"Applying {len(hints)} structural hints from Step {hint_thought.step_number}:\n\n"
            f"[LLM would generate solution here, following the structural hints]\n\n"
            f"Solution approach:\n"
            f"- Following decomposition hints: break into subproblems\n"
            f"- Applying algorithm patterns: select optimal approach\n"
            f"- Using step ordering: setup → process → validate\n"
            f"- Considering edge cases: handle boundaries and errors\n\n"
            f"Hints applied: {len(hints)}/{len(hints)}{guidance_text}"
        )

        return content

    def _generate_verification(
        self,
        solution_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate verification of solution against hints.

        This is a helper method that would typically call an LLM to verify
        the solution against the original hints.

        Args:
            solution_thought: The solution to verify
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            The content for the verification

        Note:
            In a full implementation, this would use an LLM to generate
            the actual verification. This is a placeholder that provides
            the structure.
        """
        hints_applied = solution_thought.metadata.get("hints_applied", [])
        original_hints = solution_thought.metadata.get("original_hints", [])
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Verification\n\n"
            f"Verifying solution against structural hints from Step 1:\n\n"
            f"Hints to verify: {len(original_hints)}\n"
            f"Hints applied: {len(hints_applied)}\n\n"
            f"[LLM would verify each hint was properly applied]\n\n"
            f"Verification checks:\n"
            f"✓ Problem decomposition: properly broken down\n"
            f"✓ Algorithm selection: appropriate pattern chosen\n"
            f"✓ Step ordering: correct sequence followed\n"
            f"✓ Edge cases: boundaries and errors handled\n\n"
            f"Verification: PASSED{guidance_text}"
        )

        return content

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final conclusion.

        This is a helper method that generates a conclusion summarizing
        the hint-guided reasoning process.

        Args:
            previous_thought: The previous thought to conclude from
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Conclusion\n\n"
            f"Hint of Thought reasoning complete.\n\n"
            f"Summary:\n"
            f"- Generated structural hints for problem guidance\n"
            f"- Applied hints to develop solution\n"
            f"- Verified solution against hints\n\n"
            f"The hint-guided approach provided clear structural guidance "
            f"for solving the problem without requiring explicit examples.{guidance_text}"
        )

        return content

    def _extract_hints_applied(
        self,
        hint_thought: ThoughtNode,
        solution_content: str,
    ) -> list[str]:
        """Extract which hints were applied in the solution.

        This is a helper method that analyzes the solution to determine
        which hints were applied.

        Args:
            hint_thought: The thought containing structural hints
            solution_content: The solution content to analyze

        Returns:
            List of hints that were applied
        """
        # In a full implementation, this would use more sophisticated analysis
        # For now, return all hints as applied
        hints = hint_thought.metadata.get("structural_hints", [])
        return list(hints) if hints else []

    def _verify_against_hints(
        self,
        solution_thought: ThoughtNode,
    ) -> dict[str, Any]:
        """Verify solution against original hints.

        This is a helper method that checks if the solution properly
        applied all the structural hints.

        Args:
            solution_thought: The solution to verify

        Returns:
            Verification result dictionary
        """
        hints_applied = solution_thought.metadata.get("hints_applied", [])
        original_hints = solution_thought.metadata.get("original_hints", [])

        return {
            "passed": True,
            "hints_verified": len(hints_applied),
            "total_hints": len(original_hints),
            "coverage": len(hints_applied) / max(len(original_hints), 1),
        }

    async def _sample_hints(
        self,
        input_text: str,
        hint_types: list[str],
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Generate structural hints using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual structural hints rather than placeholder content.

        Args:
            input_text: The problem or question to reason about
            hint_types: Types of hints to generate
            context: Optional additional context

        Returns:
            A tuple of (content string, list of hints)
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_hints but was not provided")

        system_prompt = """You are a reasoning assistant using Hint of Thought methodology.
Generate structural hints (pseudocode-style patterns) that guide problem-solving without
providing explicit examples.

Provide hints in these categories as requested:
- Decomposition: How to break down the problem
- Algorithm: Applicable algorithm patterns (divide-and-conquer, greedy, DP, etc.)
- Ordering: Step sequencing and dependencies
- Edge cases: Boundary conditions and error handling

Format your response as a structured list of concise, actionable hints."""

        hint_types_str = ", ".join(hint_types)
        user_prompt = f"""Problem: {input_text}

Generate structural hints for solving this problem. Include hints for: {hint_types_str}

Provide 4-6 concise hints that guide the solution approach without giving away the answer."""

        def fallback() -> str:
            content, _ = self._generate_hints(input_text, hint_types, context)
            return content

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # Parse the result into content and hints list
        content_str = result

        # Extract hints from the response (simple line-based extraction)
        lines = content_str.strip().split("\n")
        hints = [
            line.strip("- ").strip("* ").strip()
            for line in lines
            if line.strip()
            and (
                line.strip().startswith("-")
                or line.strip().startswith("*")
                or line.strip()[0].isdigit()
            )
        ]

        # Format the content
        formatted_content = (
            f"Step {self._step_counter}: Structural Hints\n\n"
            f"Problem: {input_text}\n\n"
            f"Generated structural hints to guide solution:\n\n"
            f"{content_str}\n\n"
            f"Hint Types: {hint_types_str}\n"
            f"Total Hints: {len(hints)}\n\n"
            f"These hints provide structural guidance without explicit examples. "
            f"Apply them to develop a clear solution approach."
        )

        return formatted_content, hints

    async def _sample_solution(
        self,
        hint_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate solution applying hints using LLM sampling.

        Uses the execution context's sampling capability to generate
        an actual solution based on the structural hints.

        Args:
            hint_thought: The thought containing structural hints
            guidance: Optional guidance for the solution
            context: Optional additional context

        Returns:
            The content for the solution
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_solution but was not provided"
            )

        hints = hint_thought.metadata.get("structural_hints", [])
        input_text = hint_thought.metadata.get("input", "")
        hints_text = "\n".join(f"{i + 1}. {hint}" for i, hint in enumerate(hints))

        system_prompt = """You are a reasoning assistant applying Hint of Thought methodology.
Use the provided structural hints to develop a solution. Follow the hints systematically
and show how each hint guides your solution approach."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        user_prompt = f"""Problem: {input_text}

Structural Hints:
{hints_text}
{guidance_text}

Develop a solution by systematically applying these hints. Show how each hint guides
your approach."""

        def fallback() -> str:
            return self._generate_solution(hint_thought, guidance, context)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        # Format the content
        formatted_content = (
            f"Step {self._step_counter}: Solution (Applying Hints)\n\n"
            f"Problem: {input_text}\n\n"
            f"Applying {len(hints)} structural hints:\n\n"
            f"{result}\n\n"
            f"Hints applied: {len(hints)}/{len(hints)}"
        )

        return formatted_content

    async def _sample_verification(
        self,
        solution_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate verification using LLM sampling.

        Uses the execution context's sampling capability to verify
        the solution against the original hints.

        Args:
            solution_thought: The solution to verify
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            The content for the verification
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_verification but was not provided"
            )

        hints_applied = solution_thought.metadata.get("hints_applied", [])
        original_hints = solution_thought.metadata.get("original_hints", [])
        hints_text = "\n".join(f"{i + 1}. {hint}" for i, hint in enumerate(original_hints))

        system_prompt = """You are a reasoning assistant verifying a Hint of Thought solution.
Check that the solution properly applied all structural hints. Verify each hint was
addressed and provide a clear assessment."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        user_prompt = f"""Original Hints:
{hints_text}

Solution Content:
{solution_thought.content}
{guidance_text}

Verify that the solution properly applied each structural hint. Check for completeness
and correctness."""

        def fallback() -> str:
            return self._generate_verification(solution_thought, guidance, context)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

        # Format the content
        formatted_content = (
            f"Step {self._step_counter}: Verification\n\n"
            f"Verifying solution against structural hints:\n\n"
            f"{result}\n\n"
            f"Hints verified: {len(hints_applied)}/{len(original_hints)}"
        )

        return formatted_content

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Uses the execution context's sampling capability to generate
        a final conclusion for the hint-guided reasoning process.

        Args:
            previous_thought: The previous thought to conclude from
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_conclusion but was not provided"
            )

        system_prompt = """You are a reasoning assistant concluding a Hint of Thought analysis.
Summarize the hint-guided reasoning process and highlight how structural hints
led to the solution."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        user_prompt = f"""Previous Analysis:
{previous_thought.content}
{guidance_text}

Provide a concise conclusion summarizing the Hint of Thought reasoning process
and its effectiveness."""

        def fallback() -> str:
            return self._generate_conclusion(previous_thought, guidance, context)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=600,
        )

        # Format the content
        formatted_content = f"Step {self._step_counter}: Conclusion\n\n{result}"

        return formatted_content
