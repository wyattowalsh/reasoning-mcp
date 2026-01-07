"""Least to Most reasoning method.

This module implements the Least to Most reasoning method, which progressively
decomposes complex problems into subproblems ordered from easiest to hardest,
solving each sequentially while building on previous solutions. This method
excels at multi-part problems where earlier solutions inform later ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for Least to Most method
LEAST_TO_MOST_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LEAST_TO_MOST,
    name="Least to Most",
    description="Progressive problem decomposition from easiest to hardest. "
    "Breaks complex problems into ordered subproblems, solving each sequentially "
    "while building on previous solutions.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({
        "decomposition",
        "progressive",
        "sequential",
        "building",
        "subproblems",
        "ordered",
    }),
    complexity=4,  # Medium complexity - requires problem decomposition and ordering
    supports_branching=False,  # Sequential progression through subproblems
    supports_revision=True,  # Can revise subproblem solutions
    requires_context=False,  # No special context needed
    min_thoughts=3,  # At minimum: decomposition + 2 subproblems
    max_thoughts=0,  # No limit - depends on problem complexity
    avg_tokens_per_thought=400,  # Moderate - each subproblem solution
    best_for=(
        "complex multi-part problems",
        "problems with natural difficulty progression",
        "mathematical problems requiring foundation building",
        "educational content requiring scaffolding",
        "problems where later steps depend on earlier ones",
        "compositional tasks",
    ),
    not_recommended_for=(
        "simple single-step problems",
        "problems with independent parallel tasks",
        "problems requiring non-linear exploration",
        "creative brainstorming",
        "problems with no clear difficulty ordering",
    ),
)


class LeastToMost:
    """Least to Most reasoning method implementation.

    This class implements a progressive problem-solving approach that:
    1. Decomposes the problem into subproblems
    2. Orders subproblems from easiest to hardest
    3. Solves each subproblem sequentially
    4. Builds on previous solutions to solve harder problems

    Key characteristics:
    - Progressive difficulty ordering
    - Sequential solution building
    - Each step leverages previous results
    - Scaffolded learning approach
    - Good for compositional reasoning

    The method follows this process:
    - Step 1: Analyze and decompose the problem
    - Step 2: Order subproblems by difficulty
    - Steps 3+: Solve each subproblem, building on previous solutions
    - Final step: Synthesize all subproblem solutions

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = LeastToMost()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Prove the Pythagorean theorem"
        ... )
        >>> print(result.content)  # Problem decomposition

        Continue through subproblems:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Solve the first subproblem"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    def __init__(self) -> None:
        """Initialize the Least to Most method."""
        self._initialized = False
        self._step_counter = 0
        self._subproblems: list[str] = []
        self._subproblem_solutions: list[str] = []
        self._decomposition_complete = False

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.LEAST_TO_MOST

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return LEAST_TO_MOST_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return LEAST_TO_MOST_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Least to Most method for execution,
        resetting internal state for a new reasoning session.

        Examples:
            >>> method = LeastToMost()
            >>> await method.initialize()
            >>> assert method._initialized is True
        """
        self._initialized = True
        self._step_counter = 0
        self._subproblems = []
        self._subproblem_solutions = []
        self._decomposition_complete = False

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the Least to Most method.

        This method creates the first thought which decomposes the problem
        into ordered subproblems from easiest to hardest.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            A ThoughtNode representing the problem decomposition

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LeastToMost()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Solve a complex multi-step problem"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.LEAST_TO_MOST
        """
        if not self._initialized:
            raise RuntimeError(
                "Least to Most method must be initialized before execution"
            )

        # Reset state for new execution
        self._step_counter = 1
        self._subproblems = []
        self._subproblem_solutions = []
        self._decomposition_complete = False

        # Create the initial decomposition thought
        content = self._generate_decomposition(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LEAST_TO_MOST,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,  # Good initial confidence for decomposition
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "least_to_most",
                "stage": "decomposition",
                "subproblems_identified": len(self._subproblems),
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LEAST_TO_MOST

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        This method generates the next step in the least-to-most process:
        - After decomposition: Order subproblems by difficulty
        - During solving: Solve the next subproblem using previous solutions
        - At completion: Synthesize all solutions

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LeastToMost()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Analyze problem")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Solve first subproblem"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError(
                "Least to Most method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Determine the stage and generate appropriate content
        stage = previous_thought.metadata.get("stage", "decomposition")

        if stage == "decomposition" and not self._decomposition_complete:
            # Next step: Order subproblems by difficulty
            content, new_stage = self._generate_ordering(previous_thought, guidance, context)
            self._decomposition_complete = True
        elif len(self._subproblem_solutions) < len(self._subproblems):
            # Solving phase: Solve next subproblem
            content, new_stage = self._generate_subproblem_solution(
                previous_thought, guidance, context
            )
        else:
            # Synthesis phase: Combine all solutions
            content, new_stage = self._generate_synthesis(
                previous_thought, guidance, context
            )

        # Determine thought type based on stage
        if new_stage == "synthesis":
            thought_type = ThoughtType.SYNTHESIS
        elif len(self._subproblem_solutions) == len(self._subproblems):
            thought_type = ThoughtType.CONCLUSION
        else:
            thought_type = ThoughtType.CONTINUATION

        # Calculate confidence (increases as we solve more subproblems)
        progress = len(self._subproblem_solutions) / max(1, len(self._subproblems))
        confidence = 0.7 + (0.25 * progress)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LEAST_TO_MOST,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "least_to_most",
                "stage": new_stage,
                "subproblems_total": len(self._subproblems),
                "subproblems_solved": len(self._subproblem_solutions),
                "progress": f"{len(self._subproblem_solutions)}/{len(self._subproblems)}",
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Least to Most, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = LeastToMost()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_decomposition(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial problem decomposition.

        This method analyzes the problem and identifies subproblems
        that can be solved progressively.

        Args:
            input_text: The problem to decompose
            context: Optional additional context

        Returns:
            The decomposition thought content

        Note:
            In a full implementation, this would use an LLM to intelligently
            decompose the problem. This is a placeholder structure.
        """
        # Simulate decomposition (in real implementation, LLM would do this)
        self._subproblems = [
            f"Subproblem {i+1}: Component of '{input_text}'"
            for i in range(3)  # Example: 3 subproblems
        ]

        subproblems_text = "\n".join(
            f"   {i+1}. {sp}" for i, sp in enumerate(self._subproblems)
        )

        return (
            f"Step {self._step_counter}: Problem Decomposition\n\n"
            f"Main Problem: {input_text}\n\n"
            f"I will decompose this problem into smaller, progressively more "
            f"difficult subproblems. By solving them from easiest to hardest, "
            f"each solution will provide foundation for the next.\n\n"
            f"Identified Subproblems:\n{subproblems_text}\n\n"
            f"Next: I will order these by difficulty and solve them sequentially."
        )

    def _generate_ordering(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate the subproblem ordering step.

        This method orders the subproblems from easiest to hardest.

        Args:
            previous_thought: The decomposition thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        # In real implementation, LLM would intelligently order by difficulty
        ordered_list = "\n".join(
            f"   {i+1}. {sp} [Difficulty: {i+1}/3]"
            for i, sp in enumerate(self._subproblems)
        )

        content = (
            f"Step {self._step_counter}: Ordering by Difficulty\n\n"
            f"I've analyzed the subproblems and ordered them from easiest to hardest:\n\n"
            f"{ordered_list}\n\n"
            f"This ordering ensures each solution builds on previous foundations. "
            f"I'll now solve them sequentially."
        )

        return content, "ordering"

    def _generate_subproblem_solution(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate a solution for the next subproblem.

        This method solves the next subproblem using previous solutions.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        subproblem_index = len(self._subproblem_solutions)
        current_subproblem = self._subproblems[subproblem_index]

        # Create solution building on previous ones
        solution = f"Solution to: {current_subproblem}"
        self._subproblem_solutions.append(solution)

        previous_solutions = ""
        if subproblem_index > 0:
            previous_solutions = (
                f"\n\nBuilding on previous solutions:\n"
                + "\n".join(
                    f"   - Solution {i+1}: {sol}"
                    for i, sol in enumerate(self._subproblem_solutions[:-1])
                )
            )

        content = (
            f"Step {self._step_counter}: Solving Subproblem {subproblem_index + 1}\n\n"
            f"Current Subproblem: {current_subproblem}\n\n"
            f"Approach: Using foundations from easier subproblems to inform this solution."
            f"{previous_solutions}\n\n"
            f"Solution: {solution}\n\n"
            f"Progress: {len(self._subproblem_solutions)}/{len(self._subproblems)} "
            f"subproblems solved."
        )

        return content, "solving"

    def _generate_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate the final synthesis of all solutions.

        This method combines all subproblem solutions into a complete answer.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        all_solutions = "\n".join(
            f"   {i+1}. {sol}"
            for i, sol in enumerate(self._subproblem_solutions)
        )

        content = (
            f"Step {self._step_counter}: Final Synthesis\n\n"
            f"All subproblems have been solved progressively. "
            f"Now I'll synthesize them into a complete solution.\n\n"
            f"Subproblem Solutions:\n{all_solutions}\n\n"
            f"Integrated Solution: By building from easiest to hardest, "
            f"each step provided necessary foundation for the next. "
            f"The complete solution emerges from this progressive building process."
        )

        return content, "synthesis"
