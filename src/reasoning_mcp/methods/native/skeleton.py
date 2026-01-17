"""Skeleton of Thought reasoning method.

This module implements the Skeleton of Thought approach, which creates a high-level
outline first, then expands each point in parallel, and finally assembles the complete
answer. This method is particularly effective for structured, long-form responses where
a clear organizational framework can be established upfront.

The approach follows three phases:
1. Skeleton Generation: Create a high-level outline/skeleton
2. Parallel Expansion: Expand each skeleton point independently
3. Final Assembly: Synthesize expanded points into a coherent answer
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


# Metadata for Skeleton of Thought method
SKELETON_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SKELETON_OF_THOUGHT,
    name="Skeleton of Thought",
    description="Create high-level skeleton first, then fill in details. "
    "Generates an outline, expands each point in parallel, then assembles the final answer.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "skeleton",
            "outline",
            "parallel",
            "structured",
            "long-form",
            "hierarchical",
        }
    ),
    complexity=4,  # Medium complexity - structured but requires parallelization
    supports_branching=True,  # Supports parallel expansion of skeleton points
    supports_revision=True,  # Can revise skeleton or expansions
    requires_context=False,  # No special context needed
    min_thoughts=3,  # Minimum: skeleton + 1 expansion + synthesis
    max_thoughts=0,  # No hard limit (depends on skeleton size)
    avg_tokens_per_thought=400,  # Medium-length thoughts
    best_for=(
        "long-form answers",
        "structured responses",
        "essays and reports",
        "comprehensive explanations",
        "multi-part questions",
        "organized presentations",
    ),
    not_recommended_for=(
        "simple yes/no questions",
        "single-step problems",
        "problems requiring linear dependency",
        "highly exploratory tasks",
    ),
)


class SkeletonOfThought(ReasoningMethodBase):
    """Skeleton of Thought reasoning method implementation.

    This class implements the Skeleton of Thought pattern, which creates structured
    responses by first generating a high-level outline, then expanding each point
    in parallel, and finally assembling the complete answer.

    The method proceeds in three phases:

    1. Skeleton Phase: Generate a high-level outline with main points
    2. Expansion Phase: Expand each skeleton point independently (can be parallel)
    3. Assembly Phase: Synthesize all expanded points into a coherent final answer

    Key characteristics:
    - Outline-first approach
    - Parallel point expansion
    - Structured organization
    - Good for long-form content
    - Medium complexity with branching support

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SkeletonOfThought()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain the principles of object-oriented programming"
        ... )
        >>> print(result.content)  # The skeleton/outline

        Continue to expand a specific point:
        >>> expansion = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Expand point 1: Encapsulation"
        ... )
    """

    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Skeleton of Thought method."""
        self._initialized = False
        self._step_counter = 0
        self._skeleton_points: list[str] = []
        self._expanded_points: dict[int, str] = {}
        self._phase: str = "skeleton"  # skeleton, expansion, or assembly
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.SKELETON_OF_THOUGHT

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return SKELETON_OF_THOUGHT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return SKELETON_OF_THOUGHT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Skeleton of Thought method for execution.
        Resets all internal state for a fresh reasoning session.

        Examples:
            >>> method = SkeletonOfThought()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._phase == "skeleton"
        """
        self._initialized = True
        self._step_counter = 0
        self._skeleton_points = []
        self._expanded_points = {}
        self._phase = "skeleton"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Skeleton of Thought method.

        This method creates the initial skeleton/outline for the problem.
        The skeleton provides a high-level structure that will be expanded
        in subsequent steps.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the skeleton/outline

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = SkeletonOfThought()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Explain machine learning concepts"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert "skeleton" in thought.metadata
        """
        if not self._initialized:
            raise RuntimeError("Skeleton of Thought method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._skeleton_points = []
        self._expanded_points = {}
        self._phase = "skeleton"

        # Generate the skeleton (use sampling if available)
        if execution_context and execution_context.can_sample:
            content = await self._sample_skeleton_generation(input_text, context)
        else:
            content = self._generate_skeleton_heuristic(input_text, context)

        # Extract skeleton points for tracking
        self._skeleton_points = self._extract_skeleton_points(content)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SKELETON_OF_THOUGHT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.8,  # High confidence in structural outline
            metadata={
                "input": input_text,
                "context": context or {},
                "phase": "skeleton",
                "skeleton_points": self._skeleton_points,
                "total_points": len(self._skeleton_points),
                "sampled": execution_context is not None and execution_context.can_sample,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SKELETON_OF_THOUGHT

        # Move to expansion phase
        self._phase = "expansion"

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

        Depending on the current phase, this method either:
        - Expands a skeleton point (expansion phase)
        - Assembles expanded points into final answer (assembly phase)

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "Expand point 2")
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode for the expansion or assembly

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> # After creating skeleton
            >>> expansion = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=skeleton_thought,
            ...     guidance="Expand point 1"
            ... )
            >>> assert expansion.type == ThoughtType.BRANCH
            >>> assert "expansion" in expansion.metadata["phase"]
        """
        if not self._initialized:
            raise RuntimeError("Skeleton of Thought method must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1

        if self._phase == "expansion":
            return await self._expand_point(
                session=session,
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
                execution_context=execution_context,
            )
        elif self._phase == "assembly":
            return await self._assemble_final(
                session=session,
                previous_thought=previous_thought,
                context=context,
                execution_context=execution_context,
            )
        else:
            # Fallback: continue expansion
            return await self._expand_point(
                session=session,
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
                execution_context=execution_context,
            )

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = SkeletonOfThought()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _sample_skeleton_generation(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate skeleton using LLM sampling.

        Uses the execution context's sampling capability to generate
        a high-level outline/skeleton for the problem.

        Args:
            input_text: The problem or question to create a skeleton for
            context: Optional additional context

        Returns:
            The skeleton content as a string
        """
        system_prompt = """You are a reasoning assistant using Skeleton of Thought methodology.
Generate a high-level outline/skeleton that breaks down the problem into main points.

Your skeleton should:
1. Identify 3-7 main points to address
2. Use clear, numbered structure
3. Include brief sub-points under each main point
4. Provide a logical flow from introduction to conclusion
5. Be concise but comprehensive

Format your outline clearly with numbered points and indented sub-points."""

        user_prompt = f"""Problem: {input_text}

Generate a Skeleton of Thought outline for addressing this problem.
Create a structured outline with main points and sub-points."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_skeleton_heuristic(input_text, context),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

    def _generate_skeleton_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial skeleton/outline using heuristics.

        This creates a high-level outline of the main points to address
        when LLM sampling is not available.

        Args:
            input_text: The problem or question to create a skeleton for
            context: Optional additional context

        Returns:
            The skeleton content as a string
        """
        return (
            f"Skeleton of Thought: Outline for '{input_text}'\n\n"
            f"I'll structure my response with the following main points:\n\n"
            f"1. Introduction and Overview\n"
            f"   - Define the core concept\n"
            f"   - Explain its significance\n\n"
            f"2. Key Components and Principles\n"
            f"   - Break down main elements\n"
            f"   - Describe fundamental principles\n\n"
            f"3. Practical Applications and Examples\n"
            f"   - Provide concrete examples\n"
            f"   - Show real-world usage\n\n"
            f"4. Common Challenges and Best Practices\n"
            f"   - Identify typical pitfalls\n"
            f"   - Suggest best approaches\n\n"
            f"5. Summary and Conclusions\n"
            f"   - Recap key points\n"
            f"   - Provide final insights\n\n"
            f"Each point will be expanded in detail to create a comprehensive response."
        )

    def _extract_skeleton_points(self, skeleton_content: str) -> list[str]:
        """Extract the main points from the skeleton.

        This parses the skeleton to identify the main points that need expansion.

        Args:
            skeleton_content: The skeleton content

        Returns:
            List of skeleton point identifiers
        """
        # Simple extraction - in reality would parse more carefully
        points = []
        for line in skeleton_content.split("\n"):
            line = line.strip()
            # Look for numbered points
            if line and line[0].isdigit() and "." in line:
                points.append(line.split(".")[0])
        return points if points else ["1", "2", "3", "4", "5"]

    async def _expand_point(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Expand a specific skeleton point.

        This creates a detailed expansion of one point from the skeleton.
        These expansions can be done in parallel for different points.

        Args:
            session: The current reasoning session
            previous_thought: The thought to build upon
            guidance: Guidance about which point to expand
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode with the expanded content
        """
        # Determine which point we're expanding
        point_num = self._extract_point_number(guidance)
        point_index = point_num - 1 if point_num > 0 else len(self._expanded_points)

        # Generate expansion content (use sampling if available)
        if execution_context and execution_context.can_sample:
            content = await self._sample_expansion(
                point_num=point_num,
                skeleton_points=self._skeleton_points,
                guidance=guidance,
                context=context,
            )
        else:
            content = self._generate_expansion_heuristic(
                point_num=point_num,
                skeleton_points=self._skeleton_points,
                guidance=guidance,
                context=context,
            )

        # Track this expansion
        self._expanded_points[point_index] = content

        # Determine if we should move to assembly phase
        if len(self._expanded_points) >= len(self._skeleton_points):
            self._phase = "assembly"

        thought = ThoughtNode(
            type=ThoughtType.BRANCH,  # Each expansion is a branch
            method_id=MethodIdentifier.SKELETON_OF_THOUGHT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.75,
            branch_id=f"point_{point_num}",
            metadata={
                "phase": "expansion",
                "point_number": point_num,
                "guidance": guidance or "",
                "context": context or {},
                "expansions_complete": len(self._expanded_points),
                "total_points": len(self._skeleton_points),
                "sampled": execution_context is not None and execution_context.can_sample,
            },
        )

        session.add_thought(thought)
        return thought

    async def _assemble_final(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Assemble all expanded points into the final answer.

        This synthesizes all the expanded points into a coherent, complete response.

        Args:
            session: The current reasoning session
            previous_thought: The thought to build upon
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode with the final assembled answer
        """
        # Generate assembly content (use sampling if available)
        if execution_context and execution_context.can_sample:
            content = await self._sample_assembly(
                expanded_points=self._expanded_points,
                skeleton_points=self._skeleton_points,
                context=context,
            )
        else:
            content = self._generate_assembly_heuristic(
                expanded_points=self._expanded_points,
                skeleton_points=self._skeleton_points,
                context=context,
            )

        thought = ThoughtNode(
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.SKELETON_OF_THOUGHT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.85,
            metadata={
                "phase": "assembly",
                "points_assembled": len(self._expanded_points),
                "context": context or {},
                "sampled": execution_context is not None and execution_context.can_sample,
            },
        )

        session.add_thought(thought)
        return thought

    def _extract_point_number(self, guidance: str | None) -> int:
        """Extract point number from guidance string.

        Args:
            guidance: The guidance string (e.g., "Expand point 2")

        Returns:
            The point number, or 0 if not found
        """
        if not guidance:
            return len(self._expanded_points) + 1

        # Try to extract number from guidance
        import re

        match = re.search(r"\d+", guidance)
        if match:
            return int(match.group())

        return len(self._expanded_points) + 1

    async def _sample_expansion(
        self,
        point_num: int,
        skeleton_points: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate expansion content using LLM sampling.

        Uses the execution context's sampling capability to expand
        a specific skeleton point with detailed content.

        Args:
            point_num: The point number to expand
            skeleton_points: The list of skeleton points
            guidance: Optional guidance
            context: Optional context

        Returns:
            The expansion content
        """
        skeleton_context = "\n".join(
            f"Point {i + 1}: {point}" for i, point in enumerate(skeleton_points)
        )
        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using Skeleton of Thought methodology.
Expand a specific point from the skeleton outline with detailed, comprehensive content.

Your expansion should:
1. Provide thorough coverage of the point
2. Include examples, explanations, and relevant details
3. Stay focused on the specific point
4. Use clear, well-organized prose
5. Be substantial but not overly verbose"""

        user_prompt = f"""Skeleton outline:
{skeleton_context}

Expand Point {point_num} in detail.{guidance_text}

Provide a comprehensive expansion of this point."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_expansion_heuristic(
                point_num, skeleton_points, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

    def _generate_expansion_heuristic(
        self,
        point_num: int,
        skeleton_points: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate expansion content for a specific point using heuristics.

        Fallback method when LLM sampling is not available.

        Args:
            point_num: The point number to expand
            skeleton_points: The list of skeleton points
            guidance: Optional guidance
            context: Optional context

        Returns:
            The expansion content
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Expansion of Point {point_num}\n\n"
            f"Detailed discussion of point {point_num} from the skeleton. "
            f"This section provides comprehensive coverage of this topic, "
            f"including examples, explanations, and relevant details.\n\n"
            f"[In a full implementation, this would be a detailed expansion "
            f"generated by an LLM based on the skeleton point "
            f"and the original question.]{guidance_text}"
        )

    async def _sample_assembly(
        self,
        expanded_points: dict[int, str],
        skeleton_points: list[str],
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final assembled answer using LLM sampling.

        Uses the execution context's sampling capability to synthesize
        all expanded points into a coherent, complete response.

        Args:
            expanded_points: Dict of expanded point content
            skeleton_points: The original skeleton points
            context: Optional context

        Returns:
            The final assembled content
        """
        # Format expanded points for context
        expansions_text = "\n\n".join(
            f"Point {i + 1} Expansion:\n{content}" for i, content in sorted(expanded_points.items())
        )

        system_prompt = """You are a reasoning assistant using Skeleton of Thought methodology.
Synthesize all expanded points into a coherent, complete final answer.

Your assembly should:
1. Create a natural, flowing narrative
2. Integrate all expanded points smoothly
3. Ensure logical transitions between sections
4. Provide a comprehensive, well-structured response
5. Maintain clarity and coherence throughout"""

        user_prompt = f"""Here are the expanded points from the skeleton:

{expansions_text}

Synthesize these expansions into a coherent, complete final answer that flows naturally
and provides a comprehensive response to the original question."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_assembly_heuristic(
                expanded_points, skeleton_points, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
        )

    def _generate_assembly_heuristic(
        self,
        expanded_points: dict[int, str],
        skeleton_points: list[str],
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final assembled answer using heuristics.

        Fallback method when LLM sampling is not available.

        Args:
            expanded_points: Dict of expanded point content
            skeleton_points: The original skeleton points
            context: Optional context

        Returns:
            The final assembled content
        """
        points_summary = "\n".join(
            f"- Point {i + 1}: "
            f"{skeleton_points[i] if i < len(skeleton_points) else f'Point {i + 1}'}"
            for i in range(len(expanded_points))
        )
        return (
            f"Final Assembly: Complete Answer\n\n"
            f"Synthesizing all {len(expanded_points)} expanded points "
            f"into a comprehensive response.\n\n"
            f"This final answer integrates:\n"
            f"{points_summary}\n\n"
            "[In a full implementation, this would be a coherent, well-structured "
            "response that flows naturally while incorporating all the expanded points.]"
        )
