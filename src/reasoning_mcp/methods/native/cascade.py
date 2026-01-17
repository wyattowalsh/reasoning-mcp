"""Cascade Thinking reasoning method.

This module implements the Cascade Thinking approach, which applies hierarchical
cascade refinement starting at high abstraction and cascading down through increasing
levels of detail. This method is particularly effective for complex planning, strategy
development, and multi-scale problem-solving where both high-level vision and detailed
implementation are needed.

The approach follows cascade levels:
1. STRATEGIC: Highest level - goals, vision, big picture
2. TACTICAL: Mid-level - approaches, methods, strategies
3. OPERATIONAL: Lower level - specific steps, actions
4. DETAILED: Lowest level - implementation specifics

Feedback loops allow insights at lower levels to cascade back up and refine higher levels.
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


# Metadata for Cascade Thinking method
CASCADE_THINKING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CASCADE_THINKING,
    name="Cascade Thinking",
    description="Hierarchical refinement cascading from strategic to detailed levels. "
    "Starts at high abstraction and progressively refines through tactical, operational, "
    "and detailed levels with feedback loops for upward refinement.",
    category=MethodCategory.HOLISTIC,
    tags=frozenset(
        {
            "hierarchical",
            "cascade",
            "refinement",
            "multi-level",
            "strategic",
            "tactical",
            "operational",
            "progressive",
        }
    ),
    complexity=6,  # Medium-high complexity - requires multi-level abstraction management
    supports_branching=True,  # Supports exploring alternatives at each level
    supports_revision=True,  # Lower-level insights can revise higher levels
    requires_context=False,  # No special context needed
    min_thoughts=4,  # Minimum: strategic + tactical + operational + detailed
    max_thoughts=0,  # No hard limit (depends on refinement depth)
    avg_tokens_per_thought=450,  # Medium-length thoughts at each level
    best_for=(
        "project planning",
        "strategy development",
        "complex system design",
        "multi-scale problems",
        "organizational planning",
        "architecture design",
        "policy development",
        "comprehensive analysis",
    ),
    not_recommended_for=(
        "simple single-level problems",
        "quick decisions",
        "purely abstract reasoning",
        "problems requiring immediate action",
        "narrow technical details",
    ),
)


class CascadeThinkingMethod(ReasoningMethodBase):
    """Cascade Thinking reasoning method implementation.

    This class implements hierarchical cascade refinement, progressively moving
    from high-level strategic thinking down through tactical, operational, and
    detailed levels. At each level, the method:
    - Takes input from the higher level
    - Adds appropriate detail for the current abstraction level
    - Identifies issues that may need to cascade back up
    - Produces output for the next lower level

    The cascade levels are:
    1. STRATEGIC: High-level goals, vision, objectives, big picture
    2. TACTICAL: Approaches, methods, strategies to achieve strategic goals
    3. OPERATIONAL: Specific steps, actions, processes to execute tactics
    4. DETAILED: Implementation specifics, technical details, fine-grained plans

    Feedback loops enable bottom-up refinement when lower-level insights
    reveal issues with higher-level assumptions.

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = CascadeThinkingMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Design a distributed microservices architecture"
        ... )
        >>> print(result.content)  # Strategic level analysis

        Continue to next level:
        >>> tactical = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Cascade to tactical level"
        ... )
        >>> print(tactical.metadata["cascade_level"])  # "TACTICAL"
    """

    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Cascade Thinking method."""
        self._initialized = False
        self._step_counter = 0
        self._current_level = "STRATEGIC"
        self._cascade_levels = ["STRATEGIC", "TACTICAL", "OPERATIONAL", "DETAILED"]
        self._level_outputs: dict[str, str] = {}
        self._feedback_items: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.CASCADE_THINKING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return CASCADE_THINKING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return CASCADE_THINKING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Cascade Thinking method for execution,
        resetting all internal state for a fresh reasoning session.

        Examples:
            >>> method = CascadeThinkingMethod()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._current_level == "STRATEGIC"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_level = "STRATEGIC"
        self._level_outputs = {}
        self._feedback_items = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Cascade Thinking method.

        This method creates the initial strategic-level thought, analyzing
        the problem at the highest level of abstraction to establish goals,
        vision, and big-picture understanding.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the strategic level analysis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = CascadeThinkingMethod()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Develop a climate change mitigation strategy"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.metadata["cascade_level"] == "STRATEGIC"
            >>> assert thought.step_number == 1
        """
        if not self._initialized:
            raise RuntimeError("Cascade Thinking method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset state for new execution
        self._step_counter = 1
        self._current_level = "STRATEGIC"
        self._level_outputs = {}
        self._feedback_items = []

        # Generate strategic-level analysis
        content = await self._generate_strategic_level(input_text, context)
        self._level_outputs["STRATEGIC"] = content

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CASCADE_THINKING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.70,  # Strategic level: 0.7 + (0.05 * 0)
            metadata={
                "input": input_text,
                "context": context or {},
                "cascade_level": "STRATEGIC",
                "level_index": 0,
                "total_levels": len(self._cascade_levels),
                "reasoning_type": "cascade_thinking",
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.CASCADE_THINKING

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

        This method cascades down through levels (STRATEGIC → TACTICAL →
        OPERATIONAL → DETAILED) or processes feedback to cascade back up
        when lower-level insights require higher-level refinement.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "cascade down", "feedback up")
            context: Optional additional context

        Returns:
            A new ThoughtNode for the next cascade level or feedback revision

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> # After strategic level
            >>> tactical = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=strategic_thought,
            ...     guidance="Cascade to tactical"
            ... )
            >>> assert tactical.metadata["cascade_level"] == "TACTICAL"
            >>> assert tactical.parent_id == strategic_thought.id
            >>>
            >>> # Feedback loop
            >>> revision = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=detailed_thought,
            ...     guidance="Feedback: Strategic goals need adjustment"
            ... )
            >>> assert revision.type == ThoughtType.REVISION
        """
        if not self._initialized:
            raise RuntimeError("Cascade Thinking method must be initialized before continuation")

        self._step_counter += 1

        # Check if this is a feedback/revision request
        is_feedback = guidance and ("feedback" in guidance.lower() or "revise" in guidance.lower())

        if is_feedback:
            return await self._process_feedback(
                session=session,
                previous_thought=previous_thought,
                guidance=guidance,
                context=context,
            )

        # Otherwise, cascade to next level
        return await self._cascade_to_next_level(
            session=session,
            previous_thought=previous_thought,
            guidance=guidance,
            context=context,
        )

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = CascadeThinkingMethod()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _cascade_to_next_level(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> ThoughtNode:
        """Cascade to the next level in the hierarchy.

        Args:
            session: The current reasoning session
            previous_thought: The thought from the current level
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A ThoughtNode for the next cascade level
        """
        # Determine current and next level
        current_level = previous_thought.metadata.get("cascade_level", self._current_level)
        current_index = self._cascade_levels.index(current_level)

        # Check if we can cascade further
        if current_index >= len(self._cascade_levels) - 1:
            # Already at lowest level - create synthesis
            return await self._generate_synthesis(
                session=session,
                previous_thought=previous_thought,
                context=context,
            )

        next_index = current_index + 1
        next_level = self._cascade_levels[next_index]
        self._current_level = next_level

        # Generate content for next level
        content = await self._generate_cascade_level(
            level=next_level,
            previous_level_output=self._level_outputs.get(current_level, ""),
            guidance=guidance,
            context=context,
        )
        self._level_outputs[next_level] = content

        # Determine thought type
        if next_level == "DETAILED":
            thought_type = ThoughtType.SYNTHESIS  # Detailed level synthesizes all above
        else:
            thought_type = ThoughtType.CONTINUATION

        # Calculate confidence (increases as we move down levels with more specificity)
        confidence = 0.7 + (0.05 * next_index)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CASCADE_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "cascade_level": next_level,
                "level_index": next_index,
                "total_levels": len(self._cascade_levels),
                "previous_level": current_level,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "cascade_thinking",
            },
        )

        session.add_thought(thought)
        return thought

    async def _process_feedback(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> ThoughtNode:
        """Process feedback that cascades up to revise higher levels.

        Args:
            session: The current reasoning session
            previous_thought: The thought containing the feedback
            guidance: Feedback guidance
            context: Optional additional context

        Returns:
            A ThoughtNode revising a higher level based on feedback
        """
        current_level = previous_thought.metadata.get("cascade_level", self._current_level)

        # Generate feedback revision content
        content = await self._generate_feedback_revision(
            current_level=current_level,
            feedback=guidance or "Refining based on lower-level insights",
            context=context,
        )

        # Track feedback
        self._feedback_items.append(
            {
                "from_level": current_level,
                "feedback": guidance,
                "step": self._step_counter,
            }
        )

        thought = ThoughtNode(
            type=ThoughtType.REVISION,
            method_id=MethodIdentifier.CASCADE_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.8,  # Good confidence in refinement
            metadata={
                "cascade_level": current_level,
                "feedback_type": "upward_refinement",
                "original_feedback": guidance or "",
                "context": context or {},
                "reasoning_type": "cascade_thinking",
                "is_revision": True,
            },
        )

        session.add_thought(thought)
        return thought

    async def _generate_synthesis(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> ThoughtNode:
        """Generate final synthesis across all cascade levels.

        Args:
            session: The current reasoning session
            previous_thought: The previous thought
            context: Optional additional context

        Returns:
            A ThoughtNode synthesizing all levels
        """
        content = await self._generate_final_synthesis(context)

        thought = ThoughtNode(
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CASCADE_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.85,  # High confidence in complete cascade
            metadata={
                "cascade_level": "SYNTHESIS",
                "levels_completed": len(self._level_outputs),
                "feedback_iterations": len(self._feedback_items),
                "context": context or {},
                "reasoning_type": "cascade_thinking",
            },
        )

        session.add_thought(thought)
        return thought

    async def _generate_strategic_level(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate strategic-level analysis.

        Args:
            input_text: The problem to analyze
            context: Optional additional context

        Returns:
            Strategic level content
        """
        prompt = f"""Problem to analyze at STRATEGIC level: {input_text}

Generate a strategic-level analysis for this problem. Focus on:
1. Vision & Goals - What are we fundamentally trying to achieve?
2. Key Success Factors - What critical factors determine success or failure?
3. High-Level Scope - What's in scope and out of scope at this level?
4. Stakeholder Alignment - Who are the key stakeholders and what outcomes do they need?

Provide a comprehensive strategic analysis that establishes the foundation for tactical planning."""

        system_prompt = (
            "You are a strategic thinking expert analyzing problems "
            "at the highest level of abstraction. Focus on vision, goals, "
            "success factors, scope, and stakeholder alignment. "
            "Provide clear, actionable strategic insights that will "
            "cascade down to tactical levels."
        )

        def fallback() -> str:
            return (
                f"CASCADE THINKING - STRATEGIC LEVEL\n\n"
                f"Problem: {input_text}\n\n"
                f"Strategic Analysis:\n\n"
                f"1. Vision & Goals\n"
                f"   At the highest level, I need to understand the overarching vision and\n"
                f"   objectives. What are we fundamentally trying to achieve? What is the\n"
                f"   big picture success criteria?\n\n"
                f"2. Key Success Factors\n"
                f"   Identifying the critical factors that will determine success or failure\n"
                f"   at this strategic level. What are the major constraints, opportunities,\n"
                f"   and risks?\n\n"
                f"3. High-Level Scope\n"
                f"   Defining the boundaries and scale of this endeavor. What's in scope\n"
                f"   and out of scope at the strategic level?\n\n"
                f"4. Stakeholder Alignment\n"
                f"   Understanding who the key stakeholders are and what strategic outcomes\n"
                f"   they need.\n\n"
                f"Strategic Output: High-level goals and vision established. Ready to cascade\n"
                f"to tactical level for approach definition.\n\n"
                f"[In a full implementation, this would be generated by an LLM with deep\n"
                f"strategic analysis capabilities.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - STRATEGIC LEVEL" not in result:
            return f"CASCADE THINKING - STRATEGIC LEVEL\n\nProblem: {input_text}\n\n{result}"
        return result

    async def _generate_cascade_level(
        self,
        level: str,
        previous_level_output: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for a specific cascade level.

        Args:
            level: The cascade level (TACTICAL, OPERATIONAL, or DETAILED)
            previous_level_output: Output from the previous level
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Content for this cascade level
        """
        if level == "TACTICAL":
            return await self._generate_tactical_level(previous_level_output, guidance, context)
        elif level == "OPERATIONAL":
            return await self._generate_operational_level(previous_level_output, guidance, context)
        elif level == "DETAILED":
            return await self._generate_detailed_level(previous_level_output, guidance, context)
        else:
            return f"Cascade level {level}: Refining based on previous level analysis."

    async def _generate_tactical_level(
        self,
        strategic_output: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate tactical-level analysis.

        Args:
            strategic_output: Output from strategic level
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tactical level content
        """
        prompt = f"""Strategic Level Analysis:
{strategic_output}

{f"Additional Guidance: {guidance}" if guidance else ""}

Generate a TACTICAL-level analysis that builds on this strategic foundation. Focus on:
1. Approach Selection - What methodologies will best achieve strategic objectives?
2. Resource Strategy - How to allocate resources tactically?
3. Key Tactics & Methods - What specific tactics and workstreams are needed?
4. Risk Mitigation Tactics - How to address strategic risks at tactical level?

Provide concrete tactical approaches that cascade down from the strategic goals."""

        system_prompt = (
            "You are a tactical planning expert who translates "
            "strategic vision into actionable approaches. "
            "Focus on methodologies, resource allocation, and "
            "tactical approaches that bridge strategy and operations. "
            "Ensure your tactical analysis directly supports "
            "the strategic objectives."
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        def fallback() -> str:
            return (
                f"CASCADE THINKING - TACTICAL LEVEL\n\n"
                f"Building on Strategic Foundation:\n"
                f"Taking the strategic goals and vision from the previous level, I now\n"
                f"define the tactical approaches and methods to achieve them.\n\n"
                f"Tactical Analysis:\n\n"
                f"1. Approach Selection\n"
                f"   What are the primary methodologies and approaches that will best\n"
                f"   achieve our strategic objectives? Evaluating different tactical\n"
                f"   options and selecting the most promising.\n\n"
                f"2. Resource Strategy\n"
                f"   How do we tactically allocate resources (time, people, budget)\n"
                f"   to maximize strategic goal achievement?\n\n"
                f"3. Key Tactics & Methods\n"
                f"   Defining specific tactics, techniques, and methods that align with\n"
                f"   our strategic direction. What are the main workstreams?\n\n"
                f"4. Risk Mitigation Tactics\n"
                f"   Tactical approaches to address strategic risks identified at the\n"
                f"   higher level.\n\n"
                f"Tactical Output: Approaches and methods defined. Ready to cascade to\n"
                f"operational level for step-by-step planning.{guidance_text}\n\n"
                f"[In a full implementation, this would be generated by an LLM analyzing\n"
                f"tactical options based on strategic goals.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - TACTICAL LEVEL" not in result:
            return f"CASCADE THINKING - TACTICAL LEVEL\n\n{result}{guidance_text}"
        return result

    async def _generate_operational_level(
        self,
        tactical_output: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate operational-level analysis.

        Args:
            tactical_output: Output from tactical level
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Operational level content
        """
        prompt = f"""Tactical Level Analysis:
{tactical_output}

{f"Additional Guidance: {guidance}" if guidance else ""}

Generate an OPERATIONAL-level analysis that builds on tactical approaches. Focus on:
1. Process Definition - What specific processes and workflows are needed?
2. Action Planning - What concrete actions are required, with sequencing and dependencies?
3. Operational Roles & Responsibilities - Who does what at the operational level?
4. Metrics & Monitoring - What operational metrics track progress?
5. Feedback Points - Where might execution reveal issues with tactical/strategic assumptions?

Provide detailed operational processes that execute the tactical approaches."""

        system_prompt = (
            "You are an operational planning expert who converts "
            "tactical approaches into executable processes. "
            "Focus on step-by-step procedures, action sequences, "
            "roles, and operational metrics. "
            "Ensure your operational analysis directly implements "
            "the tactical approaches."
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        def fallback() -> str:
            return (
                f"CASCADE THINKING - OPERATIONAL LEVEL\n\n"
                f"Building on Tactical Approaches:\n"
                f"Taking the tactical methods from the previous level, I now define\n"
                f"specific operational steps, actions, and processes.\n\n"
                f"Operational Analysis:\n\n"
                f"1. Process Definition\n"
                f"   What are the specific processes and workflows needed to execute\n"
                f"   our tactical approaches? Defining step-by-step procedures.\n\n"
                f"2. Action Planning\n"
                f"   Concrete actions required, with sequencing and dependencies.\n"
                f"   What needs to happen and in what order?\n\n"
                f"3. Operational Roles & Responsibilities\n"
                f"   Who does what at the operational level? Clear assignment of\n"
                f"   operational tasks and ownership.\n\n"
                f"4. Metrics & Monitoring\n"
                f"   Operational metrics to track progress and ensure tactical approaches\n"
                f"   are being executed effectively.\n\n"
                f"5. Feedback Points\n"
                f"   Identifying where operational execution might reveal issues with\n"
                f"   tactical or strategic assumptions.\n\n"
                f"Operational Output: Processes and actions defined. Ready to cascade to\n"
                f"detailed level for implementation specifics.{guidance_text}\n\n"
                f"[In a full implementation, this would be generated by an LLM creating\n"
                f"detailed operational plans from tactical approaches.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - OPERATIONAL LEVEL" not in result:
            return f"CASCADE THINKING - OPERATIONAL LEVEL\n\n{result}{guidance_text}"
        return result

    async def _generate_detailed_level(
        self,
        operational_output: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate detailed-level analysis.

        Args:
            operational_output: Output from operational level
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Detailed level content
        """
        prompt = f"""Operational Level Analysis:
{operational_output}

{f"Additional Guidance: {guidance}" if guidance else ""}

Generate a DETAILED-level analysis that specifies implementation requirements. Focus on:
1. Implementation Specifications - Precise technical details, configurations, and specifications
2. Technical Requirements - Dependencies, tools, technologies, and infrastructure needs
3. Edge Cases & Exceptions - Handling of special cases, error conditions, exceptional scenarios
4. Quality & Validation Criteria - Specific validation criteria for the implementation
5. Upward Feedback Opportunities - Insights that may require revisions at higher levels

Provide comprehensive implementation details grounded in the full cascade from
strategic to operational."""

        system_prompt = (
            "You are an implementation expert who converts "
            "operational processes into detailed technical specifications. "
            "Focus on precise technical details, requirements, "
            "edge cases, and validation criteria. "
            "Ensure your detailed analysis directly implements "
            "the operational processes while maintaining "
            "traceability to strategic goals."
        )

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        def fallback() -> str:
            return (
                f"CASCADE THINKING - DETAILED LEVEL\n\n"
                f"Building on Operational Processes:\n"
                f"Taking the operational steps from the previous level, I now specify\n"
                f"detailed implementation requirements and technical specifics.\n\n"
                f"Detailed Analysis:\n\n"
                f"1. Implementation Specifications\n"
                f"   Precise technical details, configurations, and specifications needed\n"
                f"   to execute operational processes. Exact parameters and settings.\n\n"
                f"2. Technical Requirements\n"
                f"   Detailed technical requirements, dependencies, tools, technologies,\n"
                f"   and infrastructure needs at the implementation level.\n\n"
                f"3. Edge Cases & Exceptions\n"
                f"   Detailed handling of special cases, error conditions, and exceptional\n"
                f"   scenarios that operational processes must account for.\n\n"
                f"4. Quality & Validation Criteria\n"
                f"   Specific criteria for validating that implementation meets operational,\n"
                f"   tactical, and strategic requirements.\n\n"
                f"5. Upward Feedback Opportunities\n"
                f"   Identifying insights from detailed analysis that may require revisions\n"
                f"   at operational, tactical, or strategic levels.\n\n"
                f"Detailed Output: Complete cascade through all levels achieved. Implementation\n"
                f"specifics defined with traceable connection to strategic goals.{guidance_text}\n\n"
                f"[In a full implementation, this would be generated by an LLM providing\n"
                f"comprehensive implementation details grounded in the full cascade.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - DETAILED LEVEL" not in result:
            return f"CASCADE THINKING - DETAILED LEVEL\n\n{result}{guidance_text}"
        return result

    async def _generate_feedback_revision(
        self,
        current_level: str,
        feedback: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for feedback-driven revision.

        Args:
            current_level: The level providing feedback
            feedback: The feedback content
            context: Optional additional context

        Returns:
            Revision content
        """
        prompt = f"""Feedback from {current_level} level:
{feedback}

Generate a FEEDBACK REVISION analysis that revisits assumptions at higher
abstraction levels. Focus on:
1. What insights from {current_level} level require revisiting higher-level decisions?
2. What practical constraints or opportunities were not visible at higher levels?
3. How should earlier assumptions be refined based on {current_level}-level insights?
4. What adjustments ensure consistency across all cascade levels?
5. How will the revised understanding flow back down through subsequent levels?

Provide intelligent revision of higher-level decisions based on lower-level feedback."""

        system_prompt = (
            "You are a feedback analysis expert who identifies when "
            "lower-level insights require higher-level revisions. "
            "Focus on bidirectional flow in cascade thinking - "
            "ensuring lower-level practical realities inform "
            "strategic decisions. Provide thoughtful revisions that "
            "maintain cascade coherence while adapting to new insights."
        )

        def fallback() -> str:
            return (
                f"CASCADE THINKING - FEEDBACK REVISION\n\n"
                f"Feedback from {current_level} level:\n"
                f"{feedback}\n\n"
                f"Revision Analysis:\n"
                f"Based on insights gained at the {current_level} level, I'm revisiting\n"
                f"assumptions and decisions made at higher abstraction levels.\n\n"
                f"This feedback loop is essential in cascade thinking - lower levels often\n"
                f"reveal practical constraints or opportunities not visible at higher levels.\n\n"
                f"Adjustments:\n"
                f"- Refining earlier assumptions based on {current_level}-level insights\n"
                f"- Ensuring consistency across all cascade levels\n"
                f"- Updating plans to reflect new understanding\n\n"
                f"The revised understanding will flow back down through subsequent levels.\n\n"
                f"[In a full implementation, this would intelligently revise higher-level\n"
                f"decisions based on lower-level feedback.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - FEEDBACK REVISION" not in result:
            return (
                f"CASCADE THINKING - FEEDBACK REVISION\n\n"
                f"Feedback from {current_level} level:\n{feedback}\n\n{result}"
            )
        return result

    async def _generate_final_synthesis(
        self,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final synthesis across all levels.

        Args:
            context: Optional additional context

        Returns:
            Synthesis content
        """
        levels_summary = "\n".join(
            f"   - {level}: {len(output)} chars of analysis"
            for level, output in self._level_outputs.items()
        )

        feedback_summary = ""
        if self._feedback_items:
            feedback_summary = (
                f"\n\nFeedback Iterations: {len(self._feedback_items)}\n"
                f"Upward refinement occurred {len(self._feedback_items)} time(s), "
                f"demonstrating the adaptive nature of cascade thinking."
            )

        # Compile all level outputs for comprehensive synthesis
        all_levels = "\n\n=== LEVEL OUTPUTS ===\n\n"
        for level in ["STRATEGIC", "TACTICAL", "OPERATIONAL", "DETAILED"]:
            if level in self._level_outputs:
                all_levels += f"\n{level} LEVEL:\n{self._level_outputs[level][:500]}...\n"

        prompt = f"""Complete Cascade Analysis:
{all_levels}

{feedback_summary}

Generate a FINAL SYNTHESIS that integrates insights from all cascade levels. Focus on:
1. Vertical Coherence - How do detailed decisions trace back to strategic goals?
2. Progressive Refinement - How did each level add appropriate detail?
3. Bidirectional Flow - How did feedback loops enable upward refinement?
4. Multi-Scale Understanding - What insights emerge from the full cascade?

Provide a comprehensive synthesis that demonstrates the value of hierarchical cascade thinking."""

        system_prompt = (
            "You are a synthesis expert who integrates insights "
            "across multiple abstraction levels. "
            "Focus on vertical coherence, progressive refinement, "
            "and bidirectional flow in cascade thinking. "
            "Provide a cohesive final analysis that demonstrates "
            "the value of the complete cascade from strategic "
            "to detailed."
        )

        def fallback() -> str:
            return (
                f"CASCADE THINKING - FINAL SYNTHESIS\n\n"
                f"Complete Cascade Achieved:\n"
                f"I've successfully cascaded from strategic vision down through tactical\n"
                f"approaches, operational processes, and detailed implementation specifics.\n\n"
                f"Cascade Levels Completed:\n{levels_summary}\n{feedback_summary}\n\n"
                f"Key Insights:\n\n"
                f"1. Vertical Coherence\n"
                f"   Every detailed implementation decision traces back to tactical choices,\n"
                f"   which serve strategic goals. The cascade ensures alignment across all\n"
                f"   levels of abstraction.\n\n"
                f"2. Progressive Refinement\n"
                f"   Each level added appropriate detail while maintaining consistency with\n"
                f"   higher levels. This prevents losing sight of strategic goals while\n"
                f"   getting into details.\n\n"
                f"3. Bidirectional Flow\n"
                f"   While the primary cascade flows downward (strategic -> detailed),\n"
                f"   feedback loops enable upward refinement when lower levels reveal\n"
                f"   insights that necessitate strategic adjustments.\n\n"
                f"4. Multi-Scale Understanding\n"
                f"   The cascade thinking approach produces understanding at multiple scales\n"
                f"   simultaneously - from 30,000-foot strategic view to ground-level details.\n\n"
                f"Conclusion: The hierarchical cascade from strategic vision to detailed\n"
                f"implementation provides a comprehensive, coherent solution with strong\n"
                f"vertical alignment and traceable reasoning at every level.\n\n"
                f"[In a full implementation, this synthesis would intelligently integrate\n"
                f"insights from all levels into a cohesive final analysis.]"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "CASCADE THINKING - FINAL SYNTHESIS" not in result:
            return (
                f"CASCADE THINKING - FINAL SYNTHESIS\n\n"
                f"Cascade Levels Completed:\n{levels_summary}\n"
                f"{feedback_summary}\n\n{result}"
            )
        return result
