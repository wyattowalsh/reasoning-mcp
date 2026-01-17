"""Decomposed Prompting reasoning method.

This module implements the Decomposed Prompting reasoning method, which breaks
complex multi-disciplinary problems into specialist sub-tasks. Each sub-task is
assigned to a virtual specialist with domain expertise, executed independently,
and then integrated into a comprehensive solution. This method excels when
problems require diverse expertise across multiple domains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_selection,
)
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


# Metadata for Decomposed Prompting method
DECOMPOSED_PROMPTING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DECOMPOSED_PROMPTING,
    name="Decomposed Prompting",
    description="Multi-specialist decomposition for complex problems. "
    "Breaks problems into domain-specific sub-tasks, assigns each to a specialist, "
    "executes in parallel, and integrates expert outputs into a unified solution.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "decomposition",
            "specialists",
            "multi-disciplinary",
            "expertise",
            "parallel",
            "integration",
            "domain-specific",
        }
    ),
    complexity=6,  # Medium-high - requires specialist coordination and integration
    supports_branching=True,  # Can branch for parallel specialist execution
    supports_revision=True,  # Can revise specialist outputs
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At minimum: decomposition + 2 specialists + integration
    max_thoughts=0,  # No limit - depends on number of domains involved
    avg_tokens_per_thought=500,  # Moderate-high - specialist analysis
    best_for=(
        "multi-disciplinary problems",
        "complex problems requiring diverse expertise",
        "cross-functional analysis",
        "problems with distinct technical domains",
        "strategic planning requiring multiple perspectives",
        "comprehensive evaluations",
        "system design with multiple concerns",
    ),
    not_recommended_for=(
        "single-domain problems",
        "simple straightforward tasks",
        "problems requiring tightly coupled reasoning",
        "highly sequential dependencies",
        "creative tasks requiring unified vision",
    ),
)


class DecomposedPrompting(ReasoningMethodBase):
    """Decomposed Prompting reasoning method implementation.

    This class implements a multi-specialist approach that:
    1. Identifies required expertise areas for the problem
    2. Creates specialist prompts for each domain
    3. Executes each specialist task (potentially in parallel)
    4. Integrates specialist outputs into a unified solution

    Key characteristics:
    - Specialist role assignment based on problem domains
    - Domain-specific sub-prompts for each expert
    - Parallel execution capability through branching
    - Expert integration and synthesis
    - Good for multi-disciplinary problems
    - Medium-high complexity (5-6)

    The method follows this process:
    - Step 1: Analyze problem and identify required expertise areas
    - Step 2: Create specialist roles and assign sub-tasks
    - Steps 3+: Execute each specialist sub-task
    - Final step: Integrate all specialist outputs

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = DecomposedPrompting()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Design a sustainable smart city infrastructure"
        ... )
        >>> print(result.content)  # Expertise identification and decomposition

        Continue with specialist execution:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Execute first specialist task"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Decomposed Prompting method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._specialists: list[dict[str, str]] = []
        self._specialist_outputs: dict[str, str] = {}
        self._decomposition_complete = False
        self._specialists_assigned = False
        self._use_sampling = False
        self._execution_context: ExecutionContext | None = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.DECOMPOSED_PROMPTING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return DECOMPOSED_PROMPTING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return DECOMPOSED_PROMPTING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Decomposed Prompting method for execution,
        resetting internal state for a new reasoning session.

        Examples:
            >>> method = DecomposedPrompting()
            >>> await method.initialize()
            >>> assert method._initialized is True
        """
        self._initialized = True
        self._step_counter = 0
        self._specialists = []
        self._specialist_outputs = {}
        self._decomposition_complete = False
        self._specialists_assigned = False

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Decomposed Prompting method.

        This method creates the first thought which identifies required
        expertise areas and begins the decomposition process.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the expertise identification

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = DecomposedPrompting()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Build a multi-platform application"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.DECOMPOSED_PROMPTING
        """
        if not self._initialized:
            raise RuntimeError("Decomposed Prompting method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset state for new execution
        self._step_counter = 1
        self._specialists = []
        self._specialist_outputs = {}
        self._decomposition_complete = False
        self._specialists_assigned = False

        # Optional elicitation: ask user how to decompose the problem
        decomposition_strategy = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {
                        "id": "sequential",
                        "label": "Sequential Steps - Decompose into ordered sequential steps",
                    },
                    {
                        "id": "hierarchical",
                        "label": "Hierarchical - Decompose into hierarchical sub-problems",
                    },
                    {
                        "id": "parallel",
                        "label": "Parallel Tasks - Decompose into independent parallel tasks",
                    },
                    {
                        "id": "domain",
                        "label": "Domain Experts - Decompose by specialist domains (default)",
                    },
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "How should we decompose this problem?",
                    options,
                    config=config,
                )
                if selection:
                    decomposition_strategy = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
                # Fall back to default behavior if elicitation fails

        # Create the initial expertise identification thought
        if self._use_sampling:
            content = await self._sample_expertise_identification(
                input_text, context, decomposition_strategy
            )
        else:
            content = self._generate_expertise_identification(
                input_text, context, decomposition_strategy
            )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DECOMPOSED_PROMPTING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,  # Good initial confidence for expertise identification
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "decomposed_prompting",
                "stage": "expertise_identification",
                "specialists_identified": len(self._specialists),
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DECOMPOSED_PROMPTING

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

        This method generates the next step in the decomposed prompting process:
        - After expertise identification: Assign specialist roles and create sub-prompts
        - During specialist execution: Execute each specialist task
        - After all specialists: Integrate outputs into unified solution

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = DecomposedPrompting()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Analyze problem")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Assign specialists"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError(
                "Decomposed Prompting method must be initialized before continuation"
            )

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine the stage and generate appropriate content
        stage = previous_thought.metadata.get("stage", "expertise_identification")

        if stage == "expertise_identification" and not self._specialists_assigned:
            # Next step: Assign specialist roles and create sub-prompts
            if self._use_sampling:
                content, new_stage = await self._sample_specialist_assignment(
                    previous_thought, guidance, context
                )
            else:
                content, new_stage = self._generate_specialist_assignment(
                    previous_thought, guidance, context
                )
            self._specialists_assigned = True
        elif len(self._specialist_outputs) < len(self._specialists):
            # Specialist execution phase: Execute next specialist task
            if self._use_sampling:
                content, new_stage = await self._sample_specialist_execution(
                    previous_thought, guidance, context
                )
            else:
                content, new_stage = self._generate_specialist_execution(
                    previous_thought, guidance, context
                )
        else:
            # Integration phase: Combine all specialist outputs
            if self._use_sampling:
                content, new_stage = await self._sample_integration(
                    previous_thought, guidance, context
                )
            else:
                content, new_stage = self._generate_integration(previous_thought, guidance, context)

        # Determine thought type based on stage
        if new_stage == "integration":
            thought_type = ThoughtType.SYNTHESIS
        elif new_stage == "specialist_execution":
            # Use BRANCH type for parallel specialist execution
            thought_type = ThoughtType.BRANCH
        elif len(self._specialist_outputs) == len(self._specialists):
            thought_type = ThoughtType.CONCLUSION
        else:
            thought_type = ThoughtType.CONTINUATION

        # Calculate confidence (increases as we complete more specialist tasks)
        progress = len(self._specialist_outputs) / max(1, len(self._specialists))
        confidence = 0.7 + (0.2 * progress)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.DECOMPOSED_PROMPTING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "decomposed_prompting",
                "stage": new_stage,
                "specialists_total": len(self._specialists),
                "specialists_completed": len(self._specialist_outputs),
                "progress": f"{len(self._specialist_outputs)}/{len(self._specialists)}",
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Decomposed Prompting, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = DecomposedPrompting()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_expertise_identification(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        decomposition_strategy: str | None = None,
    ) -> str:
        """Generate the initial expertise identification.

        This method analyzes the problem and identifies the required
        specialist domains and expertise areas.

        Args:
            input_text: The problem to analyze
            context: Optional additional context
            decomposition_strategy: Optional user-selected decomposition strategy

        Returns:
            The expertise identification thought content

        Note:
            In a full implementation, this would use an LLM to intelligently
            identify required domains. This is a placeholder structure.
        """
        # Simulate expertise identification (in real implementation, LLM would do this)
        # Adjust specialists based on decomposition strategy if provided
        if decomposition_strategy == "sequential":
            self._specialists = [
                {
                    "role": "Planning Specialist",
                    "domain": "Strategic Planning",
                    "focus": "Define overall approach and sequencing",
                },
                {
                    "role": "Execution Specialist",
                    "domain": "Implementation",
                    "focus": "Execute sequential steps in order",
                },
                {
                    "role": "Validation Specialist",
                    "domain": "Quality Assurance",
                    "focus": "Verify each step before proceeding",
                },
            ]
        elif decomposition_strategy == "hierarchical":
            self._specialists = [
                {
                    "role": "Top-Level Architect",
                    "domain": "High-Level Design",
                    "focus": "Overall system architecture and main components",
                },
                {
                    "role": "Mid-Level Designer",
                    "domain": "Subsystem Design",
                    "focus": "Detailed design of major subsystems",
                },
                {
                    "role": "Low-Level Implementer",
                    "domain": "Component Implementation",
                    "focus": "Implementation details of individual components",
                },
            ]
        elif decomposition_strategy == "parallel":
            self._specialists = [
                {
                    "role": "Parallel Task Coordinator",
                    "domain": "Task Management",
                    "focus": "Identify and coordinate independent parallel tasks",
                },
                {
                    "role": "Resource Specialist",
                    "domain": "Resource Allocation",
                    "focus": "Allocate resources across parallel workstreams",
                },
                {
                    "role": "Integration Specialist",
                    "domain": "Integration & Synchronization",
                    "focus": "Combine results from parallel tasks",
                },
            ]
        else:
            # Default: domain experts (original behavior)
            self._specialists = [
                {
                    "role": "Technical Architect",
                    "domain": "System Architecture & Design",
                    "focus": "Overall system structure and technical feasibility",
                },
                {
                    "role": "Domain Expert",
                    "domain": "Domain-Specific Knowledge",
                    "focus": "Industry-specific requirements and best practices",
                },
                {
                    "role": "Implementation Specialist",
                    "domain": "Development & Engineering",
                    "focus": "Practical implementation and technical details",
                },
            ]

        specialists_text = "\n".join(
            f"   {i + 1}. {sp['role']} ({sp['domain']})\n      Focus: {sp['focus']}"
            for i, sp in enumerate(self._specialists)
        )

        strategy_desc = ""
        if decomposition_strategy:
            strategy_map = {
                "sequential": "sequential steps with ordered execution",
                "hierarchical": "hierarchical levels from high-level to low-level",
                "parallel": "parallel independent tasks",
                "domain": "domain-specific expertise areas",
            }
            strategy_text = strategy_map.get(decomposition_strategy, "domain experts")
            strategy_desc = f"Decomposition Strategy: {strategy_text}\n\n"

        return (
            f"Step {self._step_counter}: Expertise Identification & Problem Analysis\n\n"
            f"Main Problem: {input_text}\n\n"
            f"{strategy_desc}"
            f"This problem requires expertise from multiple domains. I will decompose "
            f"it into specialist sub-tasks, with each expert focusing on their area "
            f"of expertise. This ensures comprehensive coverage while leveraging "
            f"domain-specific knowledge.\n\n"
            f"Required Specialists:\n{specialists_text}\n\n"
            f"Next: I will assign specific sub-tasks to each specialist and create "
            f"domain-specific prompts for their analysis."
        )

    def _generate_specialist_assignment(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate the specialist assignment and sub-prompt creation.

        This method creates specific sub-tasks and prompts for each specialist.

        Args:
            previous_thought: The expertise identification thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        assignments = []
        for i, specialist in enumerate(self._specialists):
            assignment = (
                f"   {i + 1}. {specialist['role']}\n"
                f"      Domain: {specialist['domain']}\n"
                f"      Task: Analyze the problem from a {specialist['domain'].lower()} "
                f"perspective, focusing on {specialist['focus'].lower()}.\n"
                f"      Deliverable: Detailed analysis and recommendations for your domain."
            )
            assignments.append(assignment)

        assignments_text = "\n\n".join(assignments)

        content = (
            f"Step {self._step_counter}: Specialist Assignment & Sub-Task Creation\n\n"
            f"I've created specific sub-tasks for each specialist, with domain-specific "
            f"prompts tailored to their expertise. Each specialist will work independently "
            f"on their area, ensuring depth of analysis.\n\n"
            f"Specialist Assignments:\n{assignments_text}\n\n"
            f"Next: Execute each specialist task. These can be performed in parallel "
            f"as they are independent analyses."
        )

        return content, "specialist_assignment"

    def _generate_specialist_execution(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate execution output for the next specialist.

        This method executes the next specialist sub-task.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        specialist_index = len(self._specialist_outputs)
        current_specialist = self._specialists[specialist_index]
        specialist_role = current_specialist["role"]

        # Create specialist output
        output = (
            f"[{specialist_role} Analysis]\n\n"
            f"Domain Perspective: {current_specialist['domain']}\n\n"
            f"Key Findings:\n"
            f"- Consideration 1: Analyzed from {current_specialist['focus'].lower()} viewpoint\n"
            f"- Consideration 2: Domain-specific insights and constraints\n"
            f"- Consideration 3: Recommendations based on "
            f"{current_specialist['domain']} expertise\n\n"
            f"Recommendations: Specific actionable guidance from the {specialist_role} perspective."
        )

        self._specialist_outputs[specialist_role] = output

        # Determine next action message
        if len(self._specialist_outputs) < len(self._specialists):
            next_action = "Next: Execute remaining specialist tasks."
        else:
            next_action = "All specialist analyses complete. Next: Integration."

        content = (
            f"Step {self._step_counter}: {specialist_role} Analysis\n\n"
            f"Executing specialist sub-task for {current_specialist['domain']}.\n\n"
            f"{output}\n\n"
            f"Progress: {len(self._specialist_outputs)}/{len(self._specialists)} "
            f"specialists completed.\n\n"
            f"{next_action}"
        )

        return content, "specialist_execution"

    def _generate_integration(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate the final integration of all specialist outputs.

        This method combines all specialist analyses into a unified solution.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        specialist_summaries = []
        for role, _output in self._specialist_outputs.items():
            # Extract key points from each specialist output
            summary = f"   - {role}: Domain-specific insights integrated"
            specialist_summaries.append(summary)

        summaries_text = "\n".join(specialist_summaries)

        content = (
            f"Step {self._step_counter}: Expert Integration & Synthesis\n\n"
            f"All specialist analyses are now complete. I will integrate their outputs "
            f"into a comprehensive, unified solution that leverages insights from all domains.\n\n"
            f"Specialist Contributions:\n{summaries_text}\n\n"
            f"Integrated Solution:\n"
            f"By combining expertise from {len(self._specialists)} different domains, "
            f"we achieve a comprehensive solution that addresses:\n"
            f"- Technical feasibility and architecture\n"
            f"- Domain-specific requirements and best practices\n"
            f"- Practical implementation considerations\n\n"
            f"The multi-specialist approach ensures that no single perspective dominates, "
            f"resulting in a balanced and well-informed solution. Each domain's concerns "
            f"have been addressed by qualified experts, and their insights have been "
            f"integrated to resolve any conflicts and identify synergies.\n\n"
            f"Final Recommendation: A unified solution that incorporates all specialist "
            f"recommendations while maintaining coherence and feasibility across all domains."
        )

        return content, "integration"

    async def _sample_expertise_identification(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        decomposition_strategy: str | None = None,
    ) -> str:
        """Generate expertise identification using LLM sampling.

        Uses the execution context's sampling capability to identify
        required specialists and domains.

        Args:
            input_text: The problem to analyze
            context: Optional additional context
            decomposition_strategy: Optional user-selected decomposition strategy

        Returns:
            The expertise identification thought content
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        strategy_guidance = ""
        if decomposition_strategy == "sequential":
            strategy_guidance = "\nDecomposition Strategy: Focus on sequential, ordered steps."
        elif decomposition_strategy == "hierarchical":
            strategy_guidance = (
                "\nDecomposition Strategy: Focus on hierarchical levels (high to low)."
            )
        elif decomposition_strategy == "parallel":
            strategy_guidance = "\nDecomposition Strategy: Focus on parallel, independent tasks."

        system_prompt = f"""You are a reasoning assistant using Decomposed Prompting methodology.
Analyze multi-disciplinary problems and identify required specialist domains.
{strategy_guidance}

Your task:
1. Analyze the problem and identify 2-4 specialist domains needed
2. For each domain, specify:
   - Role (e.g., "Technical Architect", "Domain Expert")
   - Domain area (e.g., "System Architecture & Design")
   - Focus area (e.g., "Overall system structure and technical feasibility")

Format your response as:
- Problem analysis
- List of specialists with their roles, domains, and focus areas
- Explanation of why these specialists are needed"""

        user_prompt = f"""Problem: {input_text}

Identify the specialist domains needed to solve this multi-disciplinary problem.
List 2-4 specialists with their roles, domains, and focus areas."""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            content = str(result) if not isinstance(result, str) else result

            # Parse specialists from the response
            # (simplified - in production would use structured output)
            # For now, use default specialists but include the LLM's analysis
            self._specialists = [
                {
                    "role": "Technical Architect",
                    "domain": "System Architecture & Design",
                    "focus": "Overall system structure and technical feasibility",
                },
                {
                    "role": "Domain Expert",
                    "domain": "Domain-Specific Knowledge",
                    "focus": "Industry-specific requirements and best practices",
                },
                {
                    "role": "Implementation Specialist",
                    "domain": "Development & Engineering",
                    "focus": "Practical implementation and technical details",
                },
            ]

            return (
                f"Step {self._step_counter}: Expertise Identification & Problem Analysis\n\n"
                f"{content}\n\n"
                f"Next: I will assign specific sub-tasks to each specialist and create "
                f"domain-specific prompts for their analysis."
            )
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_expertise_identification",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_expertise_identification(
                input_text, context, decomposition_strategy
            )

    async def _sample_specialist_assignment(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate specialist assignment using LLM sampling.

        Uses the execution context's sampling capability to create
        specific sub-tasks and prompts for each specialist.

        Args:
            previous_thought: The expertise identification thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        specialists_desc = "\n".join(
            f"{i + 1}. {sp['role']} - {sp['domain']}" for i, sp in enumerate(self._specialists)
        )

        system_prompt = """You are a reasoning assistant using Decomposed Prompting methodology.
Create specific sub-tasks for each specialist domain.

For each specialist:
1. Define a clear, specific task within their domain
2. Specify what deliverables they should produce
3. Ensure tasks are independent and can be executed in parallel"""

        user_prompt = f"""Specialists identified:
{specialists_desc}

Problem context: {previous_thought.metadata.get("input", "See previous analysis")}

Create specific sub-task assignments for each specialist with clear deliverables."""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            content = str(result) if not isinstance(result, str) else result

            return (
                f"Step {self._step_counter}: Specialist Assignment & Sub-Task Creation\n\n"
                f"{content}\n\n"
                f"Next: Execute each specialist task. These can be performed in parallel "
                f"as they are independent analyses."
            ), "specialist_assignment"
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_specialist_assignment",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_specialist_assignment(previous_thought, guidance, context)

    async def _sample_specialist_execution(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate specialist execution using LLM sampling.

        Uses the execution context's sampling capability to execute
        the next specialist's sub-task.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        specialist_index = len(self._specialist_outputs)
        current_specialist = self._specialists[specialist_index]
        specialist_role = current_specialist["role"]

        specialist_domain = current_specialist["domain"]
        system_prompt = f"""You are a {specialist_role} specialist with expertise in \
{specialist_domain}.
Analyze the problem from your domain's perspective.

Focus on: {current_specialist["focus"]}

Provide:
1. Domain-specific analysis
2. Key findings and insights
3. Recommendations based on your expertise
4. Any constraints or considerations from your domain"""

        problem_input = previous_thought.metadata.get("input", "See previous context")
        focus_lower = current_specialist["focus"].lower()
        user_prompt = f"""Problem: {problem_input}

As a {specialist_role}, analyze this problem focusing on {focus_lower}.
Provide detailed analysis and actionable recommendations from your domain's perspective."""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1200,
            )
            output = str(result) if not isinstance(result, str) else result

            self._specialist_outputs[specialist_role] = output

            next_msg = (
                "Next: Execute remaining specialist tasks."
                if len(self._specialist_outputs) < len(self._specialists)
                else "All specialist analyses complete. Next: Integration."
            )
            return (
                f"Step {self._step_counter}: {specialist_role} Analysis\n\n"
                f"{output}\n\n"
                f"Progress: {len(self._specialist_outputs)}/{len(self._specialists)} "
                f"specialists completed.\n\n"
                f"{next_msg}"
            ), "specialist_execution"
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_specialist_execution",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_specialist_execution(previous_thought, guidance, context)

    async def _sample_integration(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate integration using LLM sampling.

        Uses the execution context's sampling capability to integrate
        all specialist outputs into a unified solution.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Tuple of (content, new_stage)
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for LLM sampling")

        # Compile all specialist outputs
        specialist_analyses = []
        for role, output in self._specialist_outputs.items():
            specialist_analyses.append(f"=== {role} ===\n{output}\n")

        all_analyses = "\n".join(specialist_analyses)

        system_prompt = """You are a reasoning assistant performing expert integration and \
synthesis.
Combine multiple specialist analyses into a comprehensive, unified solution.

Your task:
1. Identify key insights from each specialist
2. Resolve any conflicts or contradictions
3. Identify synergies between different domains
4. Create an integrated solution that addresses all concerns
5. Provide a final, actionable recommendation

Ensure the solution:
- Balances all perspectives
- Maintains technical and practical feasibility
- Addresses concerns from all domains"""

        num_specialists = len(self._specialists)
        user_prompt = f"""Specialist Analyses to Integrate:

{all_analyses}

Create a comprehensive, unified solution that integrates insights from all \
{num_specialists} specialists.
Resolve conflicts, identify synergies, and provide a final recommendation."""

        try:
            result = await self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1500,
            )
            content = str(result) if not isinstance(result, str) else result

            return (
                f"Step {self._step_counter}: Expert Integration & Synthesis\n\n{content}"
            ), "integration"
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "llm_sampling_failed",
                method="_sample_integration",
                error=str(e),
            )
            # Fallback to placeholder on sampling failure
            return self._generate_integration(previous_thought, guidance, context)
