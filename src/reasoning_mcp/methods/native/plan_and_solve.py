"""Plan-and-Solve reasoning method.

This module implements a structured reasoning method that adds an explicit planning
phase before solving. The method follows four distinct phases:
1. Understand the problem - analyze and comprehend the problem
2. Plan - create a step-by-step plan to solve it
3. Execute - carry out each step of the plan sequentially
4. Synthesize - combine results into final answer

This approach is based on Wang et al. (2023) "Plan-and-Solve Prompting" which
improves reasoning by explicitly decomposing the problem-solving process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
from reasoning_mcp.methods.base import (
    DEFAULT_MAX_TOKENS,
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


# Metadata for Plan-and-Solve method
PLAN_AND_SOLVE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.PLAN_AND_SOLVE,
    name="Plan-and-Solve",
    description=(
        "Explicit planning phase before solving with step decomposition. "
        "Understands problem, creates plan, executes steps sequentially, "
        "then synthesizes final answer."
    ),
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "planning",
            "structured",
            "step-by-step",
            "decomposition",
            "sequential",
            "problem-solving",
            "systematic",
        }
    ),
    complexity=4,  # Medium complexity - requires planning and execution tracking
    supports_branching=False,  # Linear plan execution
    supports_revision=False,  # No revision - follows plan
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: understand + plan + execute + synthesize
    max_thoughts=15,  # Plan can have multiple execution steps
    avg_tokens_per_thought=350,  # Moderate - includes plan and step details
    best_for=(
        "complex problem solving",
        "multi-step tasks",
        "mathematical problems",
        "structured analysis",
        "systematic reasoning",
        "step decomposition",
    ),
    not_recommended_for=(
        "simple factual queries",
        "creative ideation",
        "open-ended exploration",
        "problems requiring flexibility",
    ),
)

logger = structlog.get_logger(__name__)


class PlanAndSolve(ReasoningMethodBase):
    """Plan-and-Solve reasoning method implementation.

    This class implements a structured reasoning pattern with explicit planning:
    1. Understand: Analyze and comprehend the problem thoroughly
    2. Plan: Create a detailed step-by-step plan
    3. Execute: Carry out each step of the plan sequentially
    4. Synthesize: Combine all results into a final answer

    Key characteristics:
    - Explicit planning phase
    - Step decomposition
    - Sequential execution
    - Structured problem-solving
    - Medium complexity (4)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = PlanAndSolve()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Solve the equation: 2x + 5 = 13"
        ... )
        >>> print(result.content)  # Understanding phase

        Continue with planning:
        >>> plan = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Create detailed plan"
        ... )
        >>> print(plan.type)  # ThoughtType.CONTINUATION (plan phase)

        Continue with execution:
        >>> execute = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=plan,
        ...     guidance="Execute plan"
        ... )
        >>> print(execute.type)  # ThoughtType.CONTINUATION (execute phase)

        Synthesize final answer:
        >>> final = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=execute,
        ...     guidance="Synthesize answer"
        ... )
        >>> print(final.type)  # ThoughtType.CONCLUSION (synthesize phase)
    """

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Plan-and-Solve method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "understand"  # understand, plan, execute, synthesize
        self._plan_steps: list[str] = []  # List of planned steps
        self._current_step_index: int = -1  # Which plan step is being executed (-1 = none)
        self._use_sampling: bool = False  # Whether to use LLM sampling
        self._execution_context: ExecutionContext | None = None  # Execution context for sampling
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.PLAN_AND_SOLVE

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return PLAN_AND_SOLVE_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return PLAN_AND_SOLVE_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Plan-and-Solve method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = PlanAndSolve()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "understand"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "understand"
        self._plan_steps = []
        self._current_step_index = -1

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Plan-and-Solve method.

        This method creates the initial understanding phase where the problem
        is analyzed and comprehended before planning begins.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the problem understanding

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = PlanAndSolve()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Solve: 3x - 7 = 14"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.PLAN_AND_SOLVE
            >>> assert thought.metadata["phase"] == "understand"
        """
        if not self._initialized:
            raise RuntimeError("Plan-and-Solve method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "understand"
        self._plan_steps = []
        self._current_step_index = -1

        # Optional elicitation: ask user about planning approach
        planning_approach: str | None = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "detailed", "label": "Detailed plan with many steps"},
                    {"id": "high_level", "label": "High-level plan with fewer steps"},
                    {"id": "adaptive", "label": "Adaptive plan that adjusts as needed"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "What planning approach should we use?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    planning_approach = selection.selected
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
                # Fall back to default behavior

        # Generate understanding of the problem (use sampling if available)
        if self._use_sampling:
            content = await self._sample_understanding(input_text, context)
        else:
            content = self._generate_understanding(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.PLAN_AND_SOLVE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Moderate confidence in initial understanding
            quality_score=0.6,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "plan_and_solve",
                "phase": self._current_phase,
                "plan_steps": [],
                "current_step_index": self._current_step_index,
                "sampled": self._use_sampling,
                "planning_approach": planning_approach,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.PLAN_AND_SOLVE

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

        This method implements the phase progression logic:
        - understand → plan (create step-by-step plan)
        - plan → execute (execute first step)
        - execute → execute (execute next step) OR execute → synthesize (if done)
        - synthesize is the final phase (CONCLUSION)

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the plan-and-solve process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = PlanAndSolve()
            >>> await method.initialize()
            >>> understand = await method.execute(session, "Solve: 2x + 3 = 7")
            >>> plan = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=understand
            ... )
            >>> assert plan.metadata["phase"] == "plan"
            >>> assert len(plan.metadata["plan_steps"]) > 0
            >>>
            >>> execute = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=plan
            ... )
            >>> assert execute.metadata["phase"] == "execute"
            >>> assert execute.metadata["current_step_index"] == 0
        """
        if not self._initialized:
            raise RuntimeError("Plan-and-Solve method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Get previous phase
        prev_phase = previous_thought.metadata.get("phase", "understand")

        # Determine next phase and generate content
        if prev_phase == "understand":
            # Next: create the plan
            self._current_phase = "plan"
            thought_type = ThoughtType.CONTINUATION

            # Generate plan (use sampling if available)
            if self._use_sampling:
                content = await self._sample_plan(previous_thought, guidance, context)
            else:
                content = self._generate_plan(previous_thought, guidance, context)

            # Extract plan steps (in real implementation, LLM would generate these)
            # For now, generate 3-5 placeholder steps
            self._plan_steps = self._extract_plan_steps(content)

            confidence = 0.75
            quality_score = 0.7
            depth = previous_thought.depth + 1

        elif prev_phase == "plan":
            # Next: execute first step
            self._current_phase = "execute"
            self._current_step_index = 0
            self._plan_steps = previous_thought.metadata.get("plan_steps", [])

            thought_type = ThoughtType.CONTINUATION

            # Generate execution (use sampling if available)
            if self._use_sampling:
                content = await self._sample_execution(
                    previous_thought, self._current_step_index, guidance, context
                )
            else:
                content = self._generate_execution(
                    previous_thought, self._current_step_index, guidance, context
                )

            confidence = 0.8
            quality_score = 0.75
            depth = previous_thought.depth + 1

        elif prev_phase == "execute":
            # Check if more steps to execute or ready to synthesize
            self._plan_steps = previous_thought.metadata.get("plan_steps", [])
            prev_step_index = previous_thought.metadata.get("current_step_index", -1)
            self._current_step_index = prev_step_index + 1

            if self._current_step_index < len(self._plan_steps):
                # Continue execution with next step
                self._current_phase = "execute"
                thought_type = ThoughtType.CONTINUATION

                # Generate execution (use sampling if available)
                if self._use_sampling:
                    content = await self._sample_execution(
                        previous_thought, self._current_step_index, guidance, context
                    )
                else:
                    content = self._generate_execution(
                        previous_thought, self._current_step_index, guidance, context
                    )

                confidence = 0.85
                quality_score = 0.8
                depth = previous_thought.depth + 1
            else:
                # All steps complete, synthesize
                self._current_phase = "synthesize"
                thought_type = ThoughtType.CONCLUSION

                # Generate synthesis (use sampling if available)
                if self._use_sampling:
                    content = await self._sample_synthesis(previous_thought, guidance, context)
                else:
                    content = self._generate_synthesis(previous_thought, guidance, context)

                confidence = 0.9
                quality_score = 0.85
                depth = previous_thought.depth + 1

        elif prev_phase == "synthesize":
            # Should not continue after synthesis, but handle gracefully
            self._current_phase = "synthesize"
            thought_type = ThoughtType.CONCLUSION
            content = "Plan-and-Solve process complete. Final answer has been synthesized."
            confidence = 0.9
            quality_score = 0.85
            depth = previous_thought.depth

        else:
            # Fallback to plan phase
            self._current_phase = "plan"
            thought_type = ThoughtType.CONTINUATION

            # Generate plan (use sampling if available)
            if self._use_sampling:
                content = await self._sample_plan(previous_thought, guidance, context)
            else:
                content = self._generate_plan(previous_thought, guidance, context)

            self._plan_steps = self._extract_plan_steps(content)
            confidence = 0.75
            quality_score = 0.7
            depth = previous_thought.depth + 1

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.PLAN_AND_SOLVE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=depth,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "plan_steps": self._plan_steps,
                "current_step_index": self._current_step_index,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "plan_and_solve",
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Plan-and-Solve, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = PlanAndSolve()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_understanding(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the problem understanding.

        This is a helper method that would typically call an LLM to analyze
        and understand the problem before planning.

        Args:
            input_text: The problem or question to understand
            context: Optional additional context

        Returns:
            The content for the understanding phase

        Note:
            In a full implementation, this would use an LLM to generate
            the actual understanding. This is a placeholder that provides
            the structure.
        """
        return (
            f"Step {self._step_counter}: Understanding the Problem\n\n"
            f"Problem: {input_text}\n\n"
            f"Let me analyze and understand this problem before creating a plan.\n\n"
            f"Problem Analysis:\n"
            f"[LLM would analyze the problem, identify key components, constraints, and goals]\n\n"
            f"Key Components:\n"
            f"[LLM would list main elements of the problem]\n\n"
            f"Constraints:\n"
            f"[LLM would identify any constraints or limitations]\n\n"
            f"Goal:\n"
            f"[LLM would state the clear objective]\n\n"
            f"Next: Create a step-by-step plan to solve this problem."
        )

    def _generate_plan(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the step-by-step plan.

        This is a helper method that would typically call an LLM to create
        a detailed plan for solving the problem.

        Args:
            previous_thought: The understanding to base the plan on
            guidance: Optional guidance for planning
            context: Optional additional context

        Returns:
            The content for the plan phase

        Note:
            In a full implementation, this would use an LLM to generate
            the actual plan. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Creating Step-by-Step Plan\n\n"
            f"Based on my understanding in Step {previous_thought.step_number}, "
            f"I will create a detailed plan to solve this problem.\n\n"
            f"Plan:\n"
            f"[STEP 1] Identify the key variables and given information\n"
            f"[STEP 2] Set up the equation or framework\n"
            f"[STEP 3] Apply the appropriate solving technique\n"
            f"[STEP 4] Verify the solution\n\n"
            f"Note: [LLM would generate a custom plan based on the specific problem]\n\n"
            f"Next: Execute each step of the plan sequentially.{guidance_text}"
        )

    def _extract_plan_steps(self, plan_content: str) -> list[str]:
        """Extract plan steps from the plan content.

        In a real implementation, this would parse the LLM-generated plan.
        For now, returns a reasonable default set of steps.

        Args:
            plan_content: The plan content to extract steps from

        Returns:
            List of plan steps
        """
        # In real implementation, parse LLM output
        # For now, return placeholder steps
        return [
            "Identify key variables and given information",
            "Set up the equation or framework",
            "Apply the appropriate solving technique",
            "Verify the solution",
        ]

    def _generate_execution(
        self,
        previous_thought: ThoughtNode,
        step_index: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate execution of a specific plan step.

        This is a helper method that would typically call an LLM to execute
        the current step of the plan.

        Args:
            previous_thought: The previous thought (plan or previous execution)
            step_index: Index of the step to execute
            guidance: Optional guidance for execution
            context: Optional additional context

        Returns:
            The content for the execution phase

        Note:
            In a full implementation, this would use an LLM to generate
            the actual execution. This is a placeholder that provides
            the structure.
        """
        plan_steps = previous_thought.metadata.get("plan_steps", [])
        current_step = plan_steps[step_index] if step_index < len(plan_steps) else "Unknown step"
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Executing Plan Step {step_index + 1}\n\n"
            f"Current Step: {current_step}\n\n"
            f"Execution:\n"
            f"[LLM would execute this specific step, showing work and reasoning]\n\n"
            f"Step Result:\n"
            f"[LLM would provide the result/outcome of this step]\n\n"
            f"Progress: Step {step_index + 1} of {len(plan_steps)} complete.\n\n"
            f"Next: "
            f"{'Execute next step' if step_index + 1 < len(plan_steps) else 'Synthesize final answer'}"  # noqa: E501
            f"{guidance_text}"
        )

    def _generate_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis.

        This is a helper method that would typically call an LLM to synthesize
        all execution results into a final answer.

        Args:
            previous_thought: The last execution step
            guidance: Optional guidance for synthesis
            context: Optional additional context

        Returns:
            The content for the synthesis phase

        Note:
            In a full implementation, this would use an LLM to generate
            the actual synthesis. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Synthesizing Final Answer\n\n"
            f"Combining all execution results from the plan steps...\n\n"
            f"Summary of Execution:\n"
            f"[LLM would summarize what was accomplished in each step]\n\n"
            f"Final Answer:\n"
            f"[LLM would provide the complete, synthesized solution]\n\n"
            f"Verification:\n"
            f"[LLM would verify the answer makes sense and satisfies problem requirements]\n\n"
            f"Plan-and-Solve process complete.{guidance_text}"
        )

    async def _sample_understanding(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate problem understanding using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual problem analysis rather than placeholder content.

        Args:
            input_text: The problem or question to understand
            context: Optional additional context

        Returns:
            The content for the understanding phase

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_understanding but was not provided"
            )

        system_prompt = """You are a reasoning assistant using Plan-and-Solve methodology.
Your task is to analyze and understand the problem thoroughly before planning.

In the UNDERSTANDING phase:
1. Restate the problem clearly
2. Identify key components, variables, and relationships
3. Note any constraints or limitations
4. State the clear objective/goal
5. Identify what information is given and what needs to be found

Be thorough but concise. This understanding will guide the planning phase."""

        user_prompt = f"""Problem: {input_text}

Analyze and understand this problem. Identify:
- Key components and variables
- Given information
- Constraints
- The goal/objective
- What needs to be determined

Step {self._step_counter}: Understanding the Problem"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_understanding(input_text, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )
        return f"Step {step_counter}: Understanding the Problem\n\n{content}"

    async def _sample_plan(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate step-by-step plan using LLM sampling.

        Uses the execution context's sampling capability to generate
        an actual detailed plan rather than placeholder content.

        Args:
            previous_thought: The understanding to base the plan on
            guidance: Optional guidance for planning
            context: Optional additional context

        Returns:
            The content for the plan phase

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_plan but was not provided"
            )

        system_prompt = """You are a reasoning assistant using Plan-and-Solve methodology.
Your task is to create a detailed, step-by-step plan for solving the problem.

In the PLANNING phase:
1. Break down the problem into logical, sequential steps
2. Each step should be clear and actionable
3. Number the steps clearly (e.g., [STEP 1], [STEP 2], etc.)
4. Ensure steps build upon each other
5. Include 3-5 steps typically
6. The plan should lead to a complete solution

Be specific and systematic. Each step will be executed individually."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        input_text = previous_thought.metadata.get("input", "the problem")

        user_prompt = f"""Based on the understanding of this problem:

{previous_thought.content}

Create a detailed step-by-step plan to solve: {input_text}

Format each step clearly as [STEP 1], [STEP 2], etc.
Ensure steps are logical, sequential, and actionable.{guidance_text}

Step {self._step_counter}: Creating Step-by-Step Plan"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_plan(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )
        return f"Step {step_counter}: Creating Step-by-Step Plan\n\n{content}"

    async def _sample_execution(
        self,
        previous_thought: ThoughtNode,
        step_index: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate execution of a specific plan step using LLM sampling.

        Uses the execution context's sampling capability to execute
        the current step of the plan.

        Args:
            previous_thought: The previous thought (plan or previous execution)
            step_index: Index of the step to execute
            guidance: Optional guidance for execution
            context: Optional additional context

        Returns:
            The content for the execution phase

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_execution but was not provided"
            )

        plan_steps = previous_thought.metadata.get("plan_steps", [])
        current_step = plan_steps[step_index] if step_index < len(plan_steps) else "Unknown step"

        system_prompt = """You are a reasoning assistant using Plan-and-Solve methodology.
Your task is to execute a specific step of the plan.

In the EXECUTION phase:
1. Focus on completing the current step thoroughly
2. Show your work and reasoning
3. Provide the result/outcome of this step
4. Be explicit about calculations, logic, or analysis
5. Verify the step was completed correctly

Execute the step systematically and show all work."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        user_prompt = f"""Execute the following step from the plan:

Current Step ({step_index + 1} of {len(plan_steps)}): {current_step}

Show your work, reasoning, and the result of completing this step.{guidance_text}

Step {self._step_counter}: Executing Plan Step {step_index + 1}"""

        step_counter = self._step_counter
        num_plan_steps = len(plan_steps)

        def fallback() -> str:
            return self._generate_execution(previous_thought, step_index, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=1200,
        )
        return (
            f"Step {step_counter}: Executing Plan Step {step_index + 1}\n\n{content}\n\n"
            f"Progress: Step {step_index + 1} of {num_plan_steps} complete."
        )

    async def _sample_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final synthesis using LLM sampling.

        Uses the execution context's sampling capability to synthesize
        all execution results into a final answer.

        Args:
            previous_thought: The last execution step
            guidance: Optional guidance for synthesis
            context: Optional additional context

        Returns:
            The content for the synthesis phase

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_synthesis but was not provided"
            )

        system_prompt = """You are a reasoning assistant using Plan-and-Solve methodology.
Your task is to synthesize all plan execution results into a final answer.

In the SYNTHESIS phase:
1. Summarize what was accomplished in each step
2. Combine results into a coherent final answer
3. Verify the answer makes sense and satisfies requirements
4. State the final answer clearly
5. Provide confidence in the solution

Be comprehensive yet concise in your synthesis."""

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""
        plan_steps = previous_thought.metadata.get("plan_steps", [])

        user_prompt = f"""All {len(plan_steps)} plan steps have been executed.

Plan steps that were completed:
{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(plan_steps))}

Synthesize the results from all execution steps into a final answer.
Provide:
- Summary of what was accomplished
- The final answer
- Verification that it satisfies the original problem{guidance_text}

Step {self._step_counter}: Synthesizing Final Answer"""

        step_counter = self._step_counter

        def fallback() -> str:
            return self._generate_synthesis(previous_thought, guidance, context)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return (
            f"Step {step_counter}: Synthesizing Final Answer\n\n{content}\n\n"
            "Plan-and-Solve process complete."
        )
