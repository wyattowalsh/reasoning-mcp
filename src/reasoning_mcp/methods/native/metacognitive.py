"""Metacognitive reasoning method.

This module implements metacognitive reasoning - "thinking about thinking" that
monitors and regulates one's own cognitive processes during reasoning. This method
enables self-awareness of reasoning strategies, progress monitoring, quality evaluation,
and adaptive strategy adjustment.
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


# Metadata for Metacognitive method
METACOGNITIVE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.METACOGNITIVE,
    name="Metacognitive Reasoning",
    description="Thinking about thinking - monitors and regulates own cognitive processes. "
    "Plans approach, monitors progress, evaluates quality, and adapts strategy as needed.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "metacognitive",
            "self-monitoring",
            "learning",
            "adaptive",
            "strategy",
            "self-awareness",
            "regulation",
        }
    ),
    complexity=7,  # High complexity - requires meta-level reasoning
    supports_branching=True,  # Can branch to try alternative strategies
    supports_revision=True,  # Can revise approach based on monitoring
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: plan, monitor, evaluate, regulate
    max_thoughts=0,  # No limit - adaptive cycles can continue
    avg_tokens_per_thought=450,  # Detailed meta-analysis
    best_for=(
        "learning optimization",
        "strategy selection",
        "self-improvement",
        "adaptive reasoning",
        "complex problem solving",
        "performance optimization",
        "error correction",
    ),
    not_recommended_for=(
        "simple factual questions",
        "time-critical tasks",
    ),
)


class MetacognitiveMethod(ReasoningMethodBase):
    """Metacognitive reasoning method implementation.

    This class implements metacognitive reasoning - thinking about one's own
    thinking processes. It operates at a meta-level, monitoring and regulating
    the reasoning process itself through four key phases:

    1. PLANNING: Plan the reasoning approach before starting
       - What strategy should I use?
       - What resources do I need?
       - What are potential obstacles?

    2. MONITORING: Track progress during reasoning
       - Am I on track?
       - Is this approach working?
       - Do I understand what I'm doing?

    3. EVALUATING: Assess reasoning quality
       - Was my approach effective?
       - What worked well?
       - What didn't work?

    4. REGULATING: Adjust strategy if needed
       - Should I change my approach?
       - What should I do differently?
       - What alternative strategies should I try?

    Key characteristics:
    - Meta-level awareness of reasoning process
    - Strategy planning and selection
    - Progress monitoring
    - Quality evaluation
    - Adaptive adjustment
    - High complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = MetacognitiveMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can I improve my problem-solving skills?"
        ... )
        >>> print(result.metadata["phase"])  # "planning"

        Continue with monitoring:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Monitor progress"
        ... )
        >>> print(next_thought.metadata["phase"])  # "monitoring"
    """

    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Metacognitive method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "planning"  # planning, monitoring, evaluating, regulating
        self._strategy_adjustments: int = 0
        self._metacognitive_cycle: int = 0

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.METACOGNITIVE

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return METACOGNITIVE_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return METACOGNITIVE_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Metacognitive method for execution,
        resetting internal state for meta-level reasoning.

        Examples:
            >>> method = MetacognitiveMethod()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "planning"
        self._strategy_adjustments = 0
        self._metacognitive_cycle = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Metacognitive method.

        This method creates the first thought in metacognitive reasoning,
        which is the PLANNING phase. It analyzes the problem and plans
        the reasoning approach before beginning actual problem-solving.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the planning phase

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = MetacognitiveMethod()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Solve a complex optimization problem"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.metadata["phase"] == "planning"
            >>> assert thought.method_id == MethodIdentifier.METACOGNITIVE
        """
        if not self._initialized:
            raise RuntimeError("Metacognitive method must be initialized before execution")

        # Reset state for new execution
        self._step_counter = 1
        self._current_phase = "planning"
        self._strategy_adjustments = 0
        self._metacognitive_cycle = 0

        # Determine if we should use sampling
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Generate planning content
        if use_sampling:
            if execution_context is None:
                raise RuntimeError("execution_context is required when use_sampling is True")
            self._execution_context = execution_context
            content = await self._sample_planning_content(input_text, context)
        else:
            content = self._generate_planning_content(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.METACOGNITIVE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Planning has moderate confidence
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "metacognitive",
                "phase": self._current_phase,
                "metacognitive_cycle": self._metacognitive_cycle,
                "strategy_awareness": 0.8,  # High awareness in planning phase
                "progress_monitoring": 0.0,  # Not yet monitoring
                "self_evaluation_quality": 0.0,  # Not yet evaluating
                "adaptive_adjustments": self._strategy_adjustments,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.METACOGNITIVE

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
        """Continue metacognitive reasoning from a previous thought.

        This method generates the next phase in the metacognitive cycle:
        - After PLANNING: Move to MONITORING
        - After MONITORING: Move to EVALUATING
        - After EVALUATING: Move to REGULATING
        - After REGULATING: May return to PLANNING (new cycle) or MONITORING

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "adjust strategy", "evaluate progress")
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the metacognitive process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = MetacognitiveMethod()
            >>> await method.initialize()
            >>> plan = await method.execute(session, "Complex problem")
            >>> monitor = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=plan
            ... )
            >>> assert monitor.metadata["phase"] == "monitoring"
            >>> assert monitor.parent_id == plan.id
        """
        if not self._initialized:
            raise RuntimeError("Metacognitive method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Determine next phase
        prev_phase = previous_thought.metadata.get("phase", "planning")
        self._current_phase, thought_type = self._determine_next_phase(prev_phase, guidance)

        # Generate content for the current phase
        content = self._generate_phase_content(
            previous_thought=previous_thought,
            phase=self._current_phase,
            guidance=guidance,
            context=context,
        )

        # Calculate metacognitive metrics
        metrics = self._calculate_metacognitive_metrics(
            self._current_phase,
            previous_thought,
        )

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.METACOGNITIVE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=metrics["confidence"],
            metadata={
                "previous_step": previous_thought.step_number,
                "phase": self._current_phase,
                "metacognitive_cycle": self._metacognitive_cycle,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "metacognitive",
                "strategy_awareness": metrics["strategy_awareness"],
                "progress_monitoring": metrics["progress_monitoring"],
                "self_evaluation_quality": metrics["self_evaluation_quality"],
                "adaptive_adjustments": self._strategy_adjustments,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Metacognitive method, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = MetacognitiveMethod()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_planning_content(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate planning phase content.

        Args:
            input_text: The problem to reason about
            context: Optional additional context

        Returns:
            The content for the planning thought
        """
        context_info = ""
        if context:
            context_info = f"\n\nContext: {context}"

        return f"""Step {self._step_counter}: PLANNING - Metacognitive Strategy Selection

Problem: {input_text}{context_info}

METACOGNITIVE PLANNING:
Before diving into problem-solving, I'll think about my thinking process:

1. PROBLEM ANALYSIS (Understanding what I need to do):
   - What type of problem is this?
   - What are the key challenges?
   - What does success look like?

2. STRATEGY SELECTION (Choosing how to approach it):
   - What reasoning strategies are available?
   - Which approach is most suitable for this problem?
   - What are the advantages and risks of this strategy?

3. RESOURCE IDENTIFICATION (What do I need):
   - What knowledge or information is required?
   - What tools or methods will I use?
   - What constraints should I consider?

4. GOAL SETTING (Defining my objectives):
   - What are my immediate goals?
   - What are my longer-term goals?
   - How will I know if I'm making progress?

5. ANTICIPATING OBSTACLES:
   - What difficulties might I encounter?
   - What are common pitfalls for this type of problem?
   - How can I prepare for potential challenges?

Next: I'll proceed with the chosen strategy while actively monitoring my progress."""

    async def _sample_planning_content(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate planning phase content using LLM sampling.

        Args:
            input_text: The problem to reason about
            context: Optional additional context

        Returns:
            The sampled content for the planning phase
        """
        context_info = ""
        if context:
            context_info = f"\n\nContext: {context}"

        prompt = f"""Problem: {input_text}{context_info}

Generate a metacognitive PLANNING analysis that thinks about the thinking process before solving the problem.

Address these metacognitive aspects:

1. PROBLEM ANALYSIS:
   - What type of problem is this?
   - What are the key challenges?
   - What does success look like?

2. STRATEGY SELECTION:
   - What reasoning strategies are available?
   - Which approach is most suitable?
   - What are the advantages and risks?

3. RESOURCE IDENTIFICATION:
   - What knowledge or information is required?
   - What tools or methods will be used?
   - What constraints should be considered?

4. GOAL SETTING:
   - What are the immediate goals?
   - What are the longer-term goals?
   - How to measure progress?

5. ANTICIPATING OBSTACLES:
   - What difficulties might occur?
   - What are common pitfalls?
   - How to prepare for challenges?"""

        system_prompt = f"""You are a metacognitive reasoning assistant in the PLANNING phase (Step {self._step_counter}).

Your role is to think about the thinking process itself - plan the approach before solving the problem.

Be thorough and strategic in planning the reasoning approach. Focus on meta-level analysis of how to approach the problem, not solving it yet."""

        def fallback() -> str:
            return self._generate_planning_content(input_text, context)

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        return f"""Step {self._step_counter}: PLANNING - Metacognitive Strategy Selection

{result}

Next: I'll proceed with the chosen strategy while actively monitoring my progress."""

    def _generate_monitoring_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate monitoring phase content.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for the monitoring thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return f"""Step {self._step_counter}: MONITORING - Tracking Reasoning Progress

Monitoring my reasoning process from step {previous_thought.step_number}...

PROGRESS MONITORING:
1. COMPREHENSION CHECK (Do I understand what I'm doing?):
   - Am I clear on the current approach?
   - Do I understand why I'm taking these steps?
   - Are there any confusions or uncertainties?

2. PROGRESS ASSESSMENT (Am I on track?):
   - How far have I progressed toward the goal?
   - Am I following my planned strategy?
   - Is the approach working as expected?

3. ATTENTION MANAGEMENT (Am I focused?):
   - Am I staying focused on relevant aspects?
   - Have I gotten distracted by tangential issues?
   - Should I redirect my attention?

4. PACING CHECK (Am I managing time/resources well?):
   - Am I spending appropriate time on each aspect?
   - Am I going too deep or too shallow?
   - Should I adjust my pace?

5. ERROR DETECTION (Are there any problems?):
   - Have I made any mistakes so far?
   - Are there any warning signs of errors?
   - Should I reconsider any assumptions?

Monitoring Status: {"⚠️ Need to adjust" if self._strategy_adjustments > 0 else "✓ On track"}{guidance_text}"""

    def _generate_evaluating_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate evaluating phase content.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for the evaluating thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return f"""Step {self._step_counter}: EVALUATING - Assessing Reasoning Quality

Evaluating the effectiveness of my reasoning approach from step {previous_thought.step_number}...

QUALITY EVALUATION:
1. EFFECTIVENESS ASSESSMENT (Did the approach work?):
   - Was my strategy effective for this problem?
   - Did I achieve the intended goals?
   - What was the quality of the results?

2. EFFICIENCY ANALYSIS (Was it the best approach?):
   - Could I have solved this more efficiently?
   - Did I waste effort on unproductive paths?
   - What could be streamlined?

3. STRENGTHS IDENTIFICATION (What worked well?):
   - Which aspects of my approach were successful?
   - What reasoning strategies proved valuable?
   - What should I continue doing?

4. WEAKNESSES IDENTIFICATION (What didn't work?):
   - Where did my approach fall short?
   - What mistakes or errors occurred?
   - What caused difficulties?

5. LEARNING OPPORTUNITIES (What can I learn?):
   - What insights did I gain about this problem type?
   - What metacognitive lessons can I extract?
   - How can I improve for similar problems?

Evaluation Summary:
- Strategy effectiveness: [Would assess actual effectiveness]
- Areas for improvement: [Would identify specific improvements]
- Key learnings: [Would extract lessons learned]{guidance_text}"""

    def _generate_regulating_content(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate regulating phase content.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for the regulating thought
        """
        self._strategy_adjustments += 1
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return f"""Step {self._step_counter}: REGULATING - Adapting Strategy (Adjustment #{self._strategy_adjustments})

Based on the evaluation in step {previous_thought.step_number}, I'll adapt my approach...

ADAPTIVE REGULATION:
1. STRATEGY ADJUSTMENT (Changing what isn't working):
   - What specific changes should I make?
   - Why will these changes improve performance?
   - What's my revised strategy?

2. ALTERNATIVE APPROACHES (Trying different methods):
   - What alternative strategies should I consider?
   - What are the trade-offs of each alternative?
   - Which alternative is most promising?

3. RESOURCE REALLOCATION (Adjusting effort/focus):
   - Should I allocate resources differently?
   - What deserves more or less attention?
   - How should I reprioritize?

4. GOAL REVISION (Adjusting objectives if needed):
   - Are my goals still appropriate?
   - Should I adjust expectations?
   - Do I need to reframe the problem?

5. IMPLEMENTATION PLAN (How to apply changes):
   - How will I implement these adjustments?
   - What will I do differently going forward?
   - How will I monitor the effectiveness of changes?

Regulation Decision: [Would specify concrete adjustments to make]
Expected Improvement: [Would predict how changes will help]{guidance_text}"""

    def _generate_phase_content(
        self,
        previous_thought: ThoughtNode,
        phase: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for the current metacognitive phase.

        Args:
            previous_thought: The previous thought
            phase: Current metacognitive phase
            guidance: Optional guidance
            context: Optional context

        Returns:
            The content for the thought
        """
        phase_generators = {
            "planning": lambda: self._generate_planning_content(
                previous_thought.metadata.get("input", "Continue reasoning"),
                context,
            ),
            "monitoring": lambda: self._generate_monitoring_content(
                previous_thought, guidance, context
            ),
            "evaluating": lambda: self._generate_evaluating_content(
                previous_thought, guidance, context
            ),
            "regulating": lambda: self._generate_regulating_content(
                previous_thought, guidance, context
            ),
        }

        generator = phase_generators.get(phase)
        if generator:
            return generator()

        # Fallback to generic continuation
        return f"""Step {self._step_counter}: Metacognitive Reasoning

Continuing metacognitive analysis from step {previous_thought.step_number}..."""

    def _determine_next_phase(
        self,
        prev_phase: str,
        guidance: str | None,
    ) -> tuple[str, ThoughtType]:
        """Determine the next metacognitive phase and thought type.

        Args:
            prev_phase: The previous phase
            guidance: Optional guidance

        Returns:
            Tuple of (next_phase, thought_type)
        """
        # Check guidance for explicit phase direction
        # Note: Check more specific/actionable keywords first to avoid false matches
        if guidance:
            guidance_lower = guidance.lower()
            # "adjust" should take priority over "strategy" when both present
            if "regulat" in guidance_lower or "adjust" in guidance_lower:
                self._metacognitive_cycle += 1
                return "regulating", ThoughtType.REVISION
            if "alternative" in guidance_lower or "branch" in guidance_lower:
                return "regulating", ThoughtType.BRANCH
            if "evaluat" in guidance_lower or "assess" in guidance_lower:
                return "evaluating", ThoughtType.VERIFICATION
            if "monitor" in guidance_lower or "track" in guidance_lower:
                return "monitoring", ThoughtType.VERIFICATION
            if "plan" in guidance_lower or "strategy" in guidance_lower:
                return "planning", ThoughtType.HYPOTHESIS

        # Default phase progression
        phase_progression = {
            "planning": ("monitoring", ThoughtType.VERIFICATION),
            "monitoring": ("evaluating", ThoughtType.VERIFICATION),
            "evaluating": ("regulating", ThoughtType.REVISION),
            "regulating": ("monitoring", ThoughtType.VERIFICATION),  # New cycle
        }

        next_phase, thought_type = phase_progression.get(
            prev_phase, ("monitoring", ThoughtType.CONTINUATION)
        )

        # Increment cycle when starting new monitoring after regulation
        if prev_phase == "regulating" and next_phase == "monitoring":
            self._metacognitive_cycle += 1

        return next_phase, thought_type

    def _calculate_metacognitive_metrics(
        self,
        phase: str,
        previous_thought: ThoughtNode,
    ) -> dict[str, float]:
        """Calculate metacognitive metrics for the current phase.

        Args:
            phase: Current metacognitive phase
            previous_thought: Previous thought

        Returns:
            Dictionary of metric values
        """
        # Base confidence from previous thought
        base_confidence = previous_thought.confidence

        # Phase-specific metrics
        phase_metrics = {
            "planning": {
                "confidence": 0.7,
                "strategy_awareness": 0.9,
                "progress_monitoring": 0.1,
                "self_evaluation_quality": 0.0,
            },
            "monitoring": {
                "confidence": min(base_confidence + 0.05, 0.85),
                "strategy_awareness": 0.8,
                "progress_monitoring": 0.9,
                "self_evaluation_quality": 0.3,
            },
            "evaluating": {
                "confidence": min(base_confidence + 0.1, 0.9),
                "strategy_awareness": 0.7,
                "progress_monitoring": 0.7,
                "self_evaluation_quality": 0.95,
            },
            "regulating": {
                "confidence": 0.75,  # Adjustment introduces some uncertainty
                "strategy_awareness": 0.95,
                "progress_monitoring": 0.5,
                "self_evaluation_quality": 0.8,
            },
        }

        return phase_metrics.get(
            phase,
            {
                "confidence": 0.7,
                "strategy_awareness": 0.7,
                "progress_monitoring": 0.5,
                "self_evaluation_quality": 0.5,
            },
        )
