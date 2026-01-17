"""Counterfactual Reasoning method.

This module implements counterfactual reasoning - a "what-if" analysis method that
explores alternative scenarios by systematically altering key variables and analyzing
how outcomes differ. This approach is valuable for decision analysis, regret minimization,
understanding causality, and learning from hypothetical situations.
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
    from reasoning_mcp.models import Session


# Metadata for Counterfactual Reasoning method
COUNTERFACTUAL_METADATA = MethodMetadata(
    identifier=MethodIdentifier.COUNTERFACTUAL,
    name="Counterfactual Reasoning",
    description="What-if analysis through alternative scenario exploration. "
    "Establishes baseline, identifies key variables, generates counterfactual scenarios, "
    "and analyzes outcome differences to gain insights.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "counterfactual",
            "what-if",
            "scenarios",
            "alternatives",
            "decision-analysis",
            "causality",
            "comparison",
            "hypothetical",
        }
    ),
    complexity=4,  # Medium complexity - systematic scenario analysis
    supports_branching=True,  # Can branch for multiple counterfactual scenarios
    supports_revision=True,  # Can revise scenarios and analyses
    requires_context=False,  # No special context needed, but can use it
    min_thoughts=5,  # Baseline, variables, scenarios, analysis, synthesis
    max_thoughts=0,  # Unlimited - can explore many scenarios
    avg_tokens_per_thought=400,  # Moderate detail per scenario
    best_for=(
        "decision analysis and evaluation",
        "regret minimization and prevention",
        "causal understanding and learning",
        "risk assessment and mitigation",
        "strategy comparison and selection",
        "policy impact analysis",
        "historical analysis and hindsight",
        "understanding opportunity costs",
    ),
    not_recommended_for=(
        "simple yes/no questions",
        "problems with no viable alternatives",
        "purely factual queries without decisions",
        "creative brainstorming (use lateral thinking)",
        "real-time constraint optimization",
    ),
)


class Counterfactual(ReasoningMethodBase):
    """Counterfactual Reasoning method implementation.

    This class implements counterfactual reasoning - a systematic approach to
    exploring alternative scenarios by identifying key variables, altering them,
    and analyzing how outcomes would differ. This method is particularly valuable
    for understanding causality, evaluating decisions, and learning from both
    actual and hypothetical situations.

    The reasoning process follows these stages:
    1. Baseline establishment - Document the actual/current scenario
    2. Variable identification - Identify key factors that could be different
    3. Scenario generation - Create alternative scenarios by altering variables
    4. Outcome analysis - Analyze how each scenario differs from baseline
    5. Insight synthesis - Draw conclusions about causality and decisions

    Key characteristics:
    - Systematic what-if analysis
    - Baseline comparison focus
    - Multiple scenario exploration
    - Causal insight generation
    - Decision support oriented

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = Counterfactual()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Should I have accepted the job offer in another city?"
        ... )
        >>> print(result.content)  # Establishes baseline scenario

        Continue with variable identification:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Identify key variables"
        ... )
        >>> print(next_thought.content)  # Lists factors that could differ

        Branch to explore alternative scenario:
        >>> scenario = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=next_thought,
        ...     guidance="Explore scenario: accepted the job offer",
        ...     context={"branch": True}
        ... )
        >>> print(scenario.type)  # ThoughtType.BRANCH
    """

    def __init__(self) -> None:
        """Initialize the Counterfactual Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_stage = "baseline"  # baseline, variables, scenarios, analysis, synthesis
        self._baseline_established = False
        self._variables_identified = False
        self._scenarios: list[str] = []
        self._use_sampling: bool = True
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.COUNTERFACTUAL

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return COUNTERFACTUAL_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return COUNTERFACTUAL_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Counterfactual Reasoning method for execution.
        It resets the internal state and prepares for a new reasoning session.

        Examples:
            >>> method = Counterfactual()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._current_stage == "baseline"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_stage = "baseline"
        self._baseline_established = False
        self._variables_identified = False
        self._scenarios = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Execute the Counterfactual Reasoning method.

        This method creates the first thought in a counterfactual reasoning process.
        It establishes the baseline scenario - the actual or current state that
        alternative scenarios will be compared against.

        Args:
            session: The current reasoning session
            input_text: The decision or situation to analyze counterfactually
            context: Optional context including baseline information
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the baseline scenario establishment

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Counterfactual()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="What if I had invested in stocks instead of bonds?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert "baseline" in thought.metadata["stage"]
        """
        if not self._initialized:
            raise RuntimeError(
                "Counterfactual Reasoning method must be initialized before execution"
            )

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_stage = "baseline"
        self._baseline_established = True
        self._variables_identified = False
        self._scenarios = []

        # Generate baseline scenario content
        if self._use_sampling:
            content = await self._sample_baseline(input_text, context)
        else:
            content = self._generate_baseline(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COUNTERFACTUAL,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.8,  # High confidence in baseline facts
            metadata={
                "input": input_text,
                "context": context or {},
                "stage": "baseline",
                "reasoning_type": "counterfactual",
                "baseline_established": True,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.COUNTERFACTUAL

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Continue counterfactual reasoning from a previous thought.

        This method advances through the counterfactual reasoning stages:
        1. After baseline: identify key variables that could be different
        2. After variables: generate counterfactual scenarios (can branch)
        3. After scenarios: analyze outcome differences
        4. After analysis: synthesize insights and conclusions

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "identify variables", "explore scenario X")
            context: Optional context (e.g., {"branch": True} for scenario exploration)
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the counterfactual reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Counterfactual()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Investment decision")
            >>> # Identify variables
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Identify key variables"
            ... )
            >>> assert second.metadata["stage"] == "variables"
            >>> # Create counterfactual scenario
            >>> scenario = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=second,
            ...     guidance="What if interest rates were 2% higher?",
            ...     context={"branch": True}
            ... )
            >>> assert scenario.type == ThoughtType.BRANCH
        """
        if not self._initialized:
            raise RuntimeError(
                "Counterfactual Reasoning method must be initialized before continuation"
            )

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next stage and thought type
        is_branch = context and context.get("branch", False)
        is_synthesis = guidance and "synthesize" in guidance.lower()
        is_conclusion = guidance and "conclude" in guidance.lower()

        # Determine stage progression
        current_stage = previous_thought.metadata.get("stage", "baseline")

        if is_conclusion:
            next_stage = "conclusion"
            thought_type = ThoughtType.CONCLUSION
        elif is_synthesis:
            next_stage = "synthesis"
            thought_type = ThoughtType.SYNTHESIS
        elif is_branch or current_stage == "variables":
            next_stage = "scenarios"
            thought_type = ThoughtType.BRANCH if is_branch else ThoughtType.CONTINUATION
            if is_branch:
                scenario_name = guidance or f"Scenario {len(self._scenarios) + 1}"
                self._scenarios.append(scenario_name)
        elif current_stage == "baseline":
            next_stage = "variables"
            thought_type = ThoughtType.CONTINUATION
            self._variables_identified = True
        elif current_stage == "scenarios":
            next_stage = "analysis"
            thought_type = ThoughtType.VERIFICATION
        else:
            next_stage = "synthesis"
            thought_type = ThoughtType.SYNTHESIS

        self._current_stage = next_stage

        # Generate content based on stage
        if self._use_sampling:
            content = await self._sample_continuation(
                previous_thought=previous_thought,
                stage=next_stage,
                guidance=guidance,
                context=context,
            )
        else:
            content = self._generate_continuation(
                previous_thought=previous_thought,
                stage=next_stage,
                guidance=guidance,
                context=context,
            )

        # Calculate confidence
        confidence = self._calculate_confidence(
            stage=next_stage,
            depth=previous_thought.depth + 1,
            num_scenarios=len(self._scenarios),
        )

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.COUNTERFACTUAL,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "stage": next_stage,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "counterfactual",
                "scenarios_explored": len(self._scenarios),
                "is_branch": is_branch,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Counterfactual Reasoning, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Counterfactual()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_baseline(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the baseline scenario content.

        This establishes the actual or current state that counterfactual
        scenarios will be compared against.

        Args:
            input_text: The decision or situation to analyze
            context: Optional context with baseline information

        Returns:
            The content for the baseline thought
        """
        baseline_info = ""
        if context and "baseline" in context:
            baseline_info = f"\n\nBaseline context: {context['baseline']}"

        return (
            f"Step {self._step_counter}: Establishing Baseline Scenario\n\n"
            f"Question/Decision: {input_text}\n\n"
            f"To perform counterfactual reasoning, I first need to establish the baseline - "
            f"the actual scenario or decision that occurred. This baseline serves as our "
            f"reference point for all alternative 'what-if' scenarios.\n\n"
            f"Baseline Analysis:\n"
            f"- What actually happened or the current state\n"
            f"- Key outcomes observed\n"
            f"- Current satisfaction/regret level\n"
            f"- Context and circumstances{baseline_info}\n\n"
            f"Next: I'll identify the key variables that could have been different."
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        stage: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation content based on the current stage.

        Args:
            previous_thought: The thought to build upon
            stage: Current stage of counterfactual reasoning
            guidance: Optional guidance for this step
            context: Optional additional context

        Returns:
            The content for the continuation thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        if stage == "variables":
            return (
                f"Step {self._step_counter}: Identifying Key Variables\n\n"
                f"Building on the baseline scenario, I'll now identify the key variables - "
                f"factors that could have been different and might have changed the outcome.\n\n"
                f"Variable Categories to Consider:\n"
                f"1. Decisions made (choices at key junctures)\n"
                f"2. Timing (when actions were taken)\n"
                f"3. External factors (market conditions, other actors)\n"
                f"4. Available information (what was known/unknown)\n"
                f"5. Resources (time, money, skills)\n"
                f"6. Environmental conditions\n\n"
                f"For each variable, I'll assess:\n"
                f"- How likely it could have been different\n"
                f"- Potential magnitude of impact\n"
                f"- Controllability (was it within our control?){guidance_text}\n\n"
                f"Next: Generate alternative scenarios by altering these variables."
            )

        elif stage == "scenarios":
            is_branch = context and context.get("branch", False)
            if is_branch:
                scenario_num = len(self._scenarios)
                return (
                    f"Step {self._step_counter}: Counterfactual Scenario {scenario_num}\n\n"
                    f"Alternative Scenario: {guidance or 'What-if analysis'}\n\n"
                    f"In this counterfactual scenario, I'll explore what would have happened "
                    f"if key variables were different from the baseline.\n\n"
                    f"Scenario Details:\n"
                    f"- What changes from baseline\n"
                    f"- Initial conditions and context\n"
                    f"- Likely chain of events\n"
                    f"- Expected outcomes\n"
                    f"- Key differences from baseline{guidance_text}\n\n"
                    f"Comparison to Baseline:\n"
                    f"- Better/worse outcomes\n"
                    f"- Trade-offs introduced\n"
                    f"- Risks avoided or introduced\n"
                    f"- Overall assessment"
                )
            else:
                return (
                    f"Step {self._step_counter}: Generating Counterfactual Scenarios\n\n"
                    f"Based on the identified variables, I'll now generate alternative scenarios "
                    f"to explore what could have happened under different conditions.\n\n"
                    f"Each scenario will alter one or more key variables while keeping others constant, "
                    f"allowing us to isolate the impact of specific factors.{guidance_text}"
                )

        elif stage == "analysis":
            return (
                f"Step {self._step_counter}: Analyzing Outcome Differences\n\n"
                f"Having explored {len(self._scenarios)} counterfactual scenario(s), "
                f"I'll now analyze how outcomes differ across scenarios.\n\n"
                f"Comparative Analysis:\n"
                f"1. Outcome comparison across all scenarios\n"
                f"2. Identification of causal relationships\n"
                f"3. Best and worst case scenarios\n"
                f"4. Robustness of baseline decision\n"
                f"5. Critical variables that drive outcomes{guidance_text}\n\n"
                f"This analysis reveals which factors were most influential and whether "
                f"alternative decisions would have led to better outcomes."
            )

        elif stage == "synthesis":
            return (
                f"Step {self._step_counter}: Synthesizing Insights\n\n"
                f"Drawing together insights from {len(self._scenarios)} scenario(s), "
                f"I'll synthesize key learnings from this counterfactual analysis.\n\n"
                f"Key Insights:\n"
                f"1. Causal understanding (what drives outcomes)\n"
                f"2. Decision quality assessment (was baseline optimal?)\n"
                f"3. Regret analysis (justified or unjustified)\n"
                f"4. Lessons learned for future decisions\n"
                f"5. Robust strategies across scenarios{guidance_text}\n\n"
                f"Recommendations:\n"
                f"- What to do differently next time\n"
                f"- Which factors to monitor\n"
                f"- How to reduce regret and improve decisions"
            )

        elif stage == "conclusion":
            return (
                f"Step {self._step_counter}: Conclusion\n\n"
                f"Final Assessment of Counterfactual Analysis:\n\n"
                f"After exploring alternative scenarios and comparing outcomes, "
                f"I can now provide a comprehensive answer to the original question.\n\n"
                f"Summary:\n"
                f"- Baseline scenario evaluation\n"
                f"- Best alternative identified\n"
                f"- Key factors that matter most\n"
                f"- Overall decision quality assessment{guidance_text}\n\n"
                f"This counterfactual reasoning process has revealed the causal structure "
                f"of the situation and provided actionable insights for future decisions."
            )

        else:
            return (
                f"Step {self._step_counter}: Continuing counterfactual analysis\n\n"
                f"Building on the previous step, continuing to explore alternative "
                f"scenarios and their implications.{guidance_text}"
            )

    def _calculate_confidence(
        self,
        stage: str,
        depth: int,
        num_scenarios: int,
    ) -> float:
        """Calculate confidence for a thought based on stage and context.

        Args:
            stage: Current reasoning stage
            depth: Depth in the thought tree
            num_scenarios: Number of scenarios explored

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence by stage
        stage_confidence = {
            "baseline": 0.8,  # High confidence in factual baseline
            "variables": 0.7,  # Good confidence in variable identification
            "scenarios": 0.6,  # Moderate - counterfactuals are hypothetical
            "analysis": 0.65,  # Improves with more scenarios analyzed
            "synthesis": 0.7,  # Higher with comprehensive analysis
            "conclusion": 0.75,  # High for well-supported conclusions
        }

        base = stage_confidence.get(stage, 0.6)

        # Adjust for depth (slight decrease with deeper reasoning)
        depth_penalty = min(0.15, depth * 0.02)

        # Adjust for number of scenarios (more scenarios = better analysis)
        scenario_bonus = (
            min(0.1, num_scenarios * 0.02)
            if stage in ("analysis", "synthesis", "conclusion")
            else 0
        )

        confidence = base - depth_penalty + scenario_bonus

        # Clamp to valid range
        return max(0.3, min(0.95, confidence))

    async def _sample_baseline(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate baseline scenario using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual counterfactual baseline analysis rather than placeholder content.

        Args:
            input_text: The decision or situation to analyze
            context: Optional context with baseline information

        Returns:
            The content for the baseline thought
        """
        baseline_info = ""
        if context and "baseline" in context:
            baseline_info = f"\n\nBaseline context: {context['baseline']}"

        system_prompt = """You are a counterfactual reasoning expert analyzing alternative scenarios.
Your task is to establish the baseline scenario - the actual situation or decision that occurred.
This baseline will serve as the reference point for all alternative 'what-if' scenarios.

Structure your analysis with:
1. Clear statement of what actually happened or the current state
2. Key outcomes that were observed
3. Current satisfaction/regret level
4. Context and circumstances that led to this baseline
5. Brief indication of what will be explored next

Be factual and thorough about the baseline scenario."""

        user_prompt = f"""Question/Decision: {input_text}{baseline_info}

Establish the baseline scenario for this counterfactual analysis. Describe:
- What actually happened or the current state
- Key outcomes observed
- Context and circumstances
- Why this baseline matters for counterfactual exploration"""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_baseline(input_text, context),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

    async def _sample_continuation(
        self,
        previous_thought: ThoughtNode,
        stage: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation using LLM sampling.

        Uses the execution context's sampling capability to continue
        counterfactual analysis from a previous thought.

        Args:
            previous_thought: The thought to build upon
            stage: Current stage of counterfactual reasoning
            guidance: Optional guidance for this step
            context: Optional additional context

        Returns:
            The content for the continuation thought
        """
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""
        context_text = f"\nAdditional Context: {context}" if context else ""

        stage_descriptions = {
            "variables": (
                "Identify key variables that could have been different from the baseline. "
                "Consider decisions made, timing, external factors, available information, "
                "resources, and environmental conditions. Assess likelihood, impact, and controllability."
            ),
            "scenarios": (
                "Generate counterfactual scenarios by altering key variables. "
                "Describe what changes from baseline, initial conditions, likely chain of events, "
                "expected outcomes, and how they compare to the baseline."
            ),
            "analysis": (
                "Analyze outcome differences across all counterfactual scenarios. "
                "Compare outcomes, identify causal relationships, determine best/worst cases, "
                "assess robustness of baseline decision, and identify critical variables."
            ),
            "synthesis": (
                "Synthesize insights from all scenarios explored. "
                "Develop causal understanding, assess decision quality, analyze regret, "
                "extract lessons learned, and identify robust strategies."
            ),
            "conclusion": (
                "Provide final assessment of the counterfactual analysis. "
                "Evaluate baseline scenario, identify best alternative, determine key factors, "
                "and assess overall decision quality with actionable insights."
            ),
        }

        stage_instruction = stage_descriptions.get(
            stage,
            "Continue the counterfactual analysis, exploring alternative scenarios and their implications.",
        )

        system_prompt = f"""You are a counterfactual reasoning expert continuing a systematic analysis.
Current Stage: {stage.replace("_", " ").title()}

{stage_instruction}

Build upon the previous analysis with clear logical connections and explicit counterfactual reasoning.
Compare scenarios to the baseline systematically."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Current Stage: {stage.replace("_", " ").title()}
{guidance_text}{context_text}

Continue the counterfactual analysis for this stage, building on the previous step."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_continuation(
                previous_thought, stage, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )
