"""Causal Reasoning method.

This module implements causal reasoning - analyzing cause-effect relationships,
tracing causal chains, identifying root causes, and predicting effects. This method
is particularly useful for debugging, diagnosis, prediction, and understanding
complex systems with interconnected causal relationships.
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
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


# Metadata for Causal Reasoning method
CAUSAL_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CAUSAL_REASONING,
    name="Causal Reasoning",
    description="Analyze cause-effect relationships, trace causal chains, identify root causes, "
    "and predict effects. Ideal for debugging, diagnosis, and prediction tasks.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "causal",
            "cause-effect",
            "root-cause",
            "diagnosis",
            "debugging",
            "prediction",
            "systems-thinking",
        }
    ),
    complexity=6,  # Medium-high complexity
    supports_branching=True,  # Can branch to explore different causal paths
    supports_revision=True,  # Can revise causal hypotheses
    requires_context=False,  # Context helpful but not required
    min_thoughts=3,  # Need at least: identify causes, trace chain, identify effects
    max_thoughts=0,  # No limit - complex causal chains can be deep
    avg_tokens_per_thought=400,  # Moderate detail for causal analysis
    best_for=(
        "debugging and troubleshooting",
        "root cause analysis",
        "system diagnosis",
        "predicting outcomes",
        "understanding complex systems",
        "failure analysis",
        "risk assessment",
        "impact analysis",
    ),
    not_recommended_for=(
        "purely creative tasks",
        "simple linear problems",
        "problems without clear causality",
        "aesthetic judgments",
    ),
)


class CausalReasoning(ReasoningMethodBase):
    """Causal Reasoning method implementation.

    This class implements causal reasoning by systematically analyzing cause-effect
    relationships. It traces causal chains from effects back to root causes and
    forward to predict consequences, making it ideal for debugging, diagnosis,
    and prediction tasks.

    Key characteristics:
    - Identifies causal factors (necessary, sufficient, contributing)
    - Maps causal relationships and dependencies
    - Traces causal chains from root causes to effects
    - Distinguishes correlation from causation
    - Supports branching to explore alternative causal paths
    - Predicts downstream effects

    The reasoning process follows these stages:
    1. Observation: Identify the effect or outcome to analyze
    2. Hypothesis: Generate potential causes
    3. Causal Tracing: Map the causal chain
    4. Root Cause Identification: Find fundamental causes
    5. Effect Prediction: Predict consequences
    6. Validation: Test causal hypotheses

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = CausalReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Why is the website loading slowly?"
        ... )
        >>> print(result.content)  # Initial causal analysis

        Continue with causal tracing:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Trace the causal chain deeper"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    def __init__(self) -> None:
        """Initialize the Causal Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._causal_chain: list[dict[str, Any]] = []
        self._root_causes: list[str] = []
        self._effects: list[str] = []
        self._use_sampling = False
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.CAUSAL_REASONING

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return CAUSAL_REASONING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return CAUSAL_REASONING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Causal Reasoning method for execution,
        resetting internal state for causal chain tracking.

        Examples:
            >>> method = CausalReasoning()
            >>> await method.initialize()
            >>> assert method._initialized is True
        """
        self._initialized = True
        self._step_counter = 0
        self._causal_chain = []
        self._root_causes = []
        self._effects = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Execute the Causal Reasoning method.

        This method creates the first thought in a causal analysis, identifying
        the effect or outcome to analyze and generating initial hypotheses about
        potential causes.

        Args:
            session: The current reasoning session
            input_text: The problem, effect, or outcome to analyze causally
            context: Optional additional context (e.g., system state, observations)
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the initial causal analysis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = CausalReasoning()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="The server crashed at 3am"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.CAUSAL_REASONING
        """
        if not self._initialized:
            raise RuntimeError("Causal Reasoning method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        # Reset state for new execution
        self._step_counter = 1
        self._causal_chain = []
        self._root_causes = []
        self._effects = []

        # Create the initial observation and hypothesis generation
        if self._use_sampling:
            content = await self._sample_initial_analysis(input_text, context)
        else:
            content = self._generate_initial_analysis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CAUSAL_REASONING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial causal hypotheses have moderate confidence
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "causal",
                "stage": "observation_and_hypothesis",
                "causal_factors": [],
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.CAUSAL_REASONING

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
        """Continue causal reasoning from a previous thought.

        This method generates the next step in causal analysis, which may involve:
        - Tracing causal chains backward to root causes
        - Tracing causal chains forward to predict effects
        - Exploring alternative causal paths (branching)
        - Validating causal hypotheses
        - Synthesizing causal understanding

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "trace root cause", "predict effects")
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the causal analysis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = CausalReasoning()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Database timeout")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Trace to root cause"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError("Causal Reasoning method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = (
            execution_context is not None
            and hasattr(execution_context, "can_sample")
            and execution_context.can_sample
        )
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine the stage of causal reasoning
        stage = self._determine_stage(previous_thought, guidance)

        # Determine thought type based on stage and guidance
        thought_type = self._determine_thought_type(stage, guidance)

        # Generate continuation content based on stage
        if self._use_sampling:
            content = await self._sample_continuation(
                previous_thought=previous_thought,
                stage=stage,
                guidance=guidance,
                context=context,
            )
        else:
            content = self._generate_continuation(
                previous_thought=previous_thought,
                stage=stage,
                guidance=guidance,
                context=context,
            )

        # Calculate confidence (may increase as we trace to root causes)
        confidence = self._calculate_confidence(stage, previous_thought)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CAUSAL_REASONING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "previous_step": previous_thought.step_number,
                "stage": stage,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "causal",
                "causal_chain_depth": len(self._causal_chain),
                "root_causes_identified": len(self._root_causes),
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Causal Reasoning, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = CausalReasoning()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial causal analysis content.

        This creates the first step: observing the effect and generating
        initial causal hypotheses.

        Args:
            input_text: The effect or outcome to analyze
            context: Optional additional context

        Returns:
            The content for the initial thought

        Note:
            In a full implementation, this would use an LLM to generate
            actual causal analysis. This is a structured template.
        """
        context_info = ""
        if context:
            context_info = f"\n\nContext: {context}"

        return f"""Step {self._step_counter}: Initial Causal Analysis - Observation & Hypothesis

Effect/Outcome to Analyze: {input_text}{context_info}

Causal Analysis Framework:
1. OBSERVATION: What is the effect/outcome we're analyzing?
   - Clearly define the phenomenon that needs explanation
   - Note when/where it occurs
   - Identify any patterns or anomalies

2. INITIAL CAUSAL HYPOTHESES:
   - What are the potential immediate causes?
   - What factors could contribute to this effect?
   - What are the necessary vs. sufficient conditions?

3. CAUSAL RELATIONSHIP TYPES:
   - Direct causes: Factors that directly produce the effect
   - Contributing causes: Factors that increase likelihood
   - Enabling conditions: Factors that allow the effect to occur
   - Preventing factors: What would prevent this effect?

Next steps: We'll trace the causal chain to identify root causes and map
the full causal network."""

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        stage: str,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate continuation thought content based on causal stage.

        Args:
            previous_thought: The thought to build upon
            stage: The current stage of causal analysis
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The content for the continuation thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        stage_templates = {
            "causal_tracing": self._generate_causal_tracing_content,
            "root_cause_identification": self._generate_root_cause_content,
            "effect_prediction": self._generate_effect_prediction_content,
            "validation": self._generate_validation_content,
            "synthesis": self._generate_synthesis_content,
            "alternative_path": self._generate_alternative_path_content,
        }

        generator = stage_templates.get(stage, self._generate_generic_continuation_content)
        return generator(previous_thought, guidance_text, context)

    def _generate_causal_tracing_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for causal chain tracing stage."""
        return f"""Step {self._step_counter}: Causal Chain Tracing

Building on step {previous_thought.step_number}, I'll trace the causal chain
backward from the effect to deeper causes.

CAUSAL CHAIN ANALYSIS:
1. Immediate Causes (Direct predecessors)
   - What directly caused the observed effect?
   - What is the temporal sequence?

2. Intermediate Causes (Chain links)
   - What caused the immediate causes?
   - Are there feedback loops or circular causation?

3. Causal Dependencies
   - Which causes are necessary (effect can't occur without them)?
   - Which are sufficient (alone can produce the effect)?
   - Which are contributing (increase probability)?

4. Confounding Factors
   - What factors might create spurious correlations?
   - Are we confusing correlation with causation?{guidance_text}"""

    def _generate_root_cause_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for root cause identification stage."""
        return f"""Step {self._step_counter}: Root Cause Identification

Following the causal chain from step {previous_thought.step_number},
I'll identify the fundamental root causes.

ROOT CAUSE ANALYSIS:
1. Fundamental Causes (Can't be traced further back)
   - What are the ultimate sources in this causal chain?
   - What would we need to change to prevent recurrence?

2. Systemic vs. Proximate Causes
   - Proximate: Immediate triggers
   - Systemic: Underlying structural or environmental factors

3. Human vs. Technical vs. Process Causes
   - Where in the system did causation originate?
   - What category of intervention would address it?

4. Root Cause Verification
   - "Five Whys" test: Can we trace this cause further back?
   - Would removing this cause prevent the effect?{guidance_text}"""

    def _generate_effect_prediction_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for effect prediction stage."""
        return f"""Step {self._step_counter}: Effect Prediction

Based on the causal understanding from step {previous_thought.step_number},
I'll predict downstream effects and consequences.

EFFECT PREDICTION:
1. Direct Effects (Immediate consequences)
   - What will happen as a direct result?
   - What is the expected timeline?

2. Cascading Effects (Secondary and tertiary impacts)
   - What will the direct effects cause?
   - How will effects propagate through the system?

3. Side Effects and Unintended Consequences
   - What unexpected outcomes might occur?
   - What trade-offs or risks should be considered?

4. Probability and Magnitude
   - How likely is each predicted effect?
   - What is the potential impact severity?{guidance_text}"""

    def _generate_validation_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for causal validation stage."""
        return f"""Step {self._step_counter}: Causal Validation

Validating the causal hypotheses developed in step {previous_thought.step_number}.

VALIDATION TESTS:
1. Temporal Precedence
   - Does the cause precede the effect in time?
   - Is the timing consistent with the causal claim?

2. Covariation
   - When the cause is present, is the effect more likely?
   - When the cause is absent, is the effect less likely?

3. Alternative Explanations
   - Have we ruled out other plausible causes?
   - Could the relationship be coincidental?

4. Mechanism
   - Can we explain HOW the cause produces the effect?
   - Is there a plausible causal mechanism?

5. Evidence Quality
   - What evidence supports this causal claim?
   - How confident can we be in this causal relationship?{guidance_text}"""

    def _generate_synthesis_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for causal synthesis stage."""
        return f"""Step {self._step_counter}: Causal Synthesis

Synthesizing the complete causal understanding from our analysis.

CAUSAL MAP SYNTHESIS:
1. Complete Causal Chain
   - Root Causes → Intermediate Causes → Proximate Causes → Effect
   - Key branch points and alternative paths
   - Feedback loops and circular causation

2. Confidence Assessment
   - Which causal links are well-established?
   - Which are hypothetical or uncertain?
   - What additional evidence would strengthen our understanding?

3. Actionable Insights
   - Where can we intervene most effectively?
   - What are the leverage points in the causal system?
   - What are the risks and benefits of intervention?

4. Knowledge Gaps
   - What aspects of causation remain unclear?
   - What further investigation is needed?{guidance_text}"""

    def _generate_alternative_path_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for exploring alternative causal paths (branching)."""
        return f"""Step {self._step_counter}: Alternative Causal Path Exploration

Branching from step {previous_thought.step_number} to explore an alternative
causal pathway or hypothesis.

ALTERNATIVE PATH ANALYSIS:
1. Alternative Hypothesis
   - What is the alternative causal explanation?
   - How does it differ from our primary hypothesis?

2. Supporting Evidence
   - What evidence supports this alternative?
   - Why should we consider this path?

3. Comparative Analysis
   - How does this path compare to our main causal chain?
   - Which explanation better fits the evidence?

4. Integration Potential
   - Could both causal paths be operating simultaneously?
   - Are these competing or complementary explanations?{guidance_text}"""

    def _generate_generic_continuation_content(
        self,
        previous_thought: ThoughtNode,
        guidance_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate generic continuation content."""
        return f"""Step {self._step_counter}: Continuing Causal Analysis

Continuing from step {previous_thought.step_number}, deepening our
causal understanding.{guidance_text}"""

    def _determine_stage(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> str:
        """Determine the current stage of causal reasoning.

        Args:
            previous_thought: The previous thought in the chain
            guidance: Optional guidance that may indicate desired stage

        Returns:
            The stage identifier
        """
        # Check guidance for explicit stage direction
        if guidance:
            guidance_lower = guidance.lower()
            if "root cause" in guidance_lower or "fundamental" in guidance_lower:
                return "root_cause_identification"
            if "predict" in guidance_lower or "effect" in guidance_lower:
                return "effect_prediction"
            if "validate" in guidance_lower or "verify" in guidance_lower:
                return "validation"
            if "synthesize" in guidance_lower or "summary" in guidance_lower:
                return "synthesis"
            if "alternative" in guidance_lower or "branch" in guidance_lower:
                return "alternative_path"

        # Default progression based on previous stage
        prev_stage = previous_thought.metadata.get("stage", "observation_and_hypothesis")

        stage_progression = {
            "observation_and_hypothesis": "causal_tracing",
            "causal_tracing": "root_cause_identification",
            "root_cause_identification": "effect_prediction",
            "effect_prediction": "validation",
            "validation": "synthesis",
            "alternative_path": "synthesis",
        }

        return stage_progression.get(prev_stage, "causal_tracing")

    def _determine_thought_type(self, stage: str, guidance: str | None) -> ThoughtType:
        """Determine the appropriate thought type for this stage.

        Args:
            stage: The current causal reasoning stage
            guidance: Optional guidance

        Returns:
            The appropriate ThoughtType
        """
        # Check for explicit branching in guidance
        if guidance and ("alternative" in guidance.lower() or "branch" in guidance.lower()):
            return ThoughtType.BRANCH

        # Map stages to thought types
        stage_to_type = {
            "observation_and_hypothesis": ThoughtType.HYPOTHESIS,
            "causal_tracing": ThoughtType.CONTINUATION,
            "root_cause_identification": ThoughtType.CONTINUATION,
            "effect_prediction": ThoughtType.HYPOTHESIS,
            "validation": ThoughtType.VERIFICATION,
            "synthesis": ThoughtType.SYNTHESIS,
            "alternative_path": ThoughtType.BRANCH,
        }

        return stage_to_type.get(stage, ThoughtType.CONTINUATION)

    def _calculate_confidence(
        self,
        stage: str,
        previous_thought: ThoughtNode,
    ) -> float:
        """Calculate confidence for the current thought.

        Confidence typically increases as we:
        - Trace to root causes
        - Validate causal hypotheses
        - Synthesize understanding

        Args:
            stage: The current stage
            previous_thought: The previous thought

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = previous_thought.confidence

        # Confidence adjustments by stage
        stage_adjustments = {
            "observation_and_hypothesis": 0.0,
            "causal_tracing": 0.05,
            "root_cause_identification": 0.1,
            "effect_prediction": -0.05,  # Predictions are less certain
            "validation": 0.15,  # Validation increases confidence
            "synthesis": 0.1,
            "alternative_path": -0.1,  # Alternatives introduce uncertainty
        }

        adjustment = stage_adjustments.get(stage, 0.0)
        new_confidence = base_confidence + adjustment

        # Keep confidence in valid range
        return max(0.3, min(0.95, new_confidence))

    async def _sample_initial_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial causal analysis using LLM sampling.

        Uses the execution context's sampling capability to generate
        actual causal analysis rather than placeholder content.

        Args:
            input_text: The effect or outcome to analyze
            context: Optional additional context

        Returns:
            The content for the initial thought
        """
        context_info = ""
        if context:
            context_info = f"\n\nAdditional Context: {context}"

        system_prompt = """You are a causal reasoning expert using systematic cause-effect analysis.
Analyze the given effect/outcome to identify potential causes and causal relationships.

Structure your analysis with:
1. OBSERVATION: Clearly define the effect/outcome to analyze
2. INITIAL CAUSAL HYPOTHESES: Identify potential immediate causes
3. CAUSAL RELATIONSHIP TYPES: Classify causes as direct, contributing, or enabling
4. Next steps: Outline how to trace the causal chain

Be systematic and explicit about causal relationships. Distinguish between
correlation and causation."""

        user_prompt = f"""Effect/Outcome to Analyze: {input_text}{context_info}

Generate an initial causal analysis following the framework:
- Observation of the effect
- Initial causal hypotheses
- Types of causal relationships to explore
- Next steps for deeper analysis"""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_initial_analysis(input_text, context),
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
        causal analysis from a previous thought.

        Args:
            previous_thought: The thought to build upon
            stage: The current stage of causal analysis
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            The content for the continuation thought
        """
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""
        context_text = f"\nAdditional Context: {context}" if context else ""

        stage_descriptions = {
            "causal_tracing": (
                "Trace the causal chain backward from the effect to deeper causes. "
                "Identify immediate causes, intermediate causes, and causal dependencies."
            ),
            "root_cause_identification": (
                "Identify fundamental root causes that can't be traced further back. "
                "Distinguish between proximate and systemic causes."
            ),
            "effect_prediction": (
                "Predict downstream effects and consequences. Include direct effects, "
                "cascading effects, and potential unintended consequences."
            ),
            "validation": (
                "Validate the causal hypotheses. Check temporal precedence, covariation, "
                "alternative explanations, and causal mechanisms."
            ),
            "synthesis": (
                "Synthesize the complete causal understanding. Map the full causal chain "
                "from root causes to effects and identify actionable insights."
            ),
            "alternative_path": (
                "Explore an alternative causal pathway or hypothesis. Compare it with "
                "the primary hypothesis and assess integration potential."
            ),
        }

        stage_instruction = stage_descriptions.get(
            stage,
            (
                "Continue the causal analysis, deepening understanding of "
                "cause-effect relationships."
            ),
        )

        system_prompt = f"""You are a causal reasoning expert continuing a systematic analysis.
Current Stage: {stage.replace("_", " ").title()}

{stage_instruction}

Build upon the previous analysis with clear logical connections and explicit causal reasoning."""

        user_prompt = f"""Previous Analysis (Step {previous_thought.step_number}):
{previous_thought.content}

Current Stage: {stage.replace("_", " ").title()}
{guidance_text}{context_text}

Continue the causal analysis for this stage, building on the previous step."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_continuation(
                previous_thought, stage, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )
