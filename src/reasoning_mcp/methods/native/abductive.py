"""Abductive reasoning method.

This module implements abductive reasoning - inference to the best explanation.
Given observations or evidence, abductive reasoning generates candidate hypotheses
and evaluates them to find the most likely explanation. This method is ideal for
diagnosis, investigation, and mystery-solving scenarios.
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


# Metadata for Abductive reasoning method
ABDUCTIVE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ABDUCTIVE,
    name="Abductive Reasoning",
    description="Inference to the best explanation from observations. "
    "Generates and evaluates candidate hypotheses to find the most likely "
    "explanation for observed phenomena.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({
        "abductive",
        "hypothesis",
        "inference",
        "diagnosis",
        "investigation",
        "explanation",
        "evidence-based",
    }),
    complexity=6,  # Medium-high complexity
    supports_branching=True,  # Branch for multiple hypotheses
    supports_revision=True,  # Can revise hypotheses based on new evidence
    requires_context=False,  # Can work with just observations
    min_thoughts=3,  # At least: observations, hypotheses, evaluation
    max_thoughts=0,  # No limit - depends on number of hypotheses
    avg_tokens_per_thought=400,  # Moderate detail per thought
    best_for=(
        "diagnostic problems",
        "investigation and mystery-solving",
        "root cause analysis",
        "medical diagnosis",
        "troubleshooting and debugging",
        "scientific hypothesis generation",
        "explaining unexpected observations",
        "forensic analysis",
    ),
    not_recommended_for=(
        "deductive proofs",
        "purely mathematical problems",
        "problems with known solutions",
        "simple linear tasks",
        "creative brainstorming without evidence",
    ),
)


class Abductive:
    """Abductive reasoning method implementation.

    This class implements abductive reasoning - the process of inferring the best
    explanation for a set of observations. Unlike deductive reasoning (which proves
    conclusions from premises) or inductive reasoning (which generalizes from examples),
    abductive reasoning seeks the most plausible explanation.

    The reasoning process follows these stages:
    1. Observation Collection: Gather and organize evidence/observations
    2. Hypothesis Generation: Create candidate explanations
    3. Hypothesis Evaluation: Assess each explanation against evidence
    4. Best Explanation Selection: Choose the most plausible hypothesis

    Key characteristics:
    - Evidence-driven reasoning
    - Multiple hypothesis generation and comparison
    - Probabilistic evaluation (not guaranteed truth)
    - Supports branching for exploring different hypotheses
    - Iterative refinement as new evidence emerges

    Examples:
        Initialize and execute for diagnosis:
        >>> from reasoning_mcp.models import Session
        >>> method = Abductive()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Patient has fever, headache, and stiff neck"
        ... )
        >>> print(result.content)  # Observation collection

        Continue with hypothesis generation:
        >>> hypotheses = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Generate candidate diagnoses"
        ... )
        >>> print(hypotheses.type)  # HYPOTHESIS
    """

    def __init__(self) -> None:
        """Initialize the Abductive reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._stage = "observations"  # Current stage in abductive process
        self._hypotheses: list[dict[str, Any]] = []  # Track generated hypotheses
        self._observations: list[str] = []  # Track collected observations

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.ABDUCTIVE

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return ABDUCTIVE_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return ABDUCTIVE_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Abductive reasoning method for execution,
        resetting internal state and counters.

        Examples:
            >>> method = Abductive()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._stage == "observations"
        """
        self._initialized = True
        self._step_counter = 0
        self._stage = "observations"
        self._hypotheses = []
        self._observations = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the Abductive reasoning method.

        This method initiates abductive reasoning by collecting initial observations
        and evidence. This is the first stage in the process of finding the best
        explanation.

        Args:
            session: The current reasoning session
            input_text: The observations or problem to explain
            context: Optional additional context with evidence or constraints

        Returns:
            A ThoughtNode representing the initial observation collection

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Abductive()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Server crashes every night at 2 AM"
            ... )
            >>> assert thought.type == ThoughtType.OBSERVATION
            >>> assert thought.step_number == 1
            >>> assert "observations" in thought.metadata["stage"]
        """
        if not self._initialized:
            raise RuntimeError(
                "Abductive reasoning method must be initialized before execution"
            )

        # Reset for new execution
        self._step_counter = 1
        self._stage = "observations"
        self._hypotheses = []
        self._observations = []

        # Parse and organize observations
        observations = self._parse_observations(input_text, context)
        self._observations = observations

        # Create the initial observation thought
        content = self._generate_observation_collection(observations, input_text)

        thought = ThoughtNode(
            type=ThoughtType.OBSERVATION,
            method_id=MethodIdentifier.ABDUCTIVE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.8,  # High confidence in observations
            metadata={
                "input": input_text,
                "context": context or {},
                "stage": "observations",
                "observations": observations,
                "reasoning_type": "abductive",
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.ABDUCTIVE

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue abductive reasoning from a previous thought.

        This method advances through the stages of abductive reasoning:
        observations → hypothesis generation → evaluation → conclusion.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context or new evidence

        Returns:
            A new ThoughtNode continuing the abductive reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Abductive()
            >>> await method.initialize()
            >>> obs = await method.execute(session, "Observations here")
            >>> hyp = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=obs,
            ...     guidance="Generate hypotheses"
            ... )
            >>> assert hyp.type == ThoughtType.HYPOTHESIS
            >>> assert hyp.step_number == 2
        """
        if not self._initialized:
            raise RuntimeError(
                "Abductive reasoning method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Determine next stage and thought type based on current state
        stage_info = self._determine_next_stage(previous_thought, guidance)
        self._stage = stage_info["stage"]

        # Generate appropriate content based on stage
        content = self._generate_stage_content(
            stage=self._stage,
            previous_thought=previous_thought,
            guidance=guidance,
            context=context,
        )

        # Create the thought with appropriate type
        thought = ThoughtNode(
            type=stage_info["thought_type"],
            method_id=MethodIdentifier.ABDUCTIVE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=stage_info["confidence"],
            metadata={
                "stage": self._stage,
                "previous_stage": previous_thought.metadata.get("stage", "unknown"),
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "abductive",
                "hypotheses_count": len(self._hypotheses),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Abductive()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _parse_observations(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[str]:
        """Parse and extract observations from input.

        Args:
            input_text: The input text containing observations
            context: Optional context with additional observations

        Returns:
            A list of parsed observations
        """
        observations = []

        # Add main observation
        observations.append(input_text.strip())

        # Extract additional observations from context
        if context:
            if "observations" in context:
                obs_list = context["observations"]
                if isinstance(obs_list, list):
                    observations.extend(obs_list)
                elif isinstance(obs_list, str):
                    observations.append(obs_list)

            if "evidence" in context:
                evidence = context["evidence"]
                if isinstance(evidence, list):
                    observations.extend(evidence)
                elif isinstance(evidence, str):
                    observations.append(evidence)

        return observations

    def _generate_observation_collection(
        self,
        observations: list[str],
        input_text: str,
    ) -> str:
        """Generate content for observation collection stage.

        Args:
            observations: List of observations to organize
            input_text: Original input text

        Returns:
            Formatted content for the observation thought
        """
        obs_list = "\n".join(f"  - {obs}" for obs in observations)

        return (
            f"Step {self._step_counter}: Observation Collection\n\n"
            f"I need to find the best explanation for the following observations:\n\n"
            f"{obs_list}\n\n"
            f"Let me organize these observations systematically to identify patterns "
            f"and key evidence that will help generate plausible hypotheses."
        )

    def _determine_next_stage(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> dict[str, Any]:
        """Determine the next stage in abductive reasoning.

        Args:
            previous_thought: The previous thought node
            guidance: Optional guidance that might influence stage

        Returns:
            Dictionary with stage name, thought type, and confidence
        """
        current_stage = previous_thought.metadata.get("stage", "observations")

        # Default progression through stages
        stage_progression = {
            "observations": {
                "stage": "hypothesis_generation",
                "thought_type": ThoughtType.HYPOTHESIS,
                "confidence": 0.6,
            },
            "hypothesis_generation": {
                "stage": "evaluation",
                "thought_type": ThoughtType.VERIFICATION,
                "confidence": 0.7,
            },
            "evaluation": {
                "stage": "conclusion",
                "thought_type": ThoughtType.CONCLUSION,
                "confidence": 0.75,
            },
            "conclusion": {
                "stage": "refinement",
                "thought_type": ThoughtType.REVISION,
                "confidence": 0.8,
            },
        }

        # Check if guidance requests a specific action
        if guidance:
            guidance_lower = guidance.lower()
            if "hypothesis" in guidance_lower or "generate" in guidance_lower:
                return {
                    "stage": "hypothesis_generation",
                    "thought_type": ThoughtType.HYPOTHESIS,
                    "confidence": 0.6,
                }
            elif "evaluat" in guidance_lower or "assess" in guidance_lower:
                return {
                    "stage": "evaluation",
                    "thought_type": ThoughtType.VERIFICATION,
                    "confidence": 0.7,
                }
            elif "conclude" in guidance_lower or "select" in guidance_lower:
                return {
                    "stage": "conclusion",
                    "thought_type": ThoughtType.CONCLUSION,
                    "confidence": 0.75,
                }
            elif "observ" in guidance_lower or "evidence" in guidance_lower:
                return {
                    "stage": "observations",
                    "thought_type": ThoughtType.OBSERVATION,
                    "confidence": 0.8,
                }

        # Default progression
        return stage_progression.get(
            current_stage,
            {
                "stage": "continuation",
                "thought_type": ThoughtType.CONTINUATION,
                "confidence": 0.7,
            },
        )

    def _generate_stage_content(
        self,
        stage: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content appropriate for the current stage.

        Args:
            stage: The current stage name
            previous_thought: The previous thought node
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            Generated content for this stage
        """
        if stage == "hypothesis_generation":
            return self._generate_hypotheses(previous_thought, context)
        elif stage == "evaluation":
            return self._generate_evaluation(previous_thought, context)
        elif stage == "conclusion":
            return self._generate_conclusion(previous_thought, context)
        elif stage == "observations":
            return self._generate_additional_observations(previous_thought, context)
        else:
            return self._generate_continuation(previous_thought, guidance, context)

    def _generate_hypotheses(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate candidate hypotheses based on observations.

        Args:
            previous_thought: The previous thought with observations
            context: Optional context

        Returns:
            Content with generated hypotheses
        """
        # Extract observations
        observations = previous_thought.metadata.get("observations", self._observations)

        # Generate candidate hypotheses (in real implementation, this would use LLM)
        # For now, create a template structure
        hypotheses = [
            {
                "id": 1,
                "explanation": "Hypothesis 1: Most common explanation",
                "supports": ["observation patterns", "known causes"],
                "likelihood": "high",
            },
            {
                "id": 2,
                "explanation": "Hypothesis 2: Alternative explanation",
                "supports": ["specific evidence", "edge cases"],
                "likelihood": "medium",
            },
            {
                "id": 3,
                "explanation": "Hypothesis 3: Less likely but possible",
                "supports": ["rare scenarios"],
                "likelihood": "low",
            },
        ]

        self._hypotheses = hypotheses

        hyp_text = "\n\n".join(
            f"{h['explanation']}\n"
            f"  Likelihood: {h['likelihood']}\n"
            f"  Supports: {', '.join(h['supports'])}"
            for h in hypotheses
        )

        return (
            f"Step {self._step_counter}: Hypothesis Generation\n\n"
            f"Based on the observations, I'll generate candidate explanations "
            f"that could account for the evidence:\n\n"
            f"{hyp_text}\n\n"
            f"Each hypothesis provides a potential explanation. Next, I'll evaluate "
            f"them against the evidence to determine which is most plausible."
        )

    def _generate_evaluation(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Evaluate hypotheses against evidence.

        Args:
            previous_thought: The previous thought with hypotheses
            context: Optional context

        Returns:
            Content with hypothesis evaluation
        """
        hypotheses_count = len(self._hypotheses)

        return (
            f"Step {self._step_counter}: Hypothesis Evaluation\n\n"
            f"Now I'll evaluate the {hypotheses_count} candidate hypotheses "
            f"against the available evidence:\n\n"
            f"Evaluation Criteria:\n"
            f"  1. Explanatory power: How well does it explain ALL observations?\n"
            f"  2. Simplicity: Is it the simplest explanation (Occam's Razor)?\n"
            f"  3. Consistency: Does it align with known facts and principles?\n"
            f"  4. Testability: Can we verify or falsify this explanation?\n\n"
            f"Applying these criteria to each hypothesis to identify the best explanation..."
        )

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Select and present the best explanation.

        Args:
            previous_thought: The previous thought with evaluation
            context: Optional context

        Returns:
            Content with final conclusion
        """
        return (
            f"Step {self._step_counter}: Best Explanation\n\n"
            f"After evaluating all candidate hypotheses against the evidence, "
            f"the most plausible explanation is:\n\n"
            f"[Selected hypothesis with highest explanatory power and consistency]\n\n"
            f"Reasoning:\n"
            f"  - Accounts for all major observations\n"
            f"  - Provides the simplest coherent explanation\n"
            f"  - Aligns with established knowledge\n"
            f"  - Can be tested or verified\n\n"
            f"Confidence level: Moderate to high, depending on evidence completeness.\n\n"
            f"Note: As with all abductive reasoning, this is the BEST explanation "
            f"given current evidence, but not guaranteed to be the only or final truth. "
            f"New evidence could lead to revising this conclusion."
        )

    def _generate_additional_observations(
        self,
        previous_thought: ThoughtNode,
        context: dict[str, Any] | None,
    ) -> str:
        """Add new observations or evidence.

        Args:
            previous_thought: The previous thought
            context: Optional context with new evidence

        Returns:
            Content with additional observations
        """
        new_evidence = []
        if context and "new_evidence" in context:
            new_evidence = context["new_evidence"]
            if isinstance(new_evidence, str):
                new_evidence = [new_evidence]

        evidence_text = "\n".join(f"  - {ev}" for ev in new_evidence) if new_evidence else "  - [New observation]"

        return (
            f"Step {self._step_counter}: Additional Observations\n\n"
            f"New evidence has emerged that may affect our hypotheses:\n\n"
            f"{evidence_text}\n\n"
            f"I'll incorporate this into the analysis and may need to revise "
            f"earlier hypotheses based on this new information."
        )

    def _generate_continuation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate general continuation content.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Content for continuation
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Continuing Analysis\n\n"
            f"Building on the previous step, I'll continue the abductive reasoning "
            f"process to refine our understanding and move toward the best "
            f"explanation.{guidance_text}"
        )
