"""Analogical Reasoning method.

This module implements analogical reasoning - solving problems by finding similar
situations or domains and transferring insights through structural mapping. This
method is particularly effective for creative problem-solving, teaching, and
understanding unfamiliar concepts through familiar ones.
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


# Metadata for Analogical Reasoning method
ANALOGICAL_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ANALOGICAL,
    name="Analogical Reasoning",
    description="Solve problems by finding analogous situations and transferring "
    "insights through structural mapping. Maps relationships from a familiar source "
    "domain to an unfamiliar target domain.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "analogical",
            "analogy",
            "mapping",
            "transfer",
            "creative",
            "teaching",
            "comparison",
            "structural",
        }
    ),
    complexity=4,  # Medium complexity - requires mapping and validation
    supports_branching=True,  # Can explore multiple analogies
    supports_revision=True,  # Can revise mappings
    requires_context=False,  # No special context needed
    min_thoughts=5,  # Structure, source, mapping, transfer, validation
    max_thoughts=0,  # No limit - can explore multiple analogies
    avg_tokens_per_thought=400,
    best_for=(
        "creative problem-solving",
        "teaching complex concepts",
        "understanding unfamiliar domains",
        "finding innovative solutions",
        "cross-domain insights",
        "explaining by comparison",
    ),
    not_recommended_for=(
        "problems requiring precise numerical solutions",
        "formal mathematical proofs",
        "when direct solutions are obvious",
        "problems where analogies could mislead",
    ),
)

logger = structlog.get_logger(__name__)


class Analogical(ReasoningMethodBase):
    """Analogical Reasoning method implementation.

    This class implements reasoning by analogy - a powerful cognitive technique
    that maps structural relationships from a familiar source domain to an
    unfamiliar target domain. The process involves:

    1. Identifying the target problem structure
    2. Finding an analogous source domain
    3. Mapping structural relationships
    4. Transferring insights to the target
    5. Validating the analogy

    Key characteristics:
    - Source-to-target domain mapping
    - Structural relationship preservation
    - Insight transfer across domains
    - Analogy validation
    - Supports exploring multiple analogies (branching)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = Analogical()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can we improve employee motivation?"
        ... )
        >>> print(result.content)  # Target problem analysis

        Continue with analogy development:
        >>> next_thought = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Find a source domain"
        ... )
        >>> print(next_thought.step_number)  # 2
    """

    def __init__(self) -> None:
        """Initialize the Analogical Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_stage = "target_analysis"
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None
        # Stages: target_analysis -> source_identification -> structural_mapping
        #         -> insight_transfer -> validation

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.ANALOGICAL

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return ANALOGICAL_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return ANALOGICAL_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Analogical Reasoning method for execution.

        Examples:
            >>> method = Analogical()
            >>> await method.initialize()
            >>> assert method._initialized is True
        """
        self._initialized = True
        self._step_counter = 0
        self._current_stage = "target_analysis"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Analogical Reasoning method.

        This method begins the analogical reasoning process by analyzing the
        target problem and identifying its key structural elements.

        Args:
            session: The current reasoning session
            input_text: The target problem to solve through analogy
            context: Optional additional context (e.g., preferred source domains)
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the target problem analysis

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Analogical()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How to manage team conflicts effectively?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.ANALOGICAL
        """
        if not self._initialized:
            raise RuntimeError("Analogical Reasoning method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_stage = "target_analysis"

        # Create the initial thought - target problem analysis
        content = await self._generate_target_analysis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ANALOGICAL,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial confidence before finding analogy
            metadata={
                "input": input_text,
                "context": context or {},
                "stage": self._current_stage,
                "reasoning_type": "analogical",
                "target_problem": input_text,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.ANALOGICAL

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
        """Continue analogical reasoning from a previous thought.

        This method progresses through the stages of analogical reasoning:
        1. Target analysis (initial)
        2. Source identification
        3. Structural mapping
        4. Insight transfer
        5. Validation

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (e.g., "try a different source domain")
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the analogical reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = Analogical()
            >>> await method.initialize()
            >>> first = await method.execute(session, "Improve customer retention")
            >>> second = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=first,
            ...     guidance="Find source domain"
            ... )
            >>> assert second.step_number == 2
            >>> assert second.parent_id == first.id
        """
        if not self._initialized:
            raise RuntimeError(
                "Analogical Reasoning method must be initialized before continuation"
            )

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next stage and thought type
        thought_type, next_stage = self._determine_next_stage(previous_thought, guidance)

        # Generate content for the current stage
        content = await self._generate_stage_content(
            stage=next_stage,
            previous_thought=previous_thought,
            guidance=guidance,
            context=context,
        )

        # Calculate confidence based on stage
        confidence = self._calculate_confidence(next_stage, previous_thought.confidence)

        # Create metadata for this thought
        metadata = {
            "previous_step": previous_thought.step_number,
            "stage": next_stage,
            "guidance": guidance or "",
            "context": context or {},
            "reasoning_type": "analogical",
            "sampled": self._use_sampling,
        }

        # Add stage-specific metadata
        if next_stage == "source_identification":
            metadata["source_domain"] = "placeholder_source"
        elif next_stage == "structural_mapping":
            metadata["mappings"] = ["placeholder_mapping"]
        elif next_stage == "insight_transfer":
            metadata["transferred_insights"] = ["placeholder_insight"]
        elif next_stage == "validation":
            metadata["validation_criteria"] = ["placeholder_criterion"]

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.ANALOGICAL,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata=metadata,
        )

        # Update current stage
        self._current_stage = next_stage

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = Analogical()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _determine_next_stage(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> tuple[ThoughtType, str]:
        """Determine the next stage and thought type based on current state.

        Args:
            previous_thought: The previous thought
            guidance: Optional guidance that might affect next stage

        Returns:
            Tuple of (ThoughtType, stage_name)
        """
        current_stage = previous_thought.metadata.get("stage", self._current_stage)

        # Check if guidance suggests branching to explore alternative analogies
        if guidance and any(
            keyword in guidance.lower()
            for keyword in ["different", "alternative", "another", "branch"]
        ):
            # Branch to explore a different source domain
            return ThoughtType.BRANCH, "source_identification"

        # Normal progression through stages
        stage_progression = {
            "target_analysis": ("source_identification", ThoughtType.CONTINUATION),
            "source_identification": ("structural_mapping", ThoughtType.CONTINUATION),
            "structural_mapping": ("insight_transfer", ThoughtType.CONTINUATION),
            "insight_transfer": ("validation", ThoughtType.VERIFICATION),
            "validation": ("validation", ThoughtType.CONCLUSION),  # Can conclude
        }

        next_stage, thought_type = stage_progression.get(
            current_stage, ("validation", ThoughtType.CONCLUSION)
        )

        return thought_type, next_stage

    def _calculate_confidence(self, stage: str, previous_confidence: float) -> float:
        """Calculate confidence score based on reasoning stage.

        Args:
            stage: Current reasoning stage
            previous_confidence: Previous thought's confidence

        Returns:
            Confidence score (0.0 - 1.0)
        """
        # Confidence generally increases as we validate the analogy
        stage_confidence = {
            "target_analysis": 0.6,
            "source_identification": 0.65,
            "structural_mapping": 0.7,
            "insight_transfer": 0.75,
            "validation": 0.8,
        }

        base_confidence = stage_confidence.get(stage, 0.6)

        # Blend with previous confidence
        return (base_confidence + previous_confidence) / 2

    async def _generate_target_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the target problem analysis.

        Args:
            input_text: The target problem
            context: Optional additional context

        Returns:
            The content for target analysis
        """
        system_prompt = """You are a reasoning assistant using Analogical Reasoning methodology.
Your task is to analyze the target problem and identify its key structural elements.

Focus on:
1. Core Challenge: The fundamental issue
2. Key Components: Essential parts of the problem
3. Relationships: How components interact
4. Desired Outcome: What success looks like
5. Constraints: Limitations and boundaries

Be thorough but concise. Set up for finding analogous domains."""

        user_prompt = f"""Target Problem: {input_text}

Analyze the structural elements of this problem following the Analogical Reasoning framework.
Identify the core challenge, key components, relationships, desired outcome, and constraints.
End by noting that you'll search for a familiar domain with similar structural patterns."""

        step_counter = self._step_counter

        def fallback() -> str:
            return (
                f"Step {step_counter}: Target Problem Analysis\n\n"
                f"Target Problem: {input_text}\n\n"
                f"Let me analyze the structural elements of this problem:\n\n"
                f"1. Core Challenge: Identifying the fundamental issue at play\n"
                f"2. Key Components: Breaking down the problem into its essential parts\n"
                f"3. Relationships: Understanding how components interact\n"
                f"4. Desired Outcome: Clarifying what success looks like\n"
                f"5. Constraints: Recognizing limitations and boundaries\n\n"
                f"Next, I'll search for a familiar domain with similar structural "
                f"patterns that can provide insights."
            )

        if not self._use_sampling:
            return fallback()

        response = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # If fallback was returned (doesn't contain "Step"), format it
        if not response.startswith(f"Step {step_counter}"):
            return f"Step {step_counter}: Target Problem Analysis\n\n{response}"
        return response

    async def _generate_stage_content(
        self,
        stage: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for a specific reasoning stage.

        Args:
            stage: The current stage
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Generated content for the stage
        """
        if self._use_sampling and self._execution_context:
            return await self._sample_stage_content(stage, previous_thought, guidance, context)

        # Fallback to heuristic generation
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        if stage == "source_identification":
            return (
                f"Step {self._step_counter}: Source Domain Identification\n\n"
                f"Building on the target analysis, I'll identify an analogous source "
                f"domain that shares structural similarities:\n\n"
                f"Potential Source Domain: [A familiar domain with similar patterns]\n\n"
                f"Why this analogy works:\n"
                f"1. Shared Structure: Both domains have similar relational patterns\n"
                f"2. Known Solutions: The source domain has established approaches\n"
                f"3. Transferability: Insights can map meaningfully to the target\n"
                f"4. Accessibility: The source is familiar and well-understood\n\n"
                f"I'll now map the structural relationships between source and target."
                f"{guidance_text}"
            )

        elif stage == "structural_mapping":
            return (
                f"Step {self._step_counter}: Structural Mapping\n\n"
                f"Mapping relationships from source to target domain:\n\n"
                f"Source → Target Mappings:\n"
                f"1. Component A in source ↔ Component X in target\n"
                f"2. Relationship B in source ↔ Relationship Y in target\n"
                f"3. Process C in source ↔ Process Z in target\n\n"
                f"Key Similarities:\n"
                f"- Both involve similar interaction patterns\n"
                f"- Underlying dynamics are comparable\n"
                f"- Success factors align across domains\n\n"
                f"Important Differences:\n"
                f"- Context-specific constraints differ\n"
                f"- Scale and complexity may vary\n\n"
                f"Next, I'll transfer insights from the source to the target."
                f"{guidance_text}"
            )

        elif stage == "insight_transfer":
            return (
                f"Step {self._step_counter}: Insight Transfer\n\n"
                f"Transferring successful approaches from source to target:\n\n"
                f"Transferred Insights:\n"
                f"1. Approach from source → Adapted approach for target\n"
                f"2. Strategy from source → Modified strategy for target\n"
                f"3. Principle from source → Applied principle in target\n\n"
                f"Adaptations Required:\n"
                f"- Adjusting for target-specific constraints\n"
                f"- Scaling to appropriate level\n"
                f"- Integrating with existing practices\n\n"
                f"Potential Solutions:\n"
                f"[Concrete solutions derived from the analogy]\n\n"
                f"Now I'll validate this analogy to ensure it holds up to scrutiny."
                f"{guidance_text}"
            )

        elif stage == "validation":
            return (
                f"Step {self._step_counter}: Analogy Validation\n\n"
                f"Validating the analogy and transferred insights:\n\n"
                f"Validation Criteria:\n"
                f"1. Structural Soundness: Do the mappings preserve key relationships?\n"
                f"2. Insight Quality: Are the transferred insights actually useful?\n"
                f"3. Practical Applicability: Can solutions be implemented?\n"
                f"4. Limitations: Where does the analogy break down?\n\n"
                f"Strengths of this Analogy:\n"
                f"- Captures essential problem structure\n"
                f"- Provides actionable insights\n"
                f"- Illuminates previously hidden patterns\n\n"
                f"Limitations to Consider:\n"
                f"- Areas where the analogy doesn't quite fit\n"
                f"- Risks of over-extending the comparison\n"
                f"- Context-specific factors not captured\n\n"
                f"Conclusion: [Overall assessment of the analogical reasoning]"
                f"{guidance_text}"
            )

        else:
            return (
                f"Step {self._step_counter}: Continuing analogical reasoning\n\n"
                f"Building on previous insights...{guidance_text}"
            )

    async def _sample_stage_content(
        self,
        stage: str,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate stage content using LLM sampling.

        Uses the execution context's sampling capability to generate
        content for each stage of analogical reasoning.

        Args:
            stage: The current stage
            previous_thought: The previous thought
            guidance: Optional guidance
            context: Optional context

        Returns:
            Generated content for the stage
        """
        if self._execution_context is None:
            raise RuntimeError(
                "_sample_stage_content requires execution context but was not provided"
            )

        guidance_text = f"\n\nUser Guidance: {guidance}" if guidance else ""

        # Build stage-specific prompts
        stage_instructions = {
            "source_identification": """Identify an analogous source domain that shares structural similarities with the target problem.

Explain:
1. What source domain you've identified
2. Why this analogy works (shared structure, known solutions, transferability, accessibility)
3. How insights can map meaningfully

End by noting you'll map the structural relationships.""",
            "structural_mapping": """Map the structural relationships from the source domain to the target domain.

Provide:
1. Specific source → target component mappings
2. Key similarities in patterns and dynamics
3. Important differences in context and constraints

End by noting you'll transfer insights next.""",
            "insight_transfer": """Transfer successful approaches from the source domain to the target domain.

Show:
1. Specific insights transferred and how they're adapted
2. Required adaptations for target-specific constraints
3. Concrete potential solutions derived from the analogy

End by noting you'll validate this analogy.""",
            "validation": """Validate the analogy and transferred insights.

Assess:
1. Structural soundness: Do mappings preserve key relationships?
2. Insight quality: Are transferred insights useful?
3. Practical applicability: Can solutions be implemented?
4. Limitations: Where does the analogy break down?

Provide a conclusion with overall assessment.""",
        }

        instruction = stage_instructions.get(stage, "Continue the analogical reasoning process.")

        system_prompt = f"""You are a reasoning assistant using Analogical Reasoning methodology.
You are at the {stage.replace("_", " ")} stage.

{instruction}

Be specific and concrete. Build on previous analysis."""

        user_prompt = f"""Previous Analysis:
{previous_thought.content}

Current Stage: {stage.replace("_", " ").title()}

{instruction}{guidance_text}"""

        step_counter = self._step_counter
        stage_title = stage.replace("_", " ").title()
        guidance_text_fallback = f"\n\nGuidance: {guidance}" if guidance else ""

        def fallback() -> str:
            return (
                f"Step {step_counter}: {stage_title}\n\n"
                f"Building on previous insights...{guidance_text_fallback}"
            )

        response = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # If response is from LLM (doesn't have "Building on" fallback text), format it
        if "Building on previous insights" not in response:
            return f"Step {step_counter}: {stage_title}\n\n{response}"
        return response
