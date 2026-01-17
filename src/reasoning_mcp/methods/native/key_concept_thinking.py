"""Key-Concept Thinking (KCT) reasoning method.

This module implements Key-Concept Thinking (Zheng et al. 2025), which improves
reasoning by first extracting and defining the key domain concepts needed to
solve a problem, then using these concepts to guide the reasoning process.

Key phases:
1. Extract: Identify key concepts from the problem
2. Define: Clarify each concept's meaning and relationships
3. Apply: Use concepts to structure the reasoning
4. Solve: Derive solution using concept-based framework

Reference: Zheng et al. (2025) - "Key-Concept Thinking for Enhanced Reasoning"
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


# Metadata for Key-Concept Thinking method
KEY_CONCEPT_THINKING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.KEY_CONCEPT_THINKING,
    name="Key-Concept Thinking",
    description="Improves reasoning by first extracting and defining key domain "
    "concepts, then using them to structure the solution. Follows extract → "
    "define → apply → solve phases for concept-grounded reasoning.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "concept-extraction",
            "domain-knowledge",
            "structured-reasoning",
            "knowledge-grounding",
            "concept-based",
            "definitions",
            "semantic",
            "2025-research",
        }
    ),
    complexity=6,  # Moderate complexity
    supports_branching=False,  # Linear concept-based flow
    supports_revision=True,  # Can refine concept understanding
    requires_context=False,  # No special context needed
    min_thoughts=4,  # extract + define + apply + solve
    max_thoughts=8,  # Multiple concepts may need attention
    avg_tokens_per_thought=350,  # Concept definitions can be detailed
    best_for=(
        "domain-specific problems",
        "technical questions",
        "terminology-heavy tasks",
        "educational content",
        "concept disambiguation",
        "specialized fields",
        "problems with jargon",
        "cross-domain reasoning",
    ),
    not_recommended_for=(
        "simple factual queries",
        "creative tasks",
        "purely numerical computation",
        "tasks without clear concepts",
    ),
)

logger = structlog.get_logger(__name__)


class KeyConceptThinking(ReasoningMethodBase):
    """Key-Concept Thinking reasoning method implementation.

    This class implements the KCT pattern:
    1. Extract: Identify key concepts from the problem
    2. Define: Clarify meanings and relationships
    3. Apply: Structure reasoning around concepts
    4. Solve: Derive solution using concept framework

    Key characteristics:
    - Concept-grounded reasoning
    - Explicit definitions
    - Domain knowledge integration
    - Moderate complexity (6)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = KeyConceptThinking()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Explain the difference between supervised and unsupervised learning"
        ... )
        >>> print(result.content)  # Concept extraction phase
    """

    # Maximum concepts to extract
    MAX_CONCEPTS = 5

    def __init__(self) -> None:
        """Initialize the Key-Concept Thinking method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "extract"
        self._extracted_concepts: list[str] = []
        self._concept_definitions: dict[str, str] = {}
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.KEY_CONCEPT_THINKING

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return KEY_CONCEPT_THINKING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return KEY_CONCEPT_THINKING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Key-Concept Thinking method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "extract"
        self._extracted_concepts = []
        self._concept_definitions = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Key-Concept Thinking method.

        Creates the initial concept extraction phase.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the extraction phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Key-Concept Thinking method must be initialized before execution")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "extract"
        self._extracted_concepts = []
        self._concept_definitions = {}

        # Generate extraction content
        if use_sampling:
            content = await self._sample_extraction(input_text, context)
        else:
            content = self._generate_extraction(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.KEY_CONCEPT_THINKING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "key_concept_thinking",
                "phase": self._current_phase,
                "concepts": self._extracted_concepts,
                "sampled": use_sampling,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.KEY_CONCEPT_THINKING

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

        Implements the KCT phase progression:
        - After extract: define concepts
        - After define: apply concepts
        - After apply: solve
        - After solve: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the KCT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Key-Concept Thinking method must be initialized before continuation"
            )

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "extract")

        # Check if we can use sampling
        use_sampling = (
            self._execution_context is not None
            and self._execution_context.can_sample
            and self._use_sampling
        )

        if prev_phase == "extract":
            # Define extracted concepts
            self._current_phase = "define"
            thought_type = ThoughtType.REASONING
            if use_sampling:
                content = await self._sample_definitions(guidance, context)
            else:
                content = self._generate_definitions(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "define":
            # Apply concepts to problem
            self._current_phase = "apply"
            thought_type = ThoughtType.SYNTHESIS
            if use_sampling:
                content = await self._sample_application(guidance, context)
            else:
                content = self._generate_application(guidance, context)
            confidence = 0.8
            quality_score = 0.8

        elif prev_phase == "apply":
            # Solve using concept framework
            self._current_phase = "solve"
            thought_type = ThoughtType.REASONING
            if use_sampling:
                content = await self._sample_solution(guidance, context)
            else:
                content = self._generate_solution(guidance, context)
            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "solve":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if use_sampling:
                content = await self._sample_conclusion(guidance, context)
            else:
                content = self._generate_conclusion(guidance, context)
            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback
            self._current_phase = "solve"
            thought_type = ThoughtType.REASONING
            if use_sampling:
                content = await self._sample_solution(guidance, context)
            else:
                content = self._generate_solution(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.KEY_CONCEPT_THINKING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "key_concept_thinking",
                "concepts": self._extracted_concepts,
                "definitions": self._concept_definitions,
                "previous_phase": prev_phase,
                "sampled": use_sampling,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    async def _sample_extraction(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample LLM for concept extraction.

        Args:
            input_text: The problem to analyze
            context: Optional additional context

        Returns:
            The content for the extraction phase
        """
        system_prompt = """You are a key-concept thinking assistant.
Your task is to identify and extract the most important domain-specific concepts from a
given problem. Focus on:
1. Technical terms that need precise definition
2. Core concepts central to understanding the problem
3. Domain-specific terminology
4. Concepts that have relationships to each other

Extract 3-5 key concepts that are essential for solving this problem."""

        user_prompt = f"""Problem: {input_text}

Extract the key concepts from this problem:
1. Identify the 3-5 most important domain concepts
2. List each concept clearly
3. Explain briefly why each concept is key to this problem

Begin your key concept extraction."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_extraction(input_text, context),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=600,
        )

    async def _sample_definitions(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample LLM for concept definitions.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the definition phase
        """
        system_prompt = """You are a key-concept thinking assistant.
Your task is to provide precise definitions for the key concepts that were extracted.
For each concept, provide:
1. A clear, accurate definition
2. Key properties or characteristics
3. How it relates to other concepts in the problem"""

        concepts_text = "\n".join(f"- {c}" for c in self._extracted_concepts)
        user_prompt = f"""Define each of these key concepts precisely:

{concepts_text}

For each concept, provide:
1. Definition: A clear and precise definition
2. Properties: Key characteristics or properties
3. Relationships: How it relates to other concepts

Begin your concept definitions."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_definitions(guidance, context),
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=800,
        )

    async def _sample_application(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample LLM for concept application.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the application phase
        """
        system_prompt = """You are a key-concept thinking assistant.
Your task is to apply the defined key concepts to the problem at hand.
For each concept, explain:
1. How it applies to this specific problem
2. What implications it has for the solution
3. How concepts work together to frame the solution"""

        concepts_text = "\n".join(
            f"- {name}: {defn}" for name, defn in self._concept_definitions.items()
        )
        user_prompt = f"""Apply these defined concepts to solve the problem:

{concepts_text}

For each concept, explain:
1. How it applies to this problem
2. What implications it has
3. How concepts combine to provide insights

Begin your concept application."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_application(guidance, context),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=700,
        )

    async def _sample_solution(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample LLM for concept-based solution.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the solution phase
        """
        system_prompt = """You are a key-concept thinking assistant.
Your task is to derive a solution using the concept framework you've built.
Use the defined concepts and their applications to:
1. Build a logical reasoning chain
2. Derive inferences from each concept
3. Synthesize a complete solution
4. Verify the solution is consistent with all concepts"""

        user_prompt = """Using the key concepts and their applications, derive the solution:

1. Build a reasoning chain from the concepts
2. Show how each concept contributes to the solution
3. Synthesize the final answer
4. Verify consistency with all concepts

Begin your concept-based solution."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_solution(guidance, context),
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=800,
        )

    async def _sample_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Sample LLM for final conclusion.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the conclusion phase
        """
        system_prompt = """You are a key-concept thinking assistant.
Your task is to provide a final conclusion that summarizes the key-concept thinking process.
Include:
1. Summary of concepts used
2. Final answer grounded in the concept framework
3. Confidence level and reasoning"""

        user_prompt = f"""Provide the final conclusion for this key-concept thinking analysis:

Concepts analyzed: {len(self._extracted_concepts)}
Concepts defined: {len(self._concept_definitions)}

Provide:
1. Brief summary of the concept-based reasoning
2. Final answer clearly stated
3. Confidence level with justification

Begin your final conclusion."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_conclusion(guidance, context),
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=500,
        )

    def _generate_extraction(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the concept extraction phase content."""
        # Simulate concept extraction
        self._extracted_concepts = [
            "Concept 1: [First key concept from problem]",
            "Concept 2: [Second key concept from problem]",
            "Concept 3: [Third key concept from problem]",
        ]

        return (
            f"Step {self._step_counter}: Key Concept Extraction\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing problem for key concepts...\n\n"
            f"Extracted Key Concepts:\n"
            f"┌────────────────────────────────────────┐\n"
            f"│ 1. [First key domain concept]         │\n"
            f"│ 2. [Second key domain concept]        │\n"
            f"│ 3. [Third key domain concept]         │\n"
            f"└────────────────────────────────────────┘\n\n"
            f"These concepts form the foundation for solving this problem.\n"
            f"Next: Define each concept precisely."
        )

    def _generate_definitions(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate concept definitions."""
        self._concept_definitions = {
            "Concept 1": "Definition: [precise definition]",
            "Concept 2": "Definition: [precise definition]",
            "Concept 3": "Definition: [precise definition]",
        }

        return (
            f"Step {self._step_counter}: Concept Definitions\n\n"
            f"Defining extracted concepts precisely...\n\n"
            f"Concept Definitions:\n\n"
            f"📖 Concept 1:\n"
            f"   Definition: [Precise definition of concept 1]\n"
            f"   Properties: [Key properties or characteristics]\n"
            f"   Related to: [How it relates to other concepts]\n\n"
            f"📖 Concept 2:\n"
            f"   Definition: [Precise definition of concept 2]\n"
            f"   Properties: [Key properties or characteristics]\n"
            f"   Related to: [How it relates to other concepts]\n\n"
            f"📖 Concept 3:\n"
            f"   Definition: [Precise definition of concept 3]\n"
            f"   Properties: [Key properties or characteristics]\n"
            f"   Related to: [How it relates to other concepts]\n\n"
            f"Concept relationships established. Ready to apply to problem."
        )

    def _generate_application(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate concept application."""
        return (
            f"Step {self._step_counter}: Applying Concepts to Problem\n\n"
            f"Structuring solution using key concepts...\n\n"
            f"Concept Application Framework:\n\n"
            f"Problem → Concept 1:\n"
            f"  - How Concept 1 applies: [application]\n"
            f"  - Implications: [what this means for the solution]\n\n"
            f"Problem → Concept 2:\n"
            f"  - How Concept 2 applies: [application]\n"
            f"  - Implications: [what this means for the solution]\n\n"
            f"Problem → Concept 3:\n"
            f"  - How Concept 3 applies: [application]\n"
            f"  - Implications: [what this means for the solution]\n\n"
            f"Synthesis:\n"
            f"  Combining concept applications reveals: [insight]\n\n"
            f"Ready to derive solution from concept framework."
        )

    def _generate_solution(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the concept-based solution."""
        return (
            f"Step {self._step_counter}: Deriving Solution from Concepts\n\n"
            f"Using concept framework to solve problem...\n\n"
            f"Reasoning Chain:\n"
            f"  1. From Concept 1 definition: [inference]\n"
            f"  2. From Concept 2 properties: [inference]\n"
            f"  3. From Concept 3 relationships: [inference]\n"
            f"  4. Combining inferences: [synthesis]\n\n"
            f"Solution:\n"
            f"┌────────────────────────────────────────┐\n"
            f"│ [Solution derived from concept-based  │\n"
            f"│  reasoning framework]                 │\n"
            f"└────────────────────────────────────────┘\n\n"
            f"Verification:\n"
            f"  ✓ Solution consistent with Concept 1\n"
            f"  ✓ Solution consistent with Concept 2\n"
            f"  ✓ Solution consistent with Concept 3"
        )

    def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Key-Concept Thinking Analysis Complete:\n\n"
            f"Summary:\n"
            f"  - Concepts extracted: {len(self._extracted_concepts)}\n"
            f"  - Concepts defined: {len(self._concept_definitions)}\n"
            f"  - Solution grounded in concept framework\n\n"
            f"Final Answer: [Concept-grounded answer]\n\n"
            f"Confidence: High (90%)\n"
            f"Reason: Solution derived from explicit concept definitions\n"
            f"and verified against each key concept's properties."
        )


# Export
__all__ = ["KeyConceptThinking", "KEY_CONCEPT_THINKING_METADATA"]
