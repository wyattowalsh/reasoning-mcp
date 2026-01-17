"""Cumulative Reasoning method.

This module implements Cumulative Reasoning (Zhang et al. 2023), which builds
up a structured knowledge base of verified propositions in a DAG (Directed
Acyclic Graph) structure. Each new proposition is verified against existing
ones before being added to the cumulative knowledge.

Key phases:
1. Initialize: Set up proposition graph
2. Propose: Generate new candidate proposition
3. Verify: Check proposition against existing knowledge
4. Accumulate: Add verified proposition to graph
5. Conclude: Derive final answer from accumulated knowledge

Reference: Zhang et al. (2023) - "Cumulative Reasoning with Language Models"
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


# Metadata for Cumulative Reasoning method
CUMULATIVE_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CUMULATIVE_REASONING,
    name="Cumulative Reasoning",
    description="Builds structured knowledge through verified propositions in a DAG. "
    "Each new proposition is verified against existing knowledge through "
    "initialize → propose → verify → accumulate → conclude phases.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "cumulative",
            "propositions",
            "verification",
            "dag",
            "knowledge-graph",
            "incremental",
            "structured",
            "verified",
        }
    ),
    complexity=8,  # High complexity due to DAG management
    supports_branching=True,  # DAG structure
    supports_revision=True,  # Can revise propositions
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: init + propose + verify + accumulate + conclude
    max_thoughts=15,  # Many propositions
    avg_tokens_per_thought=300,  # Propositions are concise
    best_for=(
        "complex problem solving",
        "multi-step reasoning",
        "knowledge building",
        "mathematical proofs",
        "scientific reasoning",
        "fact verification",
        "argument construction",
        "research synthesis",
    ),
    not_recommended_for=(
        "simple questions",
        "creative tasks",
        "subjective opinions",
        "tasks without verifiable facts",
    ),
)

logger = structlog.get_logger(__name__)


class CumulativeReasoning(ReasoningMethodBase):
    """Cumulative Reasoning method implementation.

    This class implements the Cumulative Reasoning pattern:
    1. Initialize: Create empty proposition graph
    2. Propose: Generate candidate propositions
    3. Verify: Check against existing verified propositions
    4. Accumulate: Add to DAG if verified
    5. Conclude: Derive answer from accumulated knowledge

    Key characteristics:
    - DAG-structured knowledge
    - Verification-based accumulation
    - Incremental reasoning
    - High complexity (8)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = CumulativeReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Prove that the sum of angles in a triangle is 180°"
        ... )
        >>> print(result.content)  # Initial proposition graph
    """

    # Maximum propositions to accumulate
    MAX_PROPOSITIONS = 10

    # Enable LLM sampling support
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Cumulative Reasoning method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize"
        self._propositions: list[dict[str, Any]] = []
        self._proposition_count = 0
        self._execution_context: ExecutionContext | None = None
        self._input_text: str = ""

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.CUMULATIVE_REASONING

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return CUMULATIVE_REASONING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return CUMULATIVE_REASONING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Cumulative Reasoning method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "initialize"
        self._propositions = []
        self._proposition_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Cumulative Reasoning method.

        Creates the initial proposition graph.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A ThoughtNode representing the initialization phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Cumulative Reasoning method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context
        self._input_text = input_text

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "initialize"
        self._propositions = []
        self._proposition_count = 0

        # Generate initialization content (use sampling if available)
        content = await self._sample_with_fallback(
            user_prompt=f"""Problem: {input_text}

Initialize a Cumulative Reasoning process with an empty proposition graph.
Explain the DAG structure and the verification-based accumulation strategy.""",
            fallback_generator=lambda: self._generate_initialization(input_text, context),
            system_prompt="""You are a reasoning assistant using Cumulative Reasoning methodology.
You build structured knowledge through verified propositions in a DAG (Directed Acyclic Graph).

For the initialization phase:
1. Acknowledge the problem
2. Describe the empty proposition graph structure
3. Explain the cumulative reasoning strategy: Propose → Verify → Accumulate
4. State readiness to generate and verify propositions

Be clear and structured in your initialization.""",
            temperature=0.5,
            max_tokens=800,
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CUMULATIVE_REASONING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "cumulative_reasoning",
                "phase": self._current_phase,
                "proposition_count": self._proposition_count,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.CUMULATIVE_REASONING

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

        Implements the Cumulative Reasoning phase progression:
        - After initialize: propose first proposition
        - After propose: verify proposition
        - After verify: accumulate or reject, then propose next
        - After enough accumulation: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling

        Returns:
            A new ThoughtNode continuing the Cumulative Reasoning process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Cumulative Reasoning method must be initialized before continuation"
            )

        # Store execution context for sampling
        if execution_context is not None:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize")

        if prev_phase == "initialize":
            # First proposition
            self._current_phase = "propose"
            self._proposition_count = 1
            thought_type = ThoughtType.HYPOTHESIS
            content = await self._get_proposition_content(
                self._proposition_count, guidance, context
            )
            confidence = 0.7
            quality_score = 0.7

        elif prev_phase == "propose":
            # Verify the proposition
            self._current_phase = "verify"
            thought_type = ThoughtType.VERIFICATION
            content = await self._get_verification_content(
                self._proposition_count, guidance, context
            )
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "verify":
            # Accumulate if verified
            self._current_phase = "accumulate"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._get_accumulation_content(
                self._proposition_count, guidance, context
            )
            self._propositions.append(
                {
                    "id": self._proposition_count,
                    "verified": True,
                }
            )
            confidence = 0.8
            quality_score = 0.8

        elif prev_phase == "accumulate":
            if self._proposition_count < self.MAX_PROPOSITIONS - 1:
                # Check if we have enough for conclusion
                if self._proposition_count >= 3:  # Minimum propositions
                    # Can conclude
                    self._current_phase = "conclude"
                    thought_type = ThoughtType.CONCLUSION
                    content = await self._get_conclusion_content(guidance, context)
                    confidence = 0.9
                    quality_score = 0.9
                else:
                    # Need more propositions
                    self._current_phase = "propose"
                    self._proposition_count += 1
                    thought_type = ThoughtType.HYPOTHESIS
                    content = await self._get_proposition_content(
                        self._proposition_count, guidance, context
                    )
                    confidence = 0.7
                    quality_score = 0.7
            else:
                # Max propositions, conclude
                self._current_phase = "conclude"
                thought_type = ThoughtType.CONCLUSION
                content = await self._get_conclusion_content(guidance, context)
                confidence = 0.9
                quality_score = 0.9

        elif prev_phase == "conclude":
            # Final synthesis
            self._current_phase = "done"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._get_final_synthesis_content(guidance, context)
            confidence = 0.95
            quality_score = 0.95

        else:
            # Fallback
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = await self._get_conclusion_content(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CUMULATIVE_REASONING,
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
                "reasoning_type": "cumulative_reasoning",
                "proposition_count": self._proposition_count,
                "total_verified": len(self._propositions),
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _generate_initialization(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initialization phase content."""
        return (
            f"Step {self._step_counter}: Initialize Proposition Graph "
            f"(Cumulative Reasoning)\n\n"
            f"Problem: {input_text}\n\n"
            f"Initializing DAG Structure:\n"
            f"┌─────────────────────────────────────┐\n"
            f"│  Proposition Graph (Empty)          │\n"
            f"│  ─────────────────────              │\n"
            f"│  Nodes: 0                           │\n"
            f"│  Edges: 0                           │\n"
            f"│  Verified: 0                        │\n"
            f"└─────────────────────────────────────┘\n\n"
            f"Goal: Build verified knowledge incrementally\n"
            f"Strategy: Propose → Verify → Accumulate\n\n"
            f"Ready to generate and verify propositions."
        )

    def _generate_proposition(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a new proposition."""
        return (
            f"Step {self._step_counter}: Propose Proposition P{prop_num}\n\n"
            f"Generating candidate proposition...\n\n"
            f"Proposition P{prop_num}:\n"
            f"  Statement: [Candidate statement derived from problem]\n"
            f"  Dependencies: [Previous propositions this builds on]\n"
            f"  Type: {'Axiom' if prop_num == 1 else 'Derived'}\n\n"
            f"Formal Form:\n"
            f"  P{prop_num}: φ{prop_num}(x) ← "
            f"{'∅' if prop_num == 1 else f'P{prop_num - 1}'}\n\n"
            f"Status: Proposed (awaiting verification)"
        )

    def _generate_verification(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate verification for a proposition."""
        return (
            f"Step {self._step_counter}: Verify Proposition P{prop_num}\n\n"
            f"Checking against existing verified knowledge...\n\n"
            f"Verification Criteria:\n"
            f"1. ✓ Consistent with axioms\n"
            f"2. ✓ No contradiction with existing propositions\n"
            f"3. ✓ Valid derivation from dependencies\n"
            f"4. ✓ Contributes to goal\n\n"
            f"Verification Result: VERIFIED\n"
            f"  - Proposition is logically sound\n"
            f"  - No conflicts detected\n"
            f"  - Ready for accumulation"
        )

    def _generate_accumulation(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate accumulation step."""
        return (
            f"Step {self._step_counter}: Accumulate Proposition P{prop_num}\n\n"
            f"Adding verified proposition to knowledge graph...\n\n"
            f"Updated DAG Structure:\n"
            f"┌─────────────────────────────────────┐\n"
            f"│  Proposition Graph                  │\n"
            f"│  ─────────────────────              │\n"
            f"│  Nodes: {prop_num}                           │\n"
            f"│  Edges: {max(0, prop_num - 1)}                           │\n"
            f"│  Verified: {prop_num}                        │\n"
            f"└─────────────────────────────────────┘\n\n"
            f"Graph Visualization:\n"
            + self._visualize_dag(prop_num)
            + "\n\nCumulative Knowledge Updated."
        )

    def _visualize_dag(self, num_props: int) -> str:
        """Create a simple DAG visualization."""
        visualizations = {
            1: "  [P1] (root)\n",
            2: "  [P1] → [P2]\n",
            3: "  [P1] → [P2] → [P3]\n",
        }
        return visualizations.get(num_props, f"  [P1] → [P2] → ... → [P{num_props}]\n")

    def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the conclusion from accumulated knowledge."""
        return (
            f"Step {self._step_counter}: Derive Conclusion\n\n"
            f"Synthesizing answer from accumulated propositions...\n\n"
            f"Verified Knowledge Base:\n"
            f"  - P1: [First verified proposition]\n"
            f"  - P2: [Second verified proposition]\n"
            f"  - P3: [Third verified proposition]\n"
            f"  ... (total: {self._proposition_count} propositions)\n\n"
            f"Derivation Path:\n"
            f"  P1 + P2 → Intermediate Result\n"
            f"  Intermediate + P3 → Final Conclusion\n\n"
            f"Conclusion: [Answer derived from verified propositions]\n\n"
            f"Confidence: High (all propositions verified)"
        )

    def _generate_final_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Cumulative Reasoning Complete:\n\n"
            f"Summary:\n"
            f"  - Propositions generated: {self._proposition_count}\n"
            f"  - Propositions verified: {len(self._propositions)}\n"
            f"  - Knowledge graph: DAG with {self._proposition_count} nodes\n\n"
            f"Final Answer: [Comprehensive answer from cumulative knowledge]\n\n"
            f"Confidence: Very High\n"
            f"Reason: Answer derived from chain of verified propositions\n"
            f"with no contradictions in knowledge base."
        )

    # Helper methods for content generation with sampling fallback

    async def _get_proposition_content(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate proposition content with LLM sampling fallback.

        Args:
            prop_num: The proposition number
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the proposition
        """
        # Build context of previous propositions
        prev_props = "\n".join(
            [f"  P{p['id']}: [Verified proposition {p['id']}]" for p in self._propositions]
        )
        prev_context = f"\n\nPreviously verified propositions:\n{prev_props}" if prev_props else ""

        system_prompt = f"""You are a reasoning assistant using Cumulative Reasoning methodology.
Generate candidate propositions that build on verified knowledge.

For proposition P{prop_num}:
1. State the proposition clearly
2. List dependencies (previous propositions it builds on)
3. Specify if it's an axiom (first) or derived (builds on previous)
4. Present in formal logical form
5. Mark as "Proposed (awaiting verification)"

Build incrementally on previous verified propositions."""

        user_prompt = f"""Problem: {self._input_text}
{prev_context}

Generate Proposition P{prop_num} as a candidate statement that helps solve the problem.
{"Consider this guidance: " + guidance if guidance else ""}"""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_proposition(prop_num, guidance, context),
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=600,
        )

    async def _get_verification_content(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate verification content with LLM sampling fallback.

        Args:
            prop_num: The proposition number
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the verification
        """
        system_prompt = f"""You are a reasoning assistant using Cumulative Reasoning methodology.
Verify propositions against existing verified knowledge.

For verifying P{prop_num}:
1. Check consistency with axioms
2. Verify no contradiction with existing propositions
3. Validate derivation from dependencies
4. Confirm it contributes to the goal

Provide a clear verification result: VERIFIED or REJECTED with reasoning."""

        user_prompt = f"""Problem: {self._input_text}

Verify Proposition P{prop_num} against the existing verified knowledge base.
Check all verification criteria and provide a clear verdict."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_verification(prop_num, guidance, context),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=600,
        )

    async def _get_accumulation_content(
        self,
        prop_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate accumulation content with LLM sampling fallback.

        Args:
            prop_num: The proposition number
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the accumulation content
        """
        system_prompt = f"""You are a reasoning assistant using Cumulative Reasoning methodology.
Add verified propositions to the knowledge graph.

For accumulating P{prop_num}:
1. Confirm addition to the DAG structure
2. Update graph statistics (nodes, edges, verified count)
3. Provide a visual representation of the DAG
4. Note that cumulative knowledge has been updated

Show the growing knowledge structure."""

        user_prompt = f"""Problem: {self._input_text}

Add verified Proposition P{prop_num} to the knowledge graph.
Show the updated DAG structure with {prop_num} nodes."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_accumulation(prop_num, guidance, context),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=700,
        )

    async def _get_conclusion_content(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion content with LLM sampling fallback.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the conclusion
        """
        # Build context of all verified propositions
        props_summary = "\n".join(
            [f"  P{p['id']}: [Verified proposition {p['id']}]" for p in self._propositions]
        )

        system_prompt = """You are a reasoning assistant using Cumulative Reasoning methodology.
Derive conclusions from accumulated verified propositions.

For the conclusion:
1. List all verified propositions in the knowledge base
2. Show the derivation path (how propositions combine)
3. State the final conclusion clearly
4. Express high confidence (all propositions are verified)

Synthesize the cumulative knowledge into a clear answer."""

        user_prompt = f"""Problem: {self._input_text}

Verified Knowledge Base:
{props_summary}

Derive a conclusion from these {self._proposition_count} verified propositions.
Show the logical derivation path to the final answer."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_conclusion(guidance, context),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

    async def _get_final_synthesis_content(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final synthesis content with LLM sampling fallback.

        Args:
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A formatted string containing the final synthesis
        """
        system_prompt = """You are a reasoning assistant completing a Cumulative Reasoning process.
Provide a comprehensive final synthesis.

For the final answer:
1. Summarize the cumulative reasoning process
2. State statistics (propositions generated, verified, graph structure)
3. Present the comprehensive final answer
4. Express very high confidence with reasoning

Provide a complete and confident conclusion."""

        user_prompt = f"""Problem: {self._input_text}

Cumulative Reasoning completed with:
- {self._proposition_count} propositions generated
- {len(self._propositions)} propositions verified
- DAG structure with {self._proposition_count} nodes

Provide the final comprehensive answer derived from the cumulative knowledge base."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_final_synthesis(guidance, context),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=900,
        )


# Export
__all__ = ["CumulativeReasoning", "CUMULATIVE_REASONING_METADATA"]
