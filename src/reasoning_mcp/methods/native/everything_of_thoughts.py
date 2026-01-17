"""Everything of Thoughts (EoT) reasoning method.

This module implements Everything of Thoughts (Ding et al. 2023), a meta-framework
that dynamically selects and switches between different thought structures
(Chain, Tree, Graph) based on problem characteristics. EoT adapts its reasoning
topology to match the problem's needs.

Key phases:
1. Analyze: Understand problem structure and complexity
2. Select: Choose appropriate thought topology (Chain/Tree/Graph)
3. Execute: Apply selected structure dynamically
4. Integrate: Combine insights across structure transitions
5. Conclude: Synthesize final answer

Reference: Ding et al. (2023) - "Everything of Thoughts: Defying the Law of
Penrose Triangle for Thought Generation"
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


# Metadata for Everything of Thoughts method
EVERYTHING_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.EVERYTHING_OF_THOUGHTS,
    name="Everything of Thoughts",
    description="Meta-framework that dynamically switches between Chain, Tree, and "
    "Graph thought structures based on problem needs. Adapts topology through "
    "analyze → select → execute → integrate → conclude phases.",
    category=MethodCategory.HOLISTIC,
    tags=frozenset(
        {
            "meta-framework",
            "adaptive",
            "topology",
            "chain",
            "tree",
            "graph",
            "dynamic",
            "hybrid",
        }
    ),
    complexity=9,  # Very high complexity
    supports_branching=True,  # All topologies supported
    supports_revision=True,  # Can switch structures
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: analyze + select + execute + integrate + conclude
    max_thoughts=20,  # Complex multi-topology reasoning
    avg_tokens_per_thought=400,  # Meta-reasoning is verbose
    best_for=(
        "complex multi-faceted problems",
        "problems with unclear structure",
        "adaptive reasoning",
        "hybrid problem solving",
        "research problems",
        "novel challenges",
        "cross-domain reasoning",
        "strategic planning",
    ),
    not_recommended_for=(
        "simple queries",
        "well-structured problems",
        "time-critical tasks",
        "problems with clear solution paths",
    ),
)

logger = structlog.get_logger(__name__)


class EverythingOfThoughts(ReasoningMethodBase):
    """Everything of Thoughts meta-framework implementation.

    This class implements the EoT pattern:
    1. Analyze: Assess problem structure and requirements
    2. Select: Choose Chain, Tree, or Graph topology
    3. Execute: Apply selected reasoning structure
    4. Integrate: Combine insights, possibly switch structures
    5. Conclude: Synthesize final comprehensive answer

    Key characteristics:
    - Meta-level reasoning
    - Dynamic structure selection
    - Topology switching
    - Very high complexity (9)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = EverythingOfThoughts()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Design a sustainable city infrastructure"
        ... )
        >>> print(result.content)  # Problem analysis
    """

    # Available topologies
    TOPOLOGIES = ["chain", "tree", "graph"]

    # Enable LLM sampling
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Everything of Thoughts method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "analyze"
        self._selected_topology: str = ""
        self._topology_history: list[str] = []
        self._execution_count = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.EVERYTHING_OF_THOUGHTS

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return EVERYTHING_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return EVERYTHING_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Everything of Thoughts method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "analyze"
        self._selected_topology = ""
        self._topology_history = []
        self._execution_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Everything of Thoughts method.

        Analyzes the problem to understand its structure.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional execution context for sampling

        Returns:
            A ThoughtNode representing the analysis phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Everything of Thoughts method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "analyze"
        self._selected_topology = ""
        self._topology_history = []
        self._execution_count = 0

        # Generate analysis content
        content = await self._generate_analysis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.EVERYTHING_OF_THOUGHTS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "everything_of_thoughts",
                "phase": self._current_phase,
                "selected_topology": self._selected_topology,
                "topology_history": self._topology_history.copy(),
                "execution_count": self._execution_count,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.EVERYTHING_OF_THOUGHTS

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

        Implements the EoT phase progression:
        - After analyze: select topology
        - After select: execute with chosen structure
        - During execute: continue or switch topology
        - After sufficient execution: integrate
        - After integrate: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the EoT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Everything of Thoughts method must be initialized before continuation"
            )

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "analyze")

        if prev_phase == "analyze":
            # Select topology
            self._current_phase = "select"
            thought_type = ThoughtType.HYPOTHESIS
            content = await self._generate_selection(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "select":
            # Start execution
            self._current_phase = "execute"
            self._execution_count = 1
            thought_type = ThoughtType.REASONING
            content = await self._generate_execution(self._execution_count, guidance, context)
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "execute":
            self._execution_count += 1
            # Check if we should switch topology or integrate
            should_switch = self._execution_count == 3 and len(self._topology_history) < 2
            should_integrate = self._execution_count >= 4 or len(self._topology_history) >= 2

            if should_integrate:
                # Integrate results
                self._current_phase = "integrate"
                thought_type = ThoughtType.SYNTHESIS
                content = await self._generate_integration(guidance, context)
                confidence = 0.85
                quality_score = 0.85
            elif should_switch:
                # Switch topology
                self._current_phase = "select"
                thought_type = ThoughtType.HYPOTHESIS
                content = self._generate_topology_switch(guidance, context)
                confidence = 0.75
                quality_score = 0.75
            else:
                # Continue execution
                thought_type = ThoughtType.REASONING
                content = await self._generate_execution(self._execution_count, guidance, context)
                confidence = 0.75 + (0.02 * self._execution_count)
                quality_score = 0.75

        elif prev_phase == "integrate":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = await self._generate_conclusion(guidance, context)
            confidence = 0.9
            quality_score = 0.9

        elif prev_phase == "conclude":
            # Final synthesis
            self._current_phase = "done"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._generate_final_synthesis(guidance, context)
            confidence = 0.95
            quality_score = 0.95

        else:
            # Fallback
            self._current_phase = "integrate"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._generate_integration(guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.EVERYTHING_OF_THOUGHTS,
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
                "reasoning_type": "everything_of_thoughts",
                "selected_topology": self._selected_topology,
                "topology_history": self._topology_history.copy(),
                "execution_count": self._execution_count,
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    async def _generate_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the analysis phase content."""
        prompt = (
            f"Analyze this problem for the Everything of Thoughts framework:\n\n"
            f"Problem: {input_text}\n\n"
            f"Assess the problem's structural characteristics:\n"
            f"1. Linearity (sequential vs branching)\n"
            f"2. Branch potential (multiple approaches)\n"
            f"3. Interdependencies (connections between parts)\n"
            f"4. Overall complexity\n\n"
            f"For each criterion, score as Low/Medium/High and explain what topology "
            f"(Chain/Tree/Graph) would be best suited."
        )
        system_prompt = (
            "You are analyzing a problem to determine the optimal thought topology "
            "(Chain, Tree, or Graph) for the Everything of Thoughts framework. "
            "Provide a structured assessment of problem characteristics."
        )

        # Capture step_counter for closure
        step_counter = self._step_counter

        def fallback() -> str:
            return (
                f"Step {step_counter}: Problem Analysis (Everything of Thoughts)\n\n"
                f"Problem: {input_text}\n\n"
                f"Analyzing Problem Characteristics...\n\n"
                f"Structure Assessment:\n"
                f"┌─────────────────────────────────────────────┐\n"
                f"│ Criterion            │ Score   │ Implication │\n"
                f"├─────────────────────────────────────────────┤\n"
                f"│ Linearity           │ Medium  │ Chain OK    │\n"
                f"│ Branch Potential    │ High    │ Tree useful │\n"
                f"│ Interdependencies   │ High    │ Graph best  │\n"
                f"│ Complexity          │ High    │ Hybrid?     │\n"
                f"└─────────────────────────────────────────────┘\n\n"
                f"Available Topologies:\n"
                f"  • Chain: Linear, sequential reasoning\n"
                f"  • Tree: Branching exploration with backtracking\n"
                f"  • Graph: Interconnected thoughts with cycles\n\n"
                f"Ready to select optimal topology."
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "Structure Assessment:" not in result:
            return (
                f"Step {self._step_counter}: Problem Analysis (Everything of Thoughts)\n\n"
                f"Problem: {input_text}\n\n"
                f"{result}\n\n"
                f"Available Topologies:\n"
                f"  • Chain: Linear, sequential reasoning\n"
                f"  • Tree: Branching exploration with backtracking\n"
                f"  • Graph: Interconnected thoughts with cycles\n\n"
                f"Ready to select optimal topology."
            )
        return result

    async def _generate_selection(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate topology selection."""
        prompt = (
            f"Based on the previous problem analysis, select the optimal topology "
            f"(Chain, Tree, or Graph) for reasoning:\n\n"
            f"Available options:\n"
            f"1. Chain: Linear, sequential reasoning - best for straightforward problems\n"
            f"2. Tree: Branching exploration - best for multiple alternatives\n"
            f"3. Graph: Interconnected thoughts - best for interdependent components\n\n"
            f"Choose ONE topology and provide:\n"
            f"1. Your selection\n"
            f"2. Rating for each option (1-5 stars)\n"
            f"3. Rationale for your choice\n\n"
            f"{f'Guidance: {guidance}' if guidance else ''}"
        )
        system_prompt = (
            "You are selecting the optimal thought topology for the Everything of Thoughts "
            "framework. Choose Chain, Tree, or Graph based on problem characteristics. "
            "Be decisive and justify your selection."
        )

        # Capture step_counter for closure
        step_counter = self._step_counter

        def fallback() -> str:
            return "__FALLBACK__"

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        if result == "__FALLBACK__":
            # Fallback heuristic
            self._selected_topology = "tree"  # Example selection
            self._topology_history.append(self._selected_topology)

            return (
                f"Step {step_counter}: Topology Selection\n\n"
                f"Based on problem analysis, selecting topology...\n\n"
                f"Topology Comparison for This Problem:\n"
                f"  Chain: ★★☆☆☆ (too linear for this problem)\n"
                f"  Tree:  ★★★★☆ (good for exploring alternatives)\n"
                f"  Graph: ★★★☆☆ (may add unnecessary complexity)\n\n"
                f"Selected Topology: {self._selected_topology.upper()}\n\n"
                f"Rationale:\n"
                f"  - Problem has multiple valid approaches\n"
                f"  - Need to explore and compare alternatives\n"
                f"  - Tree structure allows systematic exploration\n"
                f"  - Can switch to Graph if interdependencies emerge\n\n"
                f"Initiating {self._selected_topology}-based reasoning..."
            )

        # Extract topology from result (simple heuristic)
        result_lower = result.lower()
        if "graph" in result_lower and "graph" not in result_lower.split("chain"):
            self._selected_topology = "graph"
        elif "tree" in result_lower:
            self._selected_topology = "tree"
        else:
            self._selected_topology = "chain"

        self._topology_history.append(self._selected_topology)

        return (
            f"Step {self._step_counter}: Topology Selection\n\n"
            f"{result}\n\n"
            f"Selected Topology: {self._selected_topology.upper()}\n\n"
            f"Initiating {self._selected_topology}-based reasoning..."
        )

    def _generate_topology_switch(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate topology switch content."""
        old_topology = self._selected_topology
        # Switch to a different topology
        new_topology = "graph" if old_topology == "tree" else "tree"
        self._selected_topology = new_topology
        self._topology_history.append(new_topology)

        return (
            f"Step {self._step_counter}: Topology Switch\n\n"
            f"Current reasoning suggests topology change needed...\n\n"
            f"Switch: {old_topology.upper()} → {new_topology.upper()}\n\n"
            f"Reason for Switch:\n"
            f"  - {old_topology.title()} structure revealed limitations\n"
            f"  - Problem aspects require {new_topology} capabilities\n"
            f"  - Discovered interdependencies between branches\n\n"
            f"Preserving Insights from Previous Topology:\n"
            f"  - Carrying forward key findings\n"
            f"  - Maintaining established conclusions\n"
            f"  - Integrating into new structure\n\n"
            f"Continuing with {new_topology.upper()} topology..."
        )

    async def _generate_execution(
        self,
        exec_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate execution step with current topology."""
        topology = self._selected_topology

        prompt = (
            f"Execute reasoning step {exec_num} using {topology.upper()} topology:\n\n"
            f"Topology characteristics:\n"
        )
        if topology == "chain":
            prompt += "- Linear, sequential progression\n- Build on previous step\n"
        elif topology == "tree":
            prompt += (
                "- Explore multiple branches/alternatives\n- Compare different approaches\n"
            )
        else:  # graph
            prompt += (
                "- Consider interconnections\n- Explore relationships between components\n"
            )

        prompt += f"\n{f'Guidance: {guidance}' if guidance else ''}\n"
        prompt += "\nProvide key insights from this reasoning step."

        system_prompt = (
            f"You are executing a reasoning step using {topology} topology in the "
            f"Everything of Thoughts framework. Apply {topology}-based thinking to "
            f"develop insights toward solving the problem."
        )

        # Calculate structure for display
        if topology == "chain":
            structure = f"→ Step {exec_num} → "
            description = "linear progression"
        elif topology == "tree":
            structure = f"├── Branch {exec_num}a\n└── Branch {exec_num}b"
            description = "branching exploration"
        else:  # graph
            structure = f"Node {exec_num} ↔ [Connected to multiple nodes]"
            description = "interconnected reasoning"

        # Capture values for closure
        step_counter = self._step_counter

        def fallback() -> str:
            return (
                f"Step {step_counter}: Execution ({topology.upper()} Topology) "
                f"- Iteration {exec_num}\n\n"
                f"Applying {topology}-based reasoning ({description})...\n\n"
                f"Structure:\n"
                f"{structure}\n\n"
                f"Reasoning:\n"
                f"  - Processing within {topology} framework\n"
                f"  - Exploring {'alternatives' if topology == 'tree' else 'connections'}\n"
                f"  - Building toward solution\n\n"
                f"Insight from this step:\n"
                f"  [Key insight discovered through {topology} reasoning]\n\n"
                f"Progress: {min(exec_num * 25, 100)}% of {topology} exploration"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "Processing within" not in result:
            return (
                f"Step {self._step_counter}: Execution ({topology.upper()} Topology) "
                f"- Iteration {exec_num}\n\n"
                f"Applying {topology}-based reasoning ({description})...\n\n"
                f"Structure:\n"
                f"{structure}\n\n"
                f"{result}\n\n"
                f"Progress: {min(exec_num * 25, 100)}% of {topology} exploration"
            )
        return result

    async def _generate_integration(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate integration of multi-topology insights."""
        prompt = (
            f"Integrate insights from multiple reasoning topologies:\n\n"
            f"Topologies used: {', '.join(self._topology_history)}\n\n"
            f"For each topology, identify its key contribution:\n"
        )
        for i, t in enumerate(self._topology_history, 1):
            prompt += f"{i}. {t.upper()}: What unique insights did it provide?\n"

        prompt += (
            f"\nSynthesize these insights into a comprehensive understanding.\n"
            f"{f'Guidance: {guidance}' if guidance else ''}"
        )

        system_prompt = (
            "You are integrating insights from multiple thought topologies in the "
            "Everything of Thoughts framework. Identify what each topology contributed "
            "and synthesize a comprehensive understanding."
        )

        # Capture values for closure
        step_counter = self._step_counter
        topology_history = self._topology_history.copy()

        def fallback() -> str:
            return (
                f"Step {step_counter}: Multi-Topology Integration\n\n"
                f"Integrating insights across {len(topology_history)} topologies...\n\n"
                f"Topology Journey:\n"
                + "\n".join(f"  {i + 1}. {t.upper()}" for i, t in enumerate(topology_history))
                + "\n\n"
                "Cross-Topology Synthesis:\n"
                "┌─────────────────────────────────────────────┐\n"
                "│ Topology  │ Key Contribution                │\n"
                "├─────────────────────────────────────────────┤\n"
                + "\n".join(
                    f"│ {t.upper():9} │ [Insight from {t}]            │"
                    for t in topology_history
                )
                + "\n└─────────────────────────────────────────────┘\n\n"
                "Integration Result:\n"
                "  Combined insights reveal comprehensive solution\n"
                "  that no single topology could achieve alone."
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "Cross-Topology Synthesis:" not in result:
            return (
                f"Step {self._step_counter}: Multi-Topology Integration\n\n"
                f"Integrating insights across {len(self._topology_history)} topologies...\n\n"
                f"Topology Journey: {' → '.join(t.upper() for t in self._topology_history)}\n\n"
                f"{result}\n\n"
                f"Integration Result:\n"
                f"  Combined insights reveal comprehensive solution\n"
                f"  that no single topology could achieve alone."
            )
        return result

    async def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the conclusion."""
        prompt = (
            f"Generate a comprehensive conclusion based on multi-topology reasoning:\n\n"
            f"Analysis performed using: {', '.join(self._topology_history)}\n"
            f"Execution steps completed: {self._execution_count}\n\n"
            f"Provide:\n"
            f"1. A comprehensive answer that leverages insights from all topologies\n"
            f"2. Meta-observations about how different topologies contributed\n"
            f"3. Why the hybrid approach was valuable\n\n"
            f"{f'Guidance: {guidance}' if guidance else ''}"
        )
        system_prompt = (
            "You are generating a comprehensive conclusion for the "
            "Everything of Thoughts framework. Synthesize insights from "
            "multiple topologies into a complete answer."
        )

        # Capture values for closure
        step_counter = self._step_counter
        topology_history = self._topology_history.copy()

        def fallback() -> str:
            return (
                f"Step {step_counter}: Comprehensive Conclusion\n\n"
                f"Everything of Thoughts reasoning complete.\n\n"
                f"Multi-Topology Analysis Summary:\n"
                f"  - Analyzed problem structure\n"
                f"  - Explored with {len(topology_history)} topologies: "
                f"{', '.join(topology_history)}\n"
                f"  - Integrated cross-structure insights\n\n"
                f"Comprehensive Answer:\n"
                f"  [Answer that leverages insights from all topologies]\n\n"
                f"Meta-Observations:\n"
                f"  - Problem benefited from topology switching\n"
                f"  - Each structure revealed unique aspects\n"
                f"  - Hybrid approach was essential"
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "Comprehensive Answer:" not in result:
            return (
                f"Step {self._step_counter}: Comprehensive Conclusion\n\n"
                f"Everything of Thoughts reasoning complete.\n\n"
                f"Multi-Topology Analysis Summary:\n"
                f"  - Analyzed problem structure\n"
                f"  - Explored with {len(self._topology_history)} topologies: "
                f"{', '.join(self._topology_history)}\n"
                f"  - Integrated cross-structure insights\n\n"
                f"{result}"
            )
        return result

    async def _generate_final_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final synthesis."""
        prompt = (
            f"Provide the final comprehensive answer:\n\n"
            f"Summary of analysis:\n"
            f"  - Topologies used: {', '.join(self._topology_history)}\n"
            f"  - Execution steps: {self._execution_count}\n"
            f"  - Cross-topology integration completed\n\n"
            f"Deliver the final answer that integrates all insights from the "
            f"multi-topology analysis. Explain why this comprehensive approach "
            f"yielded a robust solution.\n\n"
            f"{f'Guidance: {guidance}' if guidance else ''}"
        )
        system_prompt = (
            "You are providing the final answer after completing Everything of Thoughts "
            "analysis. Synthesize all insights into a comprehensive, confident solution."
        )

        # Capture values for closure
        step_counter = self._step_counter
        topology_history = self._topology_history.copy()
        execution_count = self._execution_count

        def fallback() -> str:
            return (
                f"Step {step_counter}: Final Answer\n\n"
                f"Everything of Thoughts Analysis Complete:\n\n"
                f"Summary:\n"
                f"  - Problem analyzed for structural properties\n"
                f"  - Topologies used: {', '.join(topology_history)}\n"
                f"  - Execution steps: {execution_count}\n"
                f"  - Cross-topology integration performed\n\n"
                f"Final Answer: [Comprehensive solution integrating all insights]\n\n"
                f"Confidence: Very High\n"
                f"Reason: Problem attacked from multiple structural perspectives,\n"
                f"ensuring thorough exploration and robust conclusions."
            )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # If we got a sampled result (not the fallback), format it
        if "Final Answer: [Comprehensive solution" not in result:
            return (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Everything of Thoughts Analysis Complete:\n\n"
                f"Summary:\n"
                f"  - Problem analyzed for structural properties\n"
                f"  - Topologies used: {', '.join(self._topology_history)}\n"
                f"  - Execution steps: {self._execution_count}\n"
                f"  - Cross-topology integration performed\n\n"
                f"{result}\n\n"
                f"Confidence: Very High\n"
                f"Reason: Problem attacked from multiple structural perspectives,\n"
                f"ensuring thorough exploration and robust conclusions."
            )
        return result


# Export
__all__ = ["EverythingOfThoughts", "EVERYTHING_OF_THOUGHTS_METADATA"]
