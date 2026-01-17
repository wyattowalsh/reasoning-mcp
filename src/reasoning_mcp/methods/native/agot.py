"""Adaptive Graph of Thoughts (AGoT) reasoning method.

This module implements Adaptive Graph of Thoughts, a dynamic graph-based reasoning
approach that adapts its structure during reasoning. Unlike static graph methods,
AGoT continuously restructures the graph by adding/removing nodes and modifying edges
based on reasoning progress and node confidence.

Key characteristics:
- Category: ADVANCED
- Complexity: 9 (very high complexity)
- Dynamic graph adaptation during reasoning
- Node confidence scoring and graph restructuring
- Information propagation through adaptive graph structure
- Phases: initialize_graph → adapt_structure → propagate → synthesize → conclude
- Tracks: nodes_added, nodes_removed, edges_modified, graph_density, confidence_evolution

Reference: 2025 research on adaptive graph structures for reasoning
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
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for Adaptive Graph of Thoughts method
AGOT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.AGOT,
    name="Adaptive Graph of Thoughts (AGoT)",
    description=(
        "Dynamically adapts graph structure during reasoning. Continuously restructures "
        "by adding/removing nodes and modifying edges based on confidence scores and "
        "reasoning progress. Features node confidence tracking, information propagation, "
        "and graph synthesis."
    ),
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "adaptive",
            "graph-based",
            "dynamic-structure",
            "confidence-driven",
            "propagation",
            "restructuring",
            "advanced",
            "synthesis",
            "iterative",
            "evolutionary",
        }
    ),
    complexity=9,  # Very high complexity - dynamic graph adaptation
    supports_branching=True,  # Adaptive graph supports branching
    supports_revision=True,  # Can revise through node removal/restructuring
    requires_context=False,
    min_thoughts=5,  # Initialize + adapt + propagate + synthesize + conclude
    max_thoughts=40,  # Multiple adaptation cycles
    avg_tokens_per_thought=550,  # Moderate to high - graph structure descriptions
    best_for=(
        "complex multi-faceted problems",
        "evolving problem spaces",
        "interconnected reasoning",
        "hypothesis refinement",
        "knowledge synthesis",
        "adaptive exploration",
        "confidence-driven reasoning",
        "graph-based analysis",
    ),
    not_recommended_for=(
        "simple linear problems",
        "single-step tasks",
        "time-critical decisions",
        "well-defined sequential processes",
        "problems requiring fixed structure",
    ),
)


class GraphNode:
    """Internal node representation for Adaptive Graph of Thoughts.

    This class represents a node in the adaptive graph with confidence scoring,
    connections, and propagation state.

    Attributes:
        id: Unique node identifier
        thought: Associated ThoughtNode
        confidence: Node confidence score (0.0-1.0)
        edges_to: Dictionary mapping node_id to edge weight (outgoing)
        edges_from: Dictionary mapping node_id to edge weight (incoming)
        visited: Whether node has been visited during propagation
        activation: Current activation level from propagation
        metadata: Additional node metadata
    """

    def __init__(
        self,
        node_id: str,
        thought: ThoughtNode,
        confidence: float = 0.5,
    ) -> None:
        """Initialize a graph node.

        Args:
            node_id: Unique identifier for this node
            thought: Associated ThoughtNode
            confidence: Initial confidence score
        """
        self.id = node_id
        self.thought = thought
        self.confidence = confidence
        self.edges_to: dict[str, float] = {}  # outgoing edges
        self.edges_from: dict[str, float] = {}  # incoming edges
        self.visited = False
        self.activation = 0.0
        self.metadata: dict[str, Any] = {}

    def add_edge_to(self, target_id: str, weight: float = 1.0) -> None:
        """Add outgoing edge to target node."""
        self.edges_to[target_id] = weight

    def add_edge_from(self, source_id: str, weight: float = 1.0) -> None:
        """Add incoming edge from source node."""
        self.edges_from[source_id] = weight

    def remove_edge_to(self, target_id: str) -> None:
        """Remove outgoing edge to target node."""
        self.edges_to.pop(target_id, None)

    def remove_edge_from(self, source_id: str) -> None:
        """Remove incoming edge from source node."""
        self.edges_from.pop(source_id, None)

    @property
    def degree(self) -> int:
        """Total degree (in + out)."""
        return len(self.edges_to) + len(self.edges_from)

    @property
    def out_degree(self) -> int:
        """Outgoing degree."""
        return len(self.edges_to)

    @property
    def in_degree(self) -> int:
        """Incoming degree."""
        return len(self.edges_from)


class AGoT(ReasoningMethodBase):
    """Adaptive Graph of Thoughts reasoning method implementation.

    This class implements a dynamic graph-based reasoning approach that adapts
    its structure during the reasoning process. The graph evolves through:
    1. Initialize: Create initial problem decomposition graph
    2. Adapt: Add/remove nodes and edges based on confidence and progress
    3. Propagate: Spread information through the graph structure
    4. Synthesize: Combine high-confidence nodes into insights
    5. Conclude: Generate final answer from graph synthesis

    The method tracks:
    - Node confidence scores and activation levels
    - Dynamic graph structure (nodes added/removed, edges modified)
    - Information propagation patterns
    - Graph density and connectivity metrics
    - Confidence evolution over adaptation cycles

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = AGoT()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Analyze the relationship between AI safety and alignment"
        ... )

        With custom parameters:
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Complex problem analysis",
        ...     context={
        ...         "max_nodes": 20,
        ...         "confidence_threshold": 0.6,
        ...         "adaptation_cycles": 4
        ...     }
        ... )

        Continue reasoning:
        >>> continuation = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Focus on causal relationships"
        ... )
    """

    # Default configuration
    DEFAULT_MAX_NODES = 15
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_ADAPTATION_CYCLES = 3
    DEFAULT_PROPAGATION_ITERATIONS = 3
    MIN_NODE_DEGREE = 1  # Minimum connections to keep node

    def __init__(self) -> None:
        """Initialize the Adaptive Graph of Thoughts method."""
        self._initialized = False
        self._step_counter = 0
        self._graph_nodes: dict[str, GraphNode] = {}
        self._current_phase: str = (
            "initialize"  # initialize, adapt, propagate, synthesize, conclude
        )
        self._adaptation_cycle = 0
        self._nodes_added = 0
        self._nodes_removed = 0
        self._edges_added = 0
        self._edges_removed = 0
        self._confidence_history: list[float] = []
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.AGOT

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return AGOT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return AGOT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares AGoT for execution by resetting state,
        clearing the graph, and preparing for a new reasoning session.

        Examples:
            >>> method = AGoT()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert len(method._graph_nodes) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._graph_nodes = {}
        self._current_phase = "initialize"
        self._adaptation_cycle = 0
        self._nodes_added = 0
        self._nodes_removed = 0
        self._edges_added = 0
        self._edges_removed = 0
        self._confidence_history = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Adaptive Graph of Thoughts method.

        This method creates the initial problem decomposition as an adaptive graph
        and prepares for dynamic restructuring based on reasoning progress.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context with parameters:
                - max_nodes: Maximum nodes in graph (default: 15)
                - confidence_threshold: Minimum confidence to keep node (default: 0.5)
                - adaptation_cycles: Number of adaptation cycles (default: 3)
                - propagation_iterations: Propagation iterations per cycle (default: 3)

        Returns:
            A ThoughtNode representing the initial graph setup

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = AGoT()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Analyze climate change solutions"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.method_id == MethodIdentifier.AGOT
        """
        if not self._initialized:
            raise RuntimeError(
                "Adaptive Graph of Thoughts method must be initialized before execution"
            )

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._graph_nodes = {}
        self._current_phase = "initialize"
        self._adaptation_cycle = 0
        self._nodes_added = 0
        self._nodes_removed = 0
        self._edges_added = 0
        self._edges_removed = 0
        self._confidence_history = []

        # Extract context parameters
        context = context or {}
        max_nodes = context.get("max_nodes", self.DEFAULT_MAX_NODES)
        confidence_threshold = context.get(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        adaptation_cycles = context.get("adaptation_cycles", self.DEFAULT_ADAPTATION_CYCLES)
        propagation_iterations = context.get(
            "propagation_iterations", self.DEFAULT_PROPAGATION_ITERATIONS
        )

        # Generate initial graph structure
        if use_sampling:
            initial_content = await self._sample_initial_graph(
                input_text, max_nodes, confidence_threshold
            )
        else:
            initial_content = self._generate_initial_graph(
                input_text, max_nodes, confidence_threshold
            )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.AGOT,
            content=initial_content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.5,
            metadata={
                "input": input_text,
                "context": context,
                "reasoning_type": "adaptive_graph_of_thoughts",
                "phase": self._current_phase,
                "adaptation_cycle": self._adaptation_cycle,
                "sampled": use_sampling,
                "max_nodes": max_nodes,
                "confidence_threshold": confidence_threshold,
                "adaptation_cycles": adaptation_cycles,
                "propagation_iterations": propagation_iterations,
                "graph_stats": {
                    "nodes": len(self._graph_nodes),
                    "nodes_added": self._nodes_added,
                    "nodes_removed": self._nodes_removed,
                    "edges_added": self._edges_added,
                    "edges_removed": self._edges_removed,
                },
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.AGOT

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

        This method implements the adaptive graph cycle logic:
        - Adapt: Restructure graph based on confidence scores
        - Propagate: Spread information through graph
        - Synthesize: Combine high-confidence nodes
        - Conclude: Generate final answer

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the adaptive graph reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = AGoT()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Complex problem")
            >>>
            >>> # Adaptation phase
            >>> adapted = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert adapted.type == ThoughtType.BRANCH
            >>> assert adapted.metadata["phase"] == "adapt"
            >>>
            >>> # Propagation phase
            >>> propagated = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=adapted
            ... )
            >>> assert propagated.type == ThoughtType.CONTINUATION
            >>> assert propagated.metadata["phase"] == "propagate"
        """
        if not self._initialized:
            raise RuntimeError(
                "Adaptive Graph of Thoughts method must be initialized before continuation"
            )

        # Increment step counter
        self._step_counter += 1

        # Get parameters from previous thought
        prev_metadata = previous_thought.metadata
        max_nodes = prev_metadata.get("max_nodes", self.DEFAULT_MAX_NODES)
        confidence_threshold = prev_metadata.get(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        adaptation_cycles = prev_metadata.get("adaptation_cycles", self.DEFAULT_ADAPTATION_CYCLES)
        propagation_iterations = prev_metadata.get(
            "propagation_iterations", self.DEFAULT_PROPAGATION_ITERATIONS
        )
        use_sampling = prev_metadata.get("sampled", False)

        # Determine next phase based on current phase
        prev_phase = prev_metadata.get("phase", "initialize")

        if prev_phase == "initialize":
            # After initialization: adapt the graph structure
            self._current_phase = "adapt"
            thought_type = ThoughtType.BRANCH
            if use_sampling:
                content = await self._sample_adaptation(
                    previous_thought, guidance, confidence_threshold, max_nodes
                )
            else:
                content = self._generate_adaptation(
                    previous_thought, guidance, confidence_threshold, max_nodes
                )
            confidence = 0.65
            quality_score = 0.6

        elif prev_phase == "adapt":
            # After adaptation: propagate information
            self._current_phase = "propagate"
            thought_type = ThoughtType.CONTINUATION
            if use_sampling:
                content = await self._sample_propagation(
                    previous_thought, guidance, propagation_iterations
                )
            else:
                content = self._generate_propagation(
                    previous_thought, guidance, propagation_iterations
                )
            confidence = 0.7
            quality_score = 0.65

        elif prev_phase == "propagate":
            # After propagation: check if we should adapt again or synthesize
            if self._adaptation_cycle < adaptation_cycles - 1:
                # Continue adaptation cycles
                self._adaptation_cycle += 1
                self._current_phase = "adapt"
                thought_type = ThoughtType.BRANCH
                if use_sampling:
                    content = await self._sample_adaptation(
                        previous_thought, guidance, confidence_threshold, max_nodes
                    )
                else:
                    content = self._generate_adaptation(
                        previous_thought, guidance, confidence_threshold, max_nodes
                    )
                confidence = 0.7 + (0.05 * self._adaptation_cycle)
                quality_score = 0.65 + (0.05 * self._adaptation_cycle)
            else:
                # Move to synthesis
                self._current_phase = "synthesize"
                thought_type = ThoughtType.SYNTHESIS
                if use_sampling:
                    content = await self._sample_synthesis(
                        previous_thought, guidance, confidence_threshold
                    )
                else:
                    content = self._generate_synthesis(
                        previous_thought, guidance, confidence_threshold
                    )
                confidence = 0.85
                quality_score = 0.8

        elif prev_phase == "synthesize":
            # After synthesis: conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance)
            else:
                content = self._generate_conclusion(previous_thought, guidance)
            confidence = 0.9
            quality_score = 0.85

        else:
            # Fallback: continue adapting
            self._current_phase = "adapt"
            thought_type = ThoughtType.CONTINUATION
            if use_sampling:
                content = await self._sample_adaptation(
                    previous_thought, guidance, confidence_threshold, max_nodes
                )
            else:
                content = self._generate_adaptation(
                    previous_thought, guidance, confidence_threshold, max_nodes
                )
            confidence = 0.7
            quality_score = 0.65

        # Calculate average confidence
        avg_confidence = (
            sum(self._confidence_history) / len(self._confidence_history)
            if self._confidence_history
            else 0.5
        )

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.AGOT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "adaptation_cycle": self._adaptation_cycle,
                "max_nodes": max_nodes,
                "confidence_threshold": confidence_threshold,
                "adaptation_cycles": adaptation_cycles,
                "propagation_iterations": propagation_iterations,
                "graph_stats": {
                    "nodes": len(self._graph_nodes),
                    "nodes_added": self._nodes_added,
                    "nodes_removed": self._nodes_removed,
                    "edges_added": self._edges_added,
                    "edges_removed": self._edges_removed,
                    "avg_confidence": avg_confidence,
                },
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "adaptive_graph_of_thoughts",
                "previous_phase": prev_phase,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For AGoT, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = AGoT()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    # =========================================================================
    # LLM Sampling Methods
    # =========================================================================

    async def _sample_initial_graph(
        self,
        input_text: str,
        max_nodes: int,
        confidence_threshold: float,
    ) -> str:
        """Generate initial graph using LLM sampling.

        Args:
            input_text: The problem to decompose
            max_nodes: Maximum number of nodes
            confidence_threshold: Minimum confidence threshold

        Returns:
            Content describing the initial graph structure
        """
        self._require_execution_context()

        system_prompt = """You are an Adaptive Graph of Thoughts reasoning assistant.
Analyze the given problem and create an initial graph decomposition.
Your response should:
1. Decompose the problem into 3-5 key aspects (nodes)
2. Identify relationships between aspects (edges)
3. Assign initial confidence scores to each aspect
4. Describe the graph structure and how it will adapt"""

        user_prompt = f"""Problem: {input_text}

Create an initial adaptive graph structure:
- Identify 3-5 key aspects of the problem as graph nodes
- Describe relationships (edges) between nodes
- Provide initial confidence scores (0.0-1.0) for each node
- Explain how this graph can dynamically adapt

Graph parameters:
- Maximum nodes: {max_nodes}
- Confidence threshold: {confidence_threshold}

Begin your adaptive graph initialization."""

        def fallback() -> str:
            return self._generate_initial_graph(input_text, max_nodes, confidence_threshold)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800,
        )
        # Still update internal graph state
        self._update_internal_graph_state(max_nodes, confidence_threshold)
        return result

    async def _sample_adaptation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        confidence_threshold: float,
        max_nodes: int,
    ) -> str:
        """Generate graph adaptation using LLM sampling.

        Args:
            previous_thought: Previous thought in the chain
            guidance: Optional guidance
            confidence_threshold: Minimum confidence to keep nodes
            max_nodes: Maximum allowed nodes

        Returns:
            Content describing the adaptation process
        """
        self._require_execution_context()

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        system_prompt = """You are an Adaptive Graph of Thoughts reasoning assistant.
Analyze the current graph state and perform dynamic restructuring.
Your response should:
1. Identify which nodes to remove (low confidence/connectivity)
2. Identify which nodes to add (promising directions)
3. Modify edge weights based on insights
4. Explain the rationale for each adaptation"""

        user_prompt = f"""Current Graph State:
- Nodes: {len(self._graph_nodes)}
- Adaptation cycle: {self._adaptation_cycle + 1}
- Confidence threshold: {confidence_threshold}
- Max nodes: {max_nodes}

Previous reasoning: {previous_thought.content[:300]}...{guidance_text}

Perform graph adaptation:
- Identify nodes to prune (confidence < {confidence_threshold})
- Identify new nodes to add for promising directions
- Adjust edge weights to strengthen valuable connections
- Explain how these changes improve the reasoning structure

Generate your adaptive graph restructuring."""

        def fallback() -> str:
            return self._generate_adaptation(
                previous_thought, guidance, confidence_threshold, max_nodes
            )

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800,
        )
        # Still update internal graph state
        self._update_graph_adaptation(confidence_threshold, max_nodes, previous_thought)
        return result

    async def _sample_propagation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        propagation_iterations: int,
    ) -> str:
        """Generate propagation using LLM sampling.

        Args:
            previous_thought: Previous thought
            guidance: Optional guidance
            propagation_iterations: Number of propagation iterations

        Returns:
            Content describing propagation process
        """
        self._require_execution_context()

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        system_prompt = """You are an Adaptive Graph of Thoughts reasoning assistant.
Analyze how information propagates through the adaptive graph structure.
Your response should:
1. Describe activation spreading through node connections
2. Show how high-confidence nodes amplify neighbors
3. Identify central nodes and key information paths
4. Explain emergent insights from propagation dynamics"""

        user_prompt = f"""Current Graph State:
- Nodes: {len(self._graph_nodes)}
- Propagation iterations: {propagation_iterations}

Previous adaptation: {previous_thought.content[:300]}...{guidance_text}

Perform information propagation:
- Spread activation through {propagation_iterations} iterations
- Show how high-confidence nodes influence neighbors
- Identify highly activated paths and central nodes
- Extract insights from propagation patterns
- Update confidence scores based on network support

Describe your propagation dynamics and insights."""

        def fallback() -> str:
            return self._generate_propagation(previous_thought, guidance, propagation_iterations)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800,
        )
        # Still update internal graph state
        self._update_propagation_state(propagation_iterations)
        return result

    async def _sample_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        confidence_threshold: float,
    ) -> str:
        """Generate synthesis using LLM sampling.

        Args:
            previous_thought: Previous thought
            guidance: Optional guidance
            confidence_threshold: Minimum confidence for synthesis

        Returns:
            Content describing synthesis
        """
        self._require_execution_context()

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        system_prompt = """You are an Adaptive Graph of Thoughts reasoning assistant.
Synthesize insights from high-confidence nodes in the adapted graph.
Your response should:
1. Combine information from nodes above confidence threshold
2. Identify cross-node patterns and relationships
3. Generate integrated insights from graph structure
4. Validate findings through graph evolution metrics"""

        high_conf_count = sum(
            1 for node in self._graph_nodes.values() if node.confidence >= confidence_threshold
        )

        user_prompt = f"""Current Graph State:
- Total nodes: {len(self._graph_nodes)}
- High-confidence nodes: {high_conf_count}
- Confidence threshold: {confidence_threshold}
- Adaptation cycles completed: {self._adaptation_cycle + 1}

Previous propagation: {previous_thought.content[:300]}...{guidance_text}

Synthesize graph insights:
- Combine information from {high_conf_count} high-confidence nodes
- Identify emergent patterns from graph structure
- Show cross-node relationships and dependencies
- Validate through graph evolution statistics

Generate your graph synthesis."""

        def fallback() -> str:
            return self._generate_synthesis(previous_thought, guidance, confidence_threshold)

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

    async def _sample_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> str:
        """Generate conclusion using LLM sampling.

        Args:
            previous_thought: Previous synthesis thought
            guidance: Optional guidance

        Returns:
            Content describing final conclusion
        """
        self._require_execution_context()

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        system_prompt = """You are an Adaptive Graph of Thoughts reasoning assistant.
Generate a final conclusion based on the adapted graph synthesis.
Your response should:
1. Provide a definitive answer based on graph insights
2. Trace the key nodes and connections that support the conclusion
3. Assess confidence based on graph evolution
4. Explain how adaptive restructuring led to the solution"""

        user_prompt = f"""Final Graph State:
- Nodes: {len(self._graph_nodes)}
- Adaptation cycles: {self._adaptation_cycle + 1}
- Total nodes added: {self._nodes_added}
- Total nodes removed: {self._nodes_removed}

Previous synthesis: {previous_thought.content[:300]}...{guidance_text}

Generate final conclusion:
- Provide definitive answer based on graph synthesis
- Trace key reasoning paths through the adapted structure
- Assess confidence from graph evolution metrics
- Explain how dynamic adaptation enabled the solution

Conclude your Adaptive Graph of Thoughts reasoning."""

        def fallback() -> str:
            return self._generate_conclusion(previous_thought, guidance)

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

    # =========================================================================
    # Internal Graph State Update Helpers
    # =========================================================================

    def _update_internal_graph_state(self, max_nodes: int, confidence_threshold: float) -> None:
        """Update internal graph state when using LLM sampling for initial graph.

        Args:
            max_nodes: Maximum number of nodes
            confidence_threshold: Confidence threshold
        """
        # Create initial decomposition nodes (same as _generate_initial_graph)
        initial_node_count = min(5, max_nodes)

        for i in range(initial_node_count):
            node_id = f"node_{i}"
            dummy_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.AGOT,
                content=f"Node {i}: Aspect of problem",
                step_number=0,
                depth=0,
            )
            confidence = 0.5 + (i * 0.05)
            graph_node = GraphNode(node_id, dummy_thought, confidence)
            self._graph_nodes[node_id] = graph_node
            self._nodes_added += 1
            self._confidence_history.append(confidence)

        # Create initial edges
        for i in range(initial_node_count - 1):
            source = self._graph_nodes[f"node_{i}"]
            target = self._graph_nodes[f"node_{i + 1}"]
            source.add_edge_to(target.id, weight=0.7)
            target.add_edge_from(source.id, weight=0.7)
            self._edges_added += 1

        # Add some cross-connections
        if initial_node_count >= 4:
            self._graph_nodes["node_0"].add_edge_to("node_2", weight=0.5)
            self._graph_nodes["node_2"].add_edge_from("node_0", weight=0.5)
            self._edges_added += 1

    def _update_graph_adaptation(
        self,
        confidence_threshold: float,
        max_nodes: int,
        previous_thought: ThoughtNode,
    ) -> None:
        """Update internal graph state during adaptation.

        Args:
            confidence_threshold: Minimum confidence to keep nodes
            max_nodes: Maximum allowed nodes
            previous_thought: Previous thought for context
        """
        # Simulate adaptation (same as _generate_adaptation)
        nodes_to_remove = [
            node_id
            for node_id, node in self._graph_nodes.items()
            if node.confidence < confidence_threshold and node.degree < self.MIN_NODE_DEGREE
        ]

        for node_id in nodes_to_remove[:2]:
            node = self._graph_nodes[node_id]
            for target_id in list(node.edges_to.keys()):
                if target_id in self._graph_nodes:
                    self._graph_nodes[target_id].remove_edge_from(node_id)
                self._edges_removed += 1
            for source_id in list(node.edges_from.keys()):
                if source_id in self._graph_nodes:
                    self._graph_nodes[source_id].remove_edge_to(node_id)
                self._edges_removed += 1
            del self._graph_nodes[node_id]
            self._nodes_removed += 1

        # Add new nodes
        if len(self._graph_nodes) < max_nodes:
            new_nodes = min(2, max_nodes - len(self._graph_nodes))
            for i in range(new_nodes):
                node_id = f"node_{self._nodes_added}"
                dummy_thought = ThoughtNode(
                    type=ThoughtType.HYPOTHESIS,
                    method_id=MethodIdentifier.AGOT,
                    content=f"Adaptive node {self._nodes_added}",
                    step_number=self._step_counter,
                    depth=previous_thought.depth,
                )
                confidence = 0.6 + (i * 0.1)
                graph_node = GraphNode(node_id, dummy_thought, confidence)
                self._graph_nodes[node_id] = graph_node
                self._nodes_added += 1
                self._confidence_history.append(confidence)

                high_conf_nodes = [
                    n
                    for n in self._graph_nodes.values()
                    if n.confidence >= confidence_threshold and n.id != node_id
                ][:2]
                for target_node in high_conf_nodes:
                    graph_node.add_edge_to(target_node.id, weight=0.6)
                    target_node.add_edge_from(graph_node.id, weight=0.6)
                    self._edges_added += 1

    def _update_propagation_state(self, propagation_iterations: int) -> None:
        """Update internal graph state during propagation.

        Args:
            propagation_iterations: Number of propagation iterations
        """
        # Simulate propagation (same as _generate_propagation)
        for _ in range(propagation_iterations):
            for node in self._graph_nodes.values():
                incoming_activation = sum(
                    self._graph_nodes[source_id].activation * weight
                    for source_id, weight in node.edges_from.items()
                    if source_id in self._graph_nodes
                )
                node.activation = min(1.0, node.confidence + (0.3 * incoming_activation))

        # Update confidence based on activation
        for node in self._graph_nodes.values():
            node.confidence = min(1.0, (node.confidence + node.activation) / 2)
            self._confidence_history.append(node.confidence)

    # =========================================================================
    # Graph Generation and Adaptation Methods
    # =========================================================================

    def _generate_initial_graph(
        self,
        input_text: str,
        max_nodes: int,
        confidence_threshold: float,
    ) -> str:
        """Generate the initial problem decomposition graph.

        This creates the starting graph structure with nodes representing
        different aspects of the problem and edges representing relationships.

        Args:
            input_text: The problem to decompose
            max_nodes: Maximum number of nodes
            confidence_threshold: Minimum confidence threshold

        Returns:
            Content describing the initial graph structure
        """
        # Create initial decomposition nodes
        initial_node_count = min(5, max_nodes)

        # Simulate creating graph nodes
        for i in range(initial_node_count):
            node_id = f"node_{i}"
            # Create a dummy ThoughtNode for graph structure
            dummy_thought = ThoughtNode(
                type=ThoughtType.HYPOTHESIS,
                method_id=MethodIdentifier.AGOT,
                content=f"Node {i}: Aspect of problem",
                step_number=0,
                depth=0,
            )
            confidence = 0.5 + (i * 0.05)  # Varying initial confidence
            graph_node = GraphNode(node_id, dummy_thought, confidence)
            self._graph_nodes[node_id] = graph_node
            self._nodes_added += 1
            self._confidence_history.append(confidence)

        # Create initial edges (simple chain + some cross-connections)
        for i in range(initial_node_count - 1):
            source = self._graph_nodes[f"node_{i}"]
            target = self._graph_nodes[f"node_{i + 1}"]
            source.add_edge_to(target.id, weight=0.7)
            target.add_edge_from(source.id, weight=0.7)
            self._edges_added += 1

        # Add some cross-connections for richer structure
        if initial_node_count >= 4:
            self._graph_nodes["node_0"].add_edge_to("node_2", weight=0.5)
            self._graph_nodes["node_2"].add_edge_from("node_0", weight=0.5)
            self._edges_added += 1

        return (
            f"Step {self._step_counter}: Initial Graph Construction (AGoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"Phase: INITIALIZE_GRAPH\n\n"
            f"Adaptive Graph of Thoughts decomposes the problem into an initial graph "
            f"structure that will dynamically adapt based on reasoning progress.\n\n"
            f"Initial Graph Structure:\n"
            f"- Nodes created: {initial_node_count}\n"
            f"- Initial edges: {self._edges_added}\n"
            f"- Confidence threshold: {confidence_threshold}\n"
            f"- Max nodes allowed: {max_nodes}\n\n"
            f"Node representations:\n"
            f"[LLM would generate specific problem aspects here]\n"
            f"• Node 0: Core problem definition\n"
            f"• Node 1: Primary constraint analysis\n"
            f"• Node 2: Solution space exploration\n"
            f"• Node 3: Dependency mapping\n"
            f"• Node 4: Verification criteria\n\n"
            f"Graph Properties:\n"
            f"- Average node degree: {self._edges_added * 2 / initial_node_count:.2f}\n"
            f"- Graph density: "
            f"{self._edges_added / (initial_node_count * (initial_node_count - 1) / 2):.2f}\n"
            f"- Ready for adaptive restructuring\n\n"
            f"Next: Graph will adapt structure based on confidence scores "
            f"and reasoning progress."
        )

    def _generate_adaptation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        confidence_threshold: float,
        max_nodes: int,
    ) -> str:
        """Generate adaptive graph restructuring.

        This simulates the graph adaptation process: adding high-value nodes,
        removing low-confidence nodes, and modifying edge weights.

        Args:
            previous_thought: Previous thought in the chain
            guidance: Optional guidance
            confidence_threshold: Minimum confidence to keep nodes
            max_nodes: Maximum allowed nodes

        Returns:
            Content describing the adaptation process
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        nodes_before = len(self._graph_nodes)
        edges_before = self._edges_added - self._edges_removed

        # Simulate adaptation: remove low-confidence nodes
        nodes_to_remove = [
            node_id
            for node_id, node in self._graph_nodes.items()
            if node.confidence < confidence_threshold and node.degree < self.MIN_NODE_DEGREE
        ]

        removed_count = 0
        for node_id in nodes_to_remove[:2]:  # Limit removal
            node = self._graph_nodes[node_id]
            # Remove edges
            for target_id in list(node.edges_to.keys()):
                if target_id in self._graph_nodes:
                    self._graph_nodes[target_id].remove_edge_from(node_id)
                self._edges_removed += 1
            for source_id in list(node.edges_from.keys()):
                if source_id in self._graph_nodes:
                    self._graph_nodes[source_id].remove_edge_to(node_id)
                self._edges_removed += 1
            del self._graph_nodes[node_id]
            self._nodes_removed += 1
            removed_count += 1

        # Simulate adding new high-value nodes
        added_count = 0
        if len(self._graph_nodes) < max_nodes:
            new_nodes = min(2, max_nodes - len(self._graph_nodes))
            for i in range(new_nodes):
                node_id = f"node_{self._nodes_added}"
                dummy_thought = ThoughtNode(
                    type=ThoughtType.HYPOTHESIS,
                    method_id=MethodIdentifier.AGOT,
                    content=f"Adaptive node {self._nodes_added}",
                    step_number=self._step_counter,
                    depth=previous_thought.depth,
                )
                confidence = 0.6 + (i * 0.1)
                graph_node = GraphNode(node_id, dummy_thought, confidence)
                self._graph_nodes[node_id] = graph_node
                self._nodes_added += 1
                self._confidence_history.append(confidence)
                added_count += 1

                # Connect to existing high-confidence nodes
                high_conf_nodes = [
                    n
                    for n in self._graph_nodes.values()
                    if n.confidence >= confidence_threshold and n.id != node_id
                ][:2]
                for target_node in high_conf_nodes:
                    graph_node.add_edge_to(target_node.id, weight=0.6)
                    target_node.add_edge_from(graph_node.id, weight=0.6)
                    self._edges_added += 1

        nodes_after = len(self._graph_nodes)
        edges_after = self._edges_added - self._edges_removed

        return (
            f"Step {self._step_counter}: Graph Adaptation (Cycle {self._adaptation_cycle + 1})\n\n"
            f"Phase: ADAPT_STRUCTURE\n\n"
            f"Dynamically restructuring graph based on node confidence and connectivity...\n\n"
            f"Adaptation Actions:\n"
            f"- Nodes removed (low confidence): {removed_count}\n"
            f"- Nodes added (high value): {added_count}\n"
            f"- Edges modified: {abs(edges_after - edges_before)}\n\n"
            f"Graph Evolution:\n"
            f"- Nodes: {nodes_before} → {nodes_after}\n"
            f"- Edges: {edges_before} → {edges_after}\n"
            f"- Current density: "
            f"{edges_after / max(1, (nodes_after * (nodes_after - 1) / 2)):.3f}\n\n"
            f"Restructuring Rationale:\n"
            f"[LLM would explain specific changes here]\n"
            f"• Removed nodes with confidence < {confidence_threshold} "
            f"and degree < {self.MIN_NODE_DEGREE}\n"
            f"• Added nodes exploring promising directions\n"
            f"• Strengthened connections between high-confidence nodes\n\n"
            f"Cumulative Statistics:\n"
            f"- Total nodes added: {self._nodes_added}\n"
            f"- Total nodes removed: {self._nodes_removed}\n"
            f"- Total edges added: {self._edges_added}\n"
            f"- Total edges removed: {self._edges_removed}\n"
            f"- Net nodes: {nodes_after}\n\n"
            f"Next: Propagate information through adapted graph structure.{guidance_text}"
        )

    def _generate_propagation(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        propagation_iterations: int,
    ) -> str:
        """Generate information propagation through the graph.

        This simulates spreading activation and information flow through
        the adaptive graph structure.

        Args:
            previous_thought: Previous thought
            guidance: Optional guidance
            propagation_iterations: Number of propagation iterations

        Returns:
            Content describing propagation process
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Simulate propagation: update activation levels
        for _ in range(propagation_iterations):
            for node in self._graph_nodes.values():
                # Activation = confidence + incoming weighted activations
                incoming_activation = sum(
                    self._graph_nodes[source_id].activation * weight
                    for source_id, weight in node.edges_from.items()
                    if source_id in self._graph_nodes
                )
                node.activation = min(1.0, node.confidence + (0.3 * incoming_activation))

        # Update confidence based on activation
        for node in self._graph_nodes.values():
            node.confidence = min(1.0, (node.confidence + node.activation) / 2)
            self._confidence_history.append(node.confidence)

        avg_activation = (
            sum(n.activation for n in self._graph_nodes.values()) / len(self._graph_nodes)
            if self._graph_nodes
            else 0.0
        )

        avg_confidence = (
            sum(n.confidence for n in self._graph_nodes.values()) / len(self._graph_nodes)
            if self._graph_nodes
            else 0.0
        )

        max_activation_node = max(
            self._graph_nodes.values(), key=lambda n: n.activation, default=None
        )

        # Get central nodes by degree
        central_nodes = [
            n.id
            for n in sorted(self._graph_nodes.values(), key=lambda x: x.degree, reverse=True)[:3]
        ]

        # Determine next action
        next_action = (
            "Continue adaptation cycles"
            if self._adaptation_cycle < self.DEFAULT_ADAPTATION_CYCLES - 1
            else "Synthesize high-confidence nodes"
        )

        max_activation_str = (
            f"{max_activation_node.activation:.3f} ({max_activation_node.id})"
            if max_activation_node
            else "N/A"
        )

        return (
            f"Step {self._step_counter}: Information Propagation\n\n"
            f"Phase: PROPAGATE\n\n"
            f"Spreading information through adaptive graph over "
            f"{propagation_iterations} iterations...\n\n"
            f"Propagation Results:\n"
            f"- Average node activation: {avg_activation:.3f}\n"
            f"- Average node confidence: {avg_confidence:.3f}\n"
            f"- Max activation: {max_activation_str}\n"
            f"- Graph nodes: {len(self._graph_nodes)}\n\n"
            f"Activation Dynamics:\n"
            f"[LLM would describe propagation patterns here]\n"
            f"• High-confidence nodes amplified neighbors\n"
            f"• Information flowed through strong connections\n"
            f"• Weak nodes received limited activation\n"
            f"• Confidence scores updated based on network support\n\n"
            f"Key Insights from Propagation:\n"
            f"[LLM would extract insights here]\n"
            f"• Central nodes: {central_nodes}\n"
            f"• Highly activated paths emerged\n"
            f"• Confidence convergence observed\n\n"
            f"Next: {next_action}.{guidance_text}"
        )

    def _generate_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        confidence_threshold: float,
    ) -> str:
        """Generate synthesis of high-confidence nodes.

        This combines information from nodes exceeding the confidence threshold
        into coherent insights.

        Args:
            previous_thought: Previous thought
            guidance: Optional guidance
            confidence_threshold: Minimum confidence for synthesis

        Returns:
            Content describing synthesis
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Identify high-confidence nodes for synthesis
        high_conf_nodes = [
            node for node in self._graph_nodes.values() if node.confidence >= confidence_threshold
        ]

        synthesis_count = len(high_conf_nodes)
        avg_confidence = (
            sum(n.confidence for n in high_conf_nodes) / synthesis_count if high_conf_nodes else 0.0
        )

        # Sort by confidence
        sorted_nodes = sorted(high_conf_nodes, key=lambda n: n.confidence, reverse=True)
        top_nodes = sorted_nodes[:5]

        # Format top nodes list
        top_nodes_list = chr(10).join(
            f"  {i + 1}. {node.id}: {node.confidence:.3f}" for i, node in enumerate(top_nodes)
        )

        return (
            f"Step {self._step_counter}: Graph Synthesis\n\n"
            f"Phase: SYNTHESIZE\n\n"
            f"Combining insights from {synthesis_count} high-confidence nodes "
            f"(threshold: {confidence_threshold})...\n\n"
            f"Synthesis Overview:\n"
            f"- Nodes in synthesis: {synthesis_count}\n"
            f"- Average confidence: {avg_confidence:.3f}\n"
            f"- Total adaptation cycles: {self._adaptation_cycle + 1}\n"
            f"- Final graph size: {len(self._graph_nodes)} nodes\n\n"
            f"Top Confidence Nodes:\n"
            f"{top_nodes_list}\n\n"
            f"Integrated Insights:\n"
            f"[LLM would synthesize node content here]\n"
            f"• Core findings from node network\n"
            f"• Cross-node relationships and patterns\n"
            f"• Emergent understanding from graph structure\n"
            f"• Validated through propagation dynamics\n\n"
            f"Graph Evolution Summary:\n"
            f"- Total nodes created: {self._nodes_added}\n"
            f"- Total nodes pruned: {self._nodes_removed}\n"
            f"- Total edges created: {self._edges_added}\n"
            f"- Total edges removed: {self._edges_removed}\n"
            f"- Adaptation efficiency: {synthesis_count / max(1, self._nodes_added):.2%}\n\n"
            f"Next: Generate final conclusion from synthesis.{guidance_text}"
        )

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
    ) -> str:
        """Generate final conclusion from graph synthesis.

        This creates the final answer based on the adapted graph structure
        and synthesized insights.

        Args:
            previous_thought: Previous synthesis thought
            guidance: Optional guidance

        Returns:
            Content describing final conclusion
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        final_avg_confidence = (
            sum(n.confidence for n in self._graph_nodes.values()) / len(self._graph_nodes)
            if self._graph_nodes
            else 0.0
        )

        # Calculate graph density
        num_nodes = len(self._graph_nodes)
        max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
        graph_density = (self._edges_added - self._edges_removed) / max(1, max_edges)

        return (
            f"Step {self._step_counter}: Final Conclusion (AGoT)\n\n"
            f"Phase: CONCLUDE\n\n"
            f"Adaptive Graph of Thoughts Complete\n\n"
            f"Final Answer:\n"
            f"[LLM would provide definitive answer here based on graph synthesis]\n\n"
            f"Reasoning Path:\n"
            f"[Trace key nodes and connections that led to conclusion]\n\n"
            f"Confidence Assessment:\n"
            f"- Final graph confidence: {final_avg_confidence:.3f}\n"
            f"- Based on {num_nodes} adapted nodes\n"
            f"- Through {self._adaptation_cycle + 1} adaptation cycles\n"
            f"- Confidence evolution tracked across "
            f"{len(self._confidence_history)} measurements\n\n"
            f"Graph Adaptation Statistics:\n"
            f"- Nodes added: {self._nodes_added}\n"
            f"- Nodes removed: {self._nodes_removed}\n"
            f"- Edges added: {self._edges_added}\n"
            f"- Edges removed: {self._edges_removed}\n"
            f"- Final graph density: {graph_density:.3f}\n\n"
            f"The solution emerged through dynamic graph adaptation, where the reasoning "
            f"structure evolved based on confidence scores and information propagation patterns. "
            f"This adaptive approach allowed the method to focus computational resources on "
            f"high-value reasoning paths while pruning unproductive directions.{guidance_text}"
        )


# Export metadata and class
__all__ = [
    "AGoT",
    "AGOT_METADATA",
    "GraphNode",
]
