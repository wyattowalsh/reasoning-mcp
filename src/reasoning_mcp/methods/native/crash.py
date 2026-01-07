"""CRASH reasoning method implementation.

This module implements the CRASH (Confidence-gated Reasoning with Automatic Strategy
Handoff) method, which monitors confidence during reasoning and automatically switches
strategies when confidence drops below a threshold.

CRASH enables:
- Adaptive strategy selection based on confidence
- Automatic fallback when current approach struggles
- Multiple strategy attempts on difficult problems
- Confidence tracking throughout reasoning process
- Resilient problem solving with strategy diversity
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode, ThoughtGraph

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for CRASH method
CRASH_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CRASH,
    name="CRASH",
    description="Confidence-gated Reasoning with Automatic Strategy Handoff for adaptive problem solving",
    category=MethodCategory.HOLISTIC,
    tags=frozenset({
        "adaptive",
        "confidence",
        "strategy-switching",
        "resilient",
        "monitoring",
        "fallback",
        "holistic",
    }),
    complexity=8,  # High complexity - manages multiple strategies and confidence tracking
    supports_branching=True,  # Can explore multiple strategies as branches
    supports_revision=True,  # Revises approach when confidence drops
    requires_context=False,
    min_thoughts=3,  # At least: initial + confidence check + strategy switch
    max_thoughts=0,  # Unlimited - depends on number of strategy switches
    avg_tokens_per_thought=500,
    best_for=(
        "uncertain problems",
        "problems where initial approach may fail",
        "adaptive reasoning",
        "robust problem solving",
        "complex multi-faceted challenges",
        "exploratory analysis",
    ),
    not_recommended_for=(
        "problems with known solutions",
        "time-critical single-attempt tasks",
        "simple straightforward questions",
        "tasks requiring consistent methodology",
    ),
)


class CRASHMethod:
    """CRASH (Confidence-gated Reasoning with Automatic Strategy Handoff) implementation.

    This class implements an adaptive reasoning method that monitors confidence levels
    throughout the reasoning process and automatically switches strategies when confidence
    drops below a threshold. This provides resilience when the initial approach struggles.

    The method works by:
    1. Starting with a primary strategy
    2. Assessing confidence after each reasoning step
    3. Switching to alternative strategy when confidence < threshold
    4. Tracking all strategies attempted and their effectiveness
    5. Allowing recovery by switching back if confidence improves

    Available strategies:
    - "direct": Direct problem-solving approach
    - "decompose": Break problem into smaller parts
    - "analogize": Reason by analogy to similar problems
    - "abstract": Step back to higher-level concepts
    - "verify": Verification and validation approach

    Attributes:
        confidence_threshold: Confidence below which triggers strategy switch (default: 0.6)
        max_strategy_switches: Maximum number of strategy switches allowed (default: 3)
        fallback_strategy: Default strategy to use when switching (default: "decompose")

    Examples:
        Basic usage with default parameters:
        >>> crash = CRASHMethod()
        >>> session = Session().start()
        >>> await crash.initialize()
        >>> result = await crash.execute(session, "Solve this complex problem")

        Custom configuration:
        >>> crash = CRASHMethod(
        ...     confidence_threshold=0.7,
        ...     max_strategy_switches=5,
        ...     fallback_strategy="abstract"
        ... )
        >>> session = Session().start()
        >>> await crash.initialize()
        >>> result = await crash.execute(
        ...     session,
        ...     "Complex uncertain problem",
        ...     context={"initial_strategy": "direct"}
        ... )
    """

    # Available reasoning strategies
    STRATEGIES = {
        "direct": "Direct analytical approach to the problem",
        "decompose": "Break down problem into manageable subproblems",
        "analogize": "Reason by analogy to similar known problems",
        "abstract": "Step back to consider higher-level concepts",
        "verify": "Focus on verification and validation",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_strategy_switches: int = 3,
        fallback_strategy: str = "decompose",
    ) -> None:
        """Initialize the CRASH method.

        Args:
            confidence_threshold: Confidence below which to switch strategies (0.0-1.0)
            max_strategy_switches: Maximum number of strategy switches allowed
            fallback_strategy: Default fallback strategy name

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be 0.0-1.0, got {confidence_threshold}"
            )
        if max_strategy_switches < 1:
            raise ValueError(
                f"max_strategy_switches must be >= 1, got {max_strategy_switches}"
            )
        if fallback_strategy not in self.STRATEGIES:
            raise ValueError(
                f"fallback_strategy must be one of {list(self.STRATEGIES.keys())}, "
                f"got {fallback_strategy}"
            )

        self.confidence_threshold = confidence_threshold
        self.max_strategy_switches = max_strategy_switches
        self.fallback_strategy = fallback_strategy

        # Internal state
        self._initialized = False
        self._current_strategy: str = "direct"
        self._strategy_history: list[str] = []
        self._confidence_history: list[float] = []
        self._switch_count: int = 0
        self._step_counter: int = 0

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.CRASH

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return CRASH_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return CRASH_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        """Initialize the CRASH method.

        Resets all internal state for a fresh reasoning session.

        Examples:
            >>> method = CRASHMethod()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._switch_count == 0
        """
        self._initialized = True
        self._current_strategy = "direct"
        self._strategy_history = []
        self._confidence_history = []
        self._switch_count = 0
        self._step_counter = 0

    async def execute(
        self,
        session: Session,
        problem: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtGraph:
        """Execute CRASH reasoning on the problem.

        This method creates an initial thought using the primary strategy (or one
        specified in context), then returns a ThoughtGraph for continued exploration.

        Args:
            session: The current reasoning session
            problem: The problem or question to reason about
            context: Optional context with:
                - initial_strategy: Strategy to start with (default: "direct")
                - confidence_threshold: Override default threshold

        Returns:
            A ThoughtGraph containing the initial reasoning and setup for continuation

        Raises:
            RuntimeError: If the method has not been initialized
            ValueError: If session is not active

        Examples:
            >>> session = Session().start()
            >>> method = CRASHMethod()
            >>> await method.initialize()
            >>> graph = await method.execute(session, "Solve complex problem")
            >>> assert graph.root_id is not None
        """
        if not self._initialized:
            raise RuntimeError("CRASH method must be initialized before execution")

        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Reset for new execution
        self._step_counter = 1
        self._switch_count = 0
        self._strategy_history = []
        self._confidence_history = []

        # Extract context parameters
        context = context or {}
        initial_strategy = context.get("initial_strategy", "direct")
        if initial_strategy not in self.STRATEGIES:
            initial_strategy = "direct"

        threshold = context.get("confidence_threshold", self.confidence_threshold)

        self._current_strategy = initial_strategy
        self._strategy_history.append(initial_strategy)

        # Create thought graph
        graph = ThoughtGraph()

        # Create initial thought with primary strategy
        initial_confidence = 0.7  # Start with moderate confidence
        self._confidence_history.append(initial_confidence)

        content = self._generate_initial_content(problem, initial_strategy)

        root = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CRASH,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=initial_confidence,
            quality_score=0.7,
            metadata={
                "problem": problem,
                "strategy": self._current_strategy,
                "strategy_description": self.STRATEGIES[self._current_strategy],
                "confidence_threshold": threshold,
                "switch_count": self._switch_count,
                "max_switches": self.max_strategy_switches,
                "confidence_history": self._confidence_history.copy(),
                "strategy_history": self._strategy_history.copy(),
            },
        )

        graph.add_thought(root)
        session.add_thought(root)
        session.current_method = MethodIdentifier.CRASH

        return graph

    async def continue_reasoning(
        self,
        session: Session,
        graph: ThoughtGraph,
        feedback: str | None = None,
    ) -> ThoughtGraph:
        """Continue reasoning with confidence monitoring and strategy switching.

        This method continues the reasoning process by:
        1. Evaluating confidence in the current approach
        2. Switching strategies if confidence drops below threshold
        3. Continuing with current or new strategy
        4. Tracking all confidence levels and strategy changes

        Args:
            session: The current reasoning session
            graph: The thought graph from previous reasoning
            feedback: Optional feedback or guidance for next step

        Returns:
            Updated ThoughtGraph with continued reasoning

        Raises:
            RuntimeError: If the method has not been initialized
            ValueError: If session is not active or graph is invalid

        Examples:
            >>> session = Session().start()
            >>> method = CRASHMethod()
            >>> await method.initialize()
            >>> graph = await method.execute(session, "Problem")
            >>> # Continue with low confidence to trigger switch
            >>> updated_graph = await method.continue_reasoning(
            ...     session, graph, "approach struggling"
            ... )
        """
        if not self._initialized:
            raise RuntimeError("CRASH method must be initialized before continuation")

        if not session.is_active:
            raise ValueError("Session must be active to continue reasoning")

        if not graph.root_id or graph.root_id not in graph.nodes:
            raise ValueError("Graph must have a valid root node")

        self._step_counter += 1

        # Get the most recent thought (leaf with highest step number)
        leaves = [graph.nodes[leaf_id] for leaf_id in graph.leaf_ids]
        if not leaves:
            # No leaves, use root
            previous_thought = graph.nodes[graph.root_id]
        else:
            previous_thought = max(leaves, key=lambda n: n.step_number)

        # Extract threshold from previous thought
        threshold = previous_thought.metadata.get(
            "confidence_threshold", self.confidence_threshold
        )

        # Assess current confidence (simulated based on feedback and history)
        current_confidence = self._assess_confidence(previous_thought, feedback)
        self._confidence_history.append(current_confidence)

        # Determine if we need to switch strategies
        should_switch = (
            current_confidence < threshold
            and self._switch_count < self.max_strategy_switches
        )

        if should_switch:
            # Switch to a new strategy
            new_strategy = self._select_next_strategy()
            self._current_strategy = new_strategy
            self._strategy_history.append(new_strategy)
            self._switch_count += 1

            # Create strategy switch thought
            thought_type = ThoughtType.BRANCH  # Use BRANCH to indicate strategy change
            content = self._generate_strategy_switch_content(
                previous_thought, new_strategy, current_confidence, feedback
            )
            branch_id = f"strategy-{self._switch_count}-{new_strategy}"
        else:
            # Continue with current strategy
            thought_type = ThoughtType.CONTINUATION
            content = self._generate_continuation_content(
                previous_thought, current_confidence, feedback
            )
            branch_id = previous_thought.branch_id

        # Create new thought
        new_thought = ThoughtNode(
            id=str(uuid4()),
            type=thought_type,
            method_id=MethodIdentifier.CRASH,
            content=content,
            parent_id=previous_thought.id,
            branch_id=branch_id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=current_confidence,
            quality_score=self._calculate_quality_score(current_confidence),
            metadata={
                "strategy": self._current_strategy,
                "strategy_description": self.STRATEGIES[self._current_strategy],
                "confidence_threshold": threshold,
                "switch_count": self._switch_count,
                "max_switches": self.max_strategy_switches,
                "switched": should_switch,
                "confidence_history": self._confidence_history.copy(),
                "strategy_history": self._strategy_history.copy(),
                "feedback": feedback or "",
                "recovery_rate": self._calculate_recovery_rate(),
            },
        )

        graph.add_thought(new_thought)
        session.add_thought(new_thought)

        # If we've reached max switches or high confidence, create conclusion
        if (
            self._switch_count >= self.max_strategy_switches
            or current_confidence >= 0.9
        ):
            conclusion = self._create_conclusion(graph, new_thought)
            graph.add_thought(conclusion)
            session.add_thought(conclusion)

        return graph

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if initialized, False otherwise

        Examples:
            >>> method = CRASHMethod()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _generate_initial_content(self, problem: str, strategy: str) -> str:
        """Generate content for the initial thought.

        Args:
            problem: The problem to solve
            strategy: The initial strategy to use

        Returns:
            Content string for the initial thought
        """
        return (
            f"Step {self._step_counter}: CRASH Reasoning Initiated\n\n"
            f"Problem: {problem}\n\n"
            f"Initial Strategy: {strategy.upper()}\n"
            f"Description: {self.STRATEGIES[strategy]}\n\n"
            f"Confidence Monitoring: Active (threshold: {self.confidence_threshold})\n"
            f"Max Strategy Switches: {self.max_strategy_switches}\n\n"
            f"Beginning analysis with {strategy} approach...\n"
            f"[In a full implementation, this would contain LLM-generated reasoning "
            f"using the {strategy} strategy]\n\n"
            f"Initial confidence: 0.7 (moderate - monitoring for drops)"
        )

    def _generate_strategy_switch_content(
        self,
        previous: ThoughtNode,
        new_strategy: str,
        confidence: float,
        feedback: str | None,
    ) -> str:
        """Generate content for a strategy switch thought.

        Args:
            previous: The previous thought
            new_strategy: The new strategy being switched to
            confidence: Current confidence level
            feedback: Optional feedback

        Returns:
            Content string for the strategy switch thought
        """
        feedback_text = f"\n\nFeedback: {feedback}" if feedback else ""

        return (
            f"Step {self._step_counter}: STRATEGY SWITCH #{self._switch_count}\n\n"
            f"Confidence Drop Detected: {confidence:.2f} < {self.confidence_threshold}\n"
            f"Previous Strategy: {self._strategy_history[-2] if len(self._strategy_history) > 1 else 'none'}\n"
            f"New Strategy: {new_strategy.upper()}\n"
            f"Description: {self.STRATEGIES[new_strategy]}\n\n"
            f"Reason for Switch:\n"
            f"The current approach showed declining confidence. Automatically switching "
            f"to '{new_strategy}' strategy for better problem-solving fit.\n\n"
            f"Context Transfer:\n"
            f"Relevant insights from previous approach:\n"
            f"[Key findings from {previous.content[:100]}...]\n\n"
            f"Continuing with new strategy...{feedback_text}\n\n"
            f"Confidence History: {[f'{c:.2f}' for c in self._confidence_history]}\n"
            f"Strategy History: {self._strategy_history}"
        )

    def _generate_continuation_content(
        self,
        previous: ThoughtNode,
        confidence: float,
        feedback: str | None,
    ) -> str:
        """Generate content for a continuation thought.

        Args:
            previous: The previous thought
            confidence: Current confidence level
            feedback: Optional feedback

        Returns:
            Content string for the continuation thought
        """
        feedback_text = f"\n\nIncorporating feedback: {feedback}" if feedback else ""

        return (
            f"Step {self._step_counter}: Continuing with {self._current_strategy.upper()} Strategy\n\n"
            f"Current Confidence: {confidence:.2f} (threshold: {self.confidence_threshold})\n"
            f"Strategy: {self.STRATEGIES[self._current_strategy]}\n\n"
            f"Building on previous step:\n"
            f"[Continuing from: {previous.content[:100]}...]\n\n"
            f"Next reasoning step:\n"
            f"[In a full implementation, this would contain LLM-generated continuation "
            f"using the {self._current_strategy} strategy]{feedback_text}\n\n"
            f"Confidence stable - continuing with current approach.\n"
            f"Switches used: {self._switch_count}/{self.max_strategy_switches}"
        )

    def _create_conclusion(self, graph: ThoughtGraph, final_thought: ThoughtNode) -> ThoughtNode:
        """Create a conclusion thought summarizing the reasoning.

        Args:
            graph: The thought graph
            final_thought: The final thought before conclusion

        Returns:
            A conclusion ThoughtNode
        """
        avg_confidence = (
            sum(self._confidence_history) / len(self._confidence_history)
            if self._confidence_history
            else 0.0
        )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CRASH,
            content=(
                f"Step {self._step_counter + 1}: CRASH Reasoning Complete\n\n"
                f"Final Strategy: {self._current_strategy.upper()}\n"
                f"Final Confidence: {self._confidence_history[-1]:.2f}\n"
                f"Average Confidence: {avg_confidence:.2f}\n\n"
                f"Strategy Switches: {self._switch_count}/{self.max_strategy_switches}\n"
                f"Strategies Attempted: {' → '.join(self._strategy_history)}\n\n"
                f"Confidence History: {[f'{c:.2f}' for c in self._confidence_history]}\n\n"
                f"Recovery Success Rate: {self._calculate_recovery_rate():.1%}\n\n"
                f"Conclusion:\n"
                f"[In a full implementation, this would contain the final answer/solution "
                f"derived from the adaptive reasoning process]\n\n"
                f"Total Thoughts: {graph.node_count}\n"
                f"Maximum Depth: {graph.max_depth}"
            ),
            parent_id=final_thought.id,
            step_number=self._step_counter + 1,
            depth=final_thought.depth + 1,
            confidence=self._confidence_history[-1] if self._confidence_history else 0.0,
            quality_score=self._calculate_quality_score(
                self._confidence_history[-1] if self._confidence_history else 0.0
            ),
            metadata={
                "is_conclusion": True,
                "total_switches": self._switch_count,
                "final_strategy": self._current_strategy,
                "average_confidence": avg_confidence,
                "recovery_rate": self._calculate_recovery_rate(),
                "strategies_used": self._strategy_history.copy(),
            },
        )

    def _assess_confidence(self, previous: ThoughtNode, feedback: str | None) -> float:
        """Assess confidence for the current step.

        In a full implementation, this would use LLM evaluation. Here we simulate
        confidence based on history and feedback.

        Args:
            previous: Previous thought node
            feedback: Optional feedback

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with previous confidence
        base_confidence = previous.confidence

        # Simulate confidence change based on various factors
        if feedback and any(
            word in feedback.lower()
            for word in ["struggling", "difficult", "unclear", "stuck"]
        ):
            # Negative feedback decreases confidence
            adjustment = -0.15
        elif feedback and any(
            word in feedback.lower()
            for word in ["progress", "good", "clear", "working"]
        ):
            # Positive feedback increases confidence
            adjustment = 0.1
        else:
            # Natural decay or slight improvement based on switch history
            if self._switch_count > 0:
                # After a switch, confidence typically improves
                adjustment = 0.05
            else:
                # Slight random variation
                adjustment = (hash(str(datetime.now())) % 20 - 10) / 100.0

        # Calculate new confidence with bounds
        new_confidence = max(0.0, min(1.0, base_confidence + adjustment))

        return new_confidence

    def _select_next_strategy(self) -> str:
        """Select the next strategy to try.

        Returns:
            Name of the next strategy to use
        """
        # Avoid recently used strategies
        available = [
            s for s in self.STRATEGIES.keys()
            if s not in self._strategy_history[-2:] if self._strategy_history
        ]

        if not available:
            # All strategies tried recently, use fallback
            return self.fallback_strategy

        # Select based on priority order
        priority_order = ["decompose", "abstract", "analogize", "verify", "direct"]

        for strategy in priority_order:
            if strategy in available:
                return strategy

        # Fallback to first available
        return available[0]

    def _calculate_quality_score(self, confidence: float) -> float:
        """Calculate quality score based on confidence and other factors.

        Args:
            confidence: Current confidence level

        Returns:
            Quality score (0.0-1.0)
        """
        # Base quality on confidence
        base = confidence

        # Bonus for successful strategy switches (shows adaptability)
        switch_bonus = min(0.1, self._switch_count * 0.03)

        # Penalty if we've exhausted switches without resolution
        if self._switch_count >= self.max_strategy_switches:
            switch_penalty = 0.05
        else:
            switch_penalty = 0.0

        return max(0.0, min(1.0, base + switch_bonus - switch_penalty))

    def _calculate_recovery_rate(self) -> float:
        """Calculate the recovery success rate.

        Measures how often confidence improved after strategy switches.

        Returns:
            Recovery rate (0.0-1.0)
        """
        if self._switch_count == 0 or len(self._confidence_history) < 2:
            return 1.0  # No switches yet, default to perfect

        # Count improvements after switches
        improvements = 0
        for i in range(1, len(self._confidence_history)):
            if self._confidence_history[i] > self._confidence_history[i - 1]:
                improvements += 1

        return improvements / (len(self._confidence_history) - 1)


# Export metadata and class
__all__ = [
    "CRASHMethod",
    "CRASH_METADATA",
]
