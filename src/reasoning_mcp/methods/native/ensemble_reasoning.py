"""Ensemble reasoning method combining multiple approaches.

Ensemble Reasoning is a meta-method that combines multiple reasoning approaches
using configurable voting strategies to produce more robust and accurate results.
It executes multiple methods in parallel and aggregates their outputs through
voting mechanisms like majority, weighted, consensus, or synthesis.

This method is particularly effective for:
- Complex problems where single methods may be insufficient
- High-stakes decisions requiring multiple perspectives
- Problems where diverse reasoning approaches add value
- Scenarios requiring confidence calibration and uncertainty quantification

Example ensemble:
    1. Configure ensemble with multiple methods (COT, Tree of Thoughts, Self-Reflection)
    2. Execute all methods in parallel with timeout protection
    3. Aggregate results using voting strategy (e.g., majority voting)
    4. Calculate agreement score to measure consensus
    5. Return final answer with confidence and detailed voting information
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    VotingStrategy,
)

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext

# Define metadata for the Ensemble Reasoning method
ENSEMBLE_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ENSEMBLE_REASONING,
    name="Ensemble Reasoning",
    description=(
        "Combines multiple reasoning methods using voting strategies "
        "for robust, multi-perspective analysis"
    ),
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "ensemble",
            "multi-method",
            "voting",
            "robust",
            "parallel",
            "consensus",
            "aggregation",
        }
    ),
    complexity=7,
    supports_branching=False,
    supports_revision=False,
    requires_context=False,
    min_thoughts=1,
    max_thoughts=1,  # Single aggregated thought
    avg_tokens_per_thought=300,
    best_for=(
        "Complex problems requiring multiple perspectives",
        "High-stakes decisions needing validation",
        "Uncertainty quantification and confidence calibration",
        "Combining diverse reasoning approaches",
        "Robust problem solving with fallback mechanisms",
    ),
    not_recommended_for=(
        "Simple problems where single method suffices",
        "Time-critical tasks (due to parallel execution overhead)",
        "Problems with clear single optimal approach",
        "Low-complexity queries",
    ),
)


class EnsembleReasoning(ReasoningMethodBase):
    """Ensemble reasoning method combining multiple approaches.

    This class implements ensemble reasoning that orchestrates multiple reasoning
    methods in parallel and aggregates their results using configurable voting
    strategies. The method provides robust reasoning through diversity, consensus
    measurement, and graceful degradation under timeout.

    Attributes:
        identifier: Unique identifier matching MethodIdentifier.ENSEMBLE_REASONING
        name: Human-readable name "Ensemble Reasoning"
        description: Brief description of the method
        category: Category as MethodCategory.ADVANCED
        config: EnsembleConfig specifying members and voting strategy

    Examples:
        >>> method = EnsembleReasoning()
        >>> session = Session().start()
        >>> await method.initialize()
        >>> thought = await method.execute(
        ...     session,
        ...     "What is the best approach to solve this problem?"
        ... )
        >>> print(thought.content)
        # Aggregated answer from multiple reasoning methods
        >>> print(thought.metadata["agreement_score"])
        0.85  # High agreement among members
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        """Initialize with optional config (uses defaults if None).

        Args:
            config: Optional EnsembleConfig. If None, uses default configuration
                with COT, Tree of Thoughts, and Self-Reflection with majority voting.
        """
        self.config = config or self.get_default_config()
        self._is_initialized = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Return the unique identifier for this method."""
        return str(MethodIdentifier.ENSEMBLE_REASONING)

    @property
    def name(self) -> str:
        """Return the human-readable name of this method."""
        return "Ensemble Reasoning"

    @property
    def description(self) -> str:
        """Return a brief description of this method."""
        return "Combines multiple reasoning methods using voting strategies for robust analysis"

    @property
    def category(self) -> str:
        """Return the category this method belongs to."""
        return str(MethodCategory.ADVANCED)

    @staticmethod
    def get_default_config() -> EnsembleConfig:
        """Get sensible default ensemble configuration.

        Default uses COT, Tree of Thoughts, and Self-Reflection
        with majority voting.

        Returns:
            EnsembleConfig with three core methods and majority voting strategy.

        Examples:
            >>> config = EnsembleReasoning.get_default_config()
            >>> assert len(config.members) == 3
            >>> assert config.strategy == VotingStrategy.MAJORITY
            >>> assert config.min_agreement == 0.5
            >>> assert config.timeout_ms == 30000
        """
        return EnsembleConfig(
            members=[
                EnsembleMember(method_name="chain_of_thought", weight=1.0),
                EnsembleMember(method_name="tree_of_thoughts", weight=1.0),
                EnsembleMember(method_name="self_reflection", weight=1.0),
            ],
            strategy=VotingStrategy.MAJORITY,
            min_agreement=0.5,
            timeout_ms=30000,
        )

    async def initialize(self) -> None:
        """Initialize the method.

        For Ensemble Reasoning, initialization validates the configuration
        and prepares the orchestrator for parallel execution.
        """
        # Validate configuration
        if not self.config.members:
            raise ValueError("EnsembleConfig must have at least one member")

        self._is_initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute ensemble reasoning on the input.

        This method orchestrates parallel execution of multiple reasoning methods
        and aggregates their results using the configured voting strategy. It
        handles timeout gracefully, calculates agreement scores, and provides
        detailed voting information for transparency.

        The execution flow:
        1. Initialize ensemble orchestrator with config and execution context
        2. Execute all member methods in parallel
        3. Collect results (partial if timeout)
        4. Apply voting strategy to aggregate results
        5. Calculate agreement score across member outputs
        6. Create ThoughtNode with final answer and metadata

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (custom config, etc.)
            execution_context: Optional ExecutionContext for method execution

        Returns:
            A ThoughtNode containing the aggregated ensemble result with
            metadata including agreement score, voting details, and member results.

        Raises:
            RuntimeError: If orchestrator execution fails unexpectedly.
            TimeoutError: If ensemble execution times out completely.
            ValueError: If no ensemble members complete successfully.

        Examples:
            >>> session = Session().start()
            >>> method = EnsembleReasoning()
            >>> await method.initialize()
            >>> thought = await method.execute(session, "What is 2+2?")
            >>> assert thought.content is not None
            >>> assert 0.0 <= thought.metadata["agreement_score"] <= 1.0
            >>> assert "voting_details" in thought.metadata
        """
        if not self._is_initialized:
            await self.initialize()

        # Store execution context for member method execution
        self._execution_context = execution_context

        # Allow context to override config
        context = context or {}
        config = context.get("ensemble_config", self.config)

        # Execute ensemble orchestration with proper exception handling
        from reasoning_mcp.ensemble.orchestrator import EnsembleOrchestrator

        orchestrator = EnsembleOrchestrator(
            config=config,
            execution_context=execution_context,
        )

        try:
            result = await orchestrator.execute(input_text)
        except TimeoutError as e:
            logger.warning(
                "ensemble_timeout",
                method=self.identifier,
                timeout_ms=config.timeout_ms,
                error=str(e),
            )
            raise
        except (ConnectionError, OSError) as e:
            logger.warning(
                "ensemble_connection_error",
                method=self.identifier,
                error=str(e),
            )
            raise RuntimeError(
                f"Ensemble execution failed due to connection error: {e}"
            ) from e
        except ValueError as e:
            # No members completed - re-raise with context
            logger.warning(
                "ensemble_no_results",
                method=self.identifier,
                error=str(e),
            )
            raise

        # Determine thought type based on session state
        thought_type = ThoughtType.INITIAL
        parent_id = None
        depth = 0

        # If session has thoughts, this is a continuation
        if session.thought_count > 0:
            thought_type = ThoughtType.CONTINUATION
            # Get the most recent thought as parent
            recent_thoughts = session.get_recent_thoughts(n=1)
            if recent_thoughts:
                parent = recent_thoughts[0]
                parent_id = parent.id
                depth = parent.depth + 1

        # Create the thought node with aggregated result
        thought = ThoughtNode(
            id=str(uuid4()),
            type=thought_type,
            method_id=MethodIdentifier.ENSEMBLE_REASONING,
            content=result.final_answer,
            parent_id=parent_id,
            depth=depth,
            confidence=result.confidence,
            step_number=session.thought_count + 1,
            metadata={
                "agreement_score": result.agreement_score,
                "member_count": len(result.member_results),
                "strategy": config.strategy.value,
                "voting_details": result.voting_details,
                "input_text": input_text,
                "method": "ensemble_reasoning",
                "timeout_ms": config.timeout_ms,
                "min_agreement": config.min_agreement,
            },
        )

        # Add thought to session
        session.add_thought(thought)

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

        For ensemble reasoning, continuation means running the ensemble again
        with the previous result as additional context or guidance.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the continuation
            context: Optional additional context
            execution_context: Optional ExecutionContext for method execution

        Returns:
            A new ThoughtNode continuing the reasoning chain

        Examples:
            >>> # After initial execution
            >>> continuation = await method.continue_reasoning(
            ...     session,
            ...     previous_thought,
            ...     guidance="Now consider alternative approaches"
            ... )
            >>> assert continuation.parent_id == previous_thought.id
        """
        if not self._is_initialized:
            await self.initialize()

        # Build continuation input incorporating previous result and guidance
        continuation_input = f"Previous result: {previous_thought.content}\n\n"
        if guidance:
            continuation_input += f"Guidance: {guidance}\n\n"
        continuation_input += "Continue the analysis with this new context."

        # Execute ensemble with continuation context
        return await self.execute(
            session,
            continuation_input,
            context=context,
            execution_context=execution_context,
        )

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and ready, False otherwise

        Examples:
            >>> method = EnsembleReasoning()
            >>> assert await method.health_check() is False  # Not initialized
            >>> await method.initialize()
            >>> assert await method.health_check() is True  # Now ready
        """
        return self._is_initialized

    async def emit_thought(
        self,
        content: str,
        confidence: float | None = None,
    ) -> None:
        """Emit a thought for streaming (if streaming context is available).

        Args:
            content: The thought content to emit
            confidence: Optional confidence score for the thought
        """
        # Ensemble reasoning doesn't use streaming directly as it delegates
        # to the orchestrator, but we implement this for interface compatibility
        pass
