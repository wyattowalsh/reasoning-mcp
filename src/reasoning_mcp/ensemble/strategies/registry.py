"""Registry for voting strategies in ensemble reasoning.

This module provides a centralized registry that maps VotingStrategy enum values
to their corresponding implementation classes. It enables dynamic strategy selection
and instantiation based on configuration.

The registry supports all voting strategies defined in the VotingStrategy enum:
- MAJORITY: Simple majority vote
- WEIGHTED: Weighted voting by member weight and confidence
- CONSENSUS: Requires minimum agreement threshold
- BEST_SCORE: Selects answer with highest confidence score
- SYNTHESIS: LLM-powered synthesis of multiple results
- RANKED_CHOICE: Ranked choice voting with preference ordering
- BORDA_COUNT: Borda count voting system with ranked preferences

Examples:
    Get a strategy instance:
    >>> from reasoning_mcp.models.ensemble import VotingStrategy
    >>> strategy = get_strategy(VotingStrategy.MAJORITY)
    >>> assert isinstance(strategy, MajorityVoting)

    Get a weighted strategy with custom config:
    >>> strategy = get_strategy(
    ...     VotingStrategy.WEIGHTED,
    ...     min_weight=0.1
    ... )
    >>> assert isinstance(strategy, WeightedVoting)

    Handle missing strategy:
    >>> try:
    ...     strategy = get_strategy("invalid")
    ... except KeyError as e:
    ...     print(f"Strategy not found: {e}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.ensemble.strategies.best_score import BestScoreVoting
from reasoning_mcp.ensemble.strategies.borda_count import BordaCountVoting
from reasoning_mcp.ensemble.strategies.consensus import ConsensusVoting
from reasoning_mcp.ensemble.strategies.majority import MajorityVoting
from reasoning_mcp.ensemble.strategies.ranked_choice import RankedChoiceVoting
from reasoning_mcp.ensemble.strategies.synthesis import SynthesisVoting
from reasoning_mcp.ensemble.strategies.weighted import WeightedVoting
from reasoning_mcp.models.ensemble import VotingStrategy

if TYPE_CHECKING:
    from reasoning_mcp.ensemble.strategies.base import VotingStrategyProtocol


# Task 5.3: Registry mapping VotingStrategy enums to implementation classes
STRATEGY_REGISTRY: dict[VotingStrategy, type[VotingStrategyProtocol]] = {
    VotingStrategy.MAJORITY: MajorityVoting,
    VotingStrategy.WEIGHTED: WeightedVoting,
    VotingStrategy.CONSENSUS: ConsensusVoting,
    VotingStrategy.BEST_SCORE: BestScoreVoting,
    VotingStrategy.SYNTHESIS: SynthesisVoting,
    VotingStrategy.RANKED_CHOICE: RankedChoiceVoting,
    VotingStrategy.BORDA_COUNT: BordaCountVoting,
}


# Task 5.4: Factory function for creating strategy instances
def get_strategy(
    strategy: VotingStrategy,
    **kwargs: Any,
) -> VotingStrategyProtocol:
    """Get a voting strategy instance by enum value.

    This factory function looks up the appropriate strategy class in the registry
    and instantiates it with the provided keyword arguments. This enables dynamic
    strategy selection based on configuration.

    Args:
        strategy: The VotingStrategy enum value specifying which strategy to use.
            Must be one of the values defined in the VotingStrategy enum.
        **kwargs: Additional keyword arguments passed to the strategy constructor.
            These are strategy-specific configuration parameters. For example:
            - ConsensusVoting: min_agreement (float)
            - WeightedVoting: min_weight (float)
            - SynthesisVoting: llm_config (dict)

    Returns:
        An instance of the corresponding voting strategy class that implements
        the VotingStrategyProtocol.

    Raises:
        KeyError: If the provided strategy enum value is not found in the registry.
            This can happen if:
            1. An invalid/unsupported strategy enum value was provided
            2. The strategy is defined in the enum but not yet implemented
            3. The registry mapping is incomplete or misconfigured

    Examples:
        Get a majority voting strategy:
        >>> strategy = get_strategy(VotingStrategy.MAJORITY)
        >>> assert isinstance(strategy, MajorityVoting)

        Get a consensus strategy with custom threshold:
        >>> strategy = get_strategy(
        ...     VotingStrategy.CONSENSUS,
        ...     min_agreement=0.75
        ... )
        >>> assert isinstance(strategy, ConsensusVoting)

        Get a weighted voting strategy:
        >>> strategy = get_strategy(
        ...     VotingStrategy.WEIGHTED,
        ...     min_weight=0.1
        ... )
        >>> assert isinstance(strategy, WeightedVoting)

        Handle missing strategy:
        >>> try:
        ...     # This will raise KeyError if RANKED_CHOICE not yet implemented
        ...     strategy = get_strategy(VotingStrategy.RANKED_CHOICE)
        ... except KeyError:
        ...     print("Strategy not yet implemented")
        Strategy not yet implemented

    Note:
        Some strategies defined in the VotingStrategy enum (RANKED_CHOICE,
        BORDA_COUNT) may not yet be implemented. Attempting to get these
        strategies will raise a KeyError until they are added to the registry.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Voting strategy '{strategy}' not found in registry. "
            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}. "
            f"This strategy may not be implemented yet."
        )

    strategy_class = STRATEGY_REGISTRY[strategy]
    return strategy_class(**kwargs)
