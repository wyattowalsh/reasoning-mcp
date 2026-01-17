"""Voting strategies for ensemble reasoning.

This package provides different strategies for aggregating results from multiple
reasoning methods in an ensemble. Each strategy implements the VotingStrategyProtocol
and provides a different approach to combining member results into a final answer.

Available strategies:
- MajorityVoting: Simple majority vote - most common answer wins
- WeightedVoting: Weighted voting by member weight * confidence score
- ConsensusVoting: Requires minimum agreement threshold to reach decision
- BestScoreVoting: Select answer with highest confidence score
- SynthesisVoting: LLM-powered synthesis of multiple results into unified answer
- RankedChoiceVoting: Instant runoff voting with elimination rounds
- BordaCountVoting: Borda count voting with ranked preference scoring

The registry module provides:
- STRATEGY_REGISTRY: Mapping from VotingStrategy enums to implementation classes
- get_strategy: Factory function for creating strategy instances by enum value
"""

from __future__ import annotations

from reasoning_mcp.ensemble.strategies.base import VotingStrategyProtocol
from reasoning_mcp.ensemble.strategies.best_score import BestScoreVoting
from reasoning_mcp.ensemble.strategies.borda_count import BordaCountVoting
from reasoning_mcp.ensemble.strategies.consensus import ConsensusVoting
from reasoning_mcp.ensemble.strategies.majority import MajorityVoting
from reasoning_mcp.ensemble.strategies.ranked_choice import RankedChoiceVoting
from reasoning_mcp.ensemble.strategies.registry import STRATEGY_REGISTRY, get_strategy
from reasoning_mcp.ensemble.strategies.synthesis import SynthesisVoting
from reasoning_mcp.ensemble.strategies.weighted import WeightedVoting

__all__ = [
    "VotingStrategyProtocol",
    "MajorityVoting",
    "WeightedVoting",
    "ConsensusVoting",
    "BestScoreVoting",
    "SynthesisVoting",
    "RankedChoiceVoting",
    "BordaCountVoting",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
