"""Ensemble reasoning package for combining multiple reasoning approaches.

This package provides ensemble methods that combine outputs from multiple
reasoning strategies to produce more robust and accurate results. Ensemble
techniques include:

- Voting mechanisms (majority, weighted, ranked)
- Aggregation strategies (mean, median, weighted fusion)
- Multi-model orchestration
- Consensus-based decision making
- Dynamic model selection and switching
- Training-free ensemble optimization

Each ensemble method can coordinate multiple reasoning methods or models
to leverage their complementary strengths and mitigate individual weaknesses.

The main components are:
- EnsembleOrchestrator: Coordinates parallel execution and result aggregation
- EnsembleAggregator: Meta-aggregates results from multiple ensemble runs
- VotingStrategy implementations: Various strategies for combining results
- EnsembleConfig: Configuration for ensemble members and settings
"""

from reasoning_mcp.ensemble.aggregator import EnsembleAggregator
from reasoning_mcp.ensemble.calibration import ConfidenceCalibrator
from reasoning_mcp.ensemble.orchestrator import EnsembleOrchestrator
from reasoning_mcp.ensemble.presets import ENSEMBLE_PRESETS, get_preset

__all__ = [
    "ENSEMBLE_PRESETS",
    "ConfidenceCalibrator",
    "EnsembleAggregator",
    "EnsembleOrchestrator",
    "get_preset",
]
