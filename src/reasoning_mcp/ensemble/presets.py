"""Ensemble configuration presets for common use cases.

This module provides pre-configured ensemble settings optimized for different
scenarios such as balanced reasoning, high accuracy requirements, speed-focused
execution, and consensus building.

The presets define common combinations of reasoning methods with appropriate
voting strategies and timeouts, making it easy to use ensembles without manual
configuration.

Examples:
    Use a balanced preset:
    >>> from reasoning_mcp.ensemble.presets import get_preset
    >>> config = get_preset("balanced")
    >>> assert len(config.members) == 3
    >>> assert config.strategy == VotingStrategy.MAJORITY

    Use an accuracy-focused preset:
    >>> config = get_preset("accuracy")
    >>> assert config.strategy == VotingStrategy.WEIGHTED
    >>> assert any(m.weight > 1.0 for m in config.members)

    List available presets:
    >>> from reasoning_mcp.ensemble.presets import ENSEMBLE_PRESETS
    >>> preset_names = list(ENSEMBLE_PRESETS.keys())
    >>> assert "balanced" in preset_names
    >>> assert "accuracy" in preset_names
    >>> assert "speed" in preset_names
    >>> assert "consensus" in preset_names
"""

from __future__ import annotations

from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    VotingStrategy,
)

# Preset configurations
ENSEMBLE_PRESETS: dict[str, EnsembleConfig] = {
    "balanced": EnsembleConfig(
        members=[
            EnsembleMember(method_name="chain_of_thought", weight=1.0),
            EnsembleMember(method_name="tree_of_thoughts", weight=1.0),
            EnsembleMember(method_name="self_reflection", weight=1.0),
        ],
        strategy=VotingStrategy.MAJORITY,
    ),
    "accuracy": EnsembleConfig(
        members=[
            EnsembleMember(method_name="chain_of_thought", weight=1.0),
            EnsembleMember(method_name="self_consistency", weight=2.0),
            EnsembleMember(method_name="tree_of_thoughts", weight=1.5),
            EnsembleMember(method_name="self_reflection", weight=1.0),
        ],
        strategy=VotingStrategy.WEIGHTED,
    ),
    "speed": EnsembleConfig(
        members=[
            EnsembleMember(method_name="chain_of_thought", weight=1.0),
            EnsembleMember(method_name="react", weight=1.0),
        ],
        strategy=VotingStrategy.BEST_SCORE,
        timeout_ms=10000,
    ),
    "consensus": EnsembleConfig(
        members=[
            EnsembleMember(method_name="chain_of_thought"),
            EnsembleMember(method_name="tree_of_thoughts"),
            EnsembleMember(method_name="self_consistency"),
        ],
        strategy=VotingStrategy.CONSENSUS,
        min_agreement=0.67,
    ),
}


def get_preset(name: str) -> EnsembleConfig:
    """Get an ensemble preset configuration by name.

    Retrieves a pre-configured EnsembleConfig optimized for a specific use case.
    Available presets include:
    - balanced: General-purpose ensemble with equal weights
    - accuracy: High-accuracy ensemble with weighted voting
    - speed: Fast execution with minimal members
    - consensus: Consensus-based decision making

    Args:
        name: Preset name (balanced, accuracy, speed, consensus)

    Returns:
        The preset EnsembleConfig

    Raises:
        KeyError: If preset name not found

    Examples:
        Get a balanced preset:
        >>> config = get_preset("balanced")
        >>> assert config.strategy == VotingStrategy.MAJORITY

        Get an accuracy preset:
        >>> config = get_preset("accuracy")
        >>> assert config.strategy == VotingStrategy.WEIGHTED

        Handle unknown preset:
        >>> try:
        ...     config = get_preset("unknown")
        ... except KeyError as e:
        ...     assert "unknown" in str(e).lower()
        ...     assert "available" in str(e).lower()
    """
    if name not in ENSEMBLE_PRESETS:
        available = ", ".join(ENSEMBLE_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return ENSEMBLE_PRESETS[name]


__all__ = [
    "ENSEMBLE_PRESETS",
    "get_preset",
]
