"""Reasoning methods for reasoning-mcp.

This package contains the base protocol, metadata definitions, and all
native reasoning method implementations.
"""

from reasoning_mcp.methods.base import (
    CREATIVE_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SAMPLING_TEMPERATURE,
    PRECISE_TEMPERATURE,
    MethodMetadata,
    ReasoningMethod,
    ReasoningMethodBase,
)

__all__ = [
    # Classes
    "ReasoningMethod",
    "ReasoningMethodBase",
    "MethodMetadata",
    # Sampling constants
    "DEFAULT_SAMPLING_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "CREATIVE_TEMPERATURE",
    "PRECISE_TEMPERATURE",
]
