"""Utility modules for reasoning-mcp.

This package provides optional utility modules that extend the core
functionality with advanced features like graph analysis and input validation.

Components:
    - graph_utils: NetworkX-backed analysis for ThoughtGraph
    - validation: Input validation and sanitization utilities
"""

from reasoning_mcp.utils.graph_utils import (
    ThoughtGraphNetworkX,
    is_networkx_available,
)
from reasoning_mcp.utils.validation import (
    MAX_INPUT_LENGTH,
    ValidationError,
    sanitize_prompt,
    validate_input_length,
    validate_metadata,
    validate_method_identifier,
)

__all__ = [
    # Graph utilities
    "ThoughtGraphNetworkX",
    "is_networkx_available",
    # Validation utilities
    "MAX_INPUT_LENGTH",
    "ValidationError",
    "sanitize_prompt",
    "validate_input_length",
    "validate_metadata",
    "validate_method_identifier",
]
