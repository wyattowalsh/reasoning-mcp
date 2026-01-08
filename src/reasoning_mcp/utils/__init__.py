"""Utility modules for reasoning-mcp.

This package provides optional utility modules that extend the core
functionality with advanced features like graph analysis.

Components:
    - graph_utils: NetworkX-backed analysis for ThoughtGraph
"""

from reasoning_mcp.utils.graph_utils import (
    ThoughtGraphNetworkX,
    is_networkx_available,
)

__all__ = [
    "ThoughtGraphNetworkX",
    "is_networkx_available",
]
