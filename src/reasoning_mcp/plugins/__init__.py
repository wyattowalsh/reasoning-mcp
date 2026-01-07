"""Plugin system for reasoning-mcp.

This package provides the plugin architecture for extending the server
with custom reasoning methods.
"""

from reasoning_mcp.plugins.interface import (
    Plugin,
    PluginContext,
    PluginError,
    PluginMetadata,
)
from reasoning_mcp.plugins.loader import PluginLoader

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginMetadata",
    "PluginLoader",
]
