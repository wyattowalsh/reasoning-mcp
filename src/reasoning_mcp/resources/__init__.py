"""MCP resources for reasoning-mcp.

This module provides resource registration and management for the reasoning-mcp
server. Resources expose session state, method documentation, traces, and
pipeline templates through the MCP resource protocol.

Resources are registered with the FastMCP server instance and provide
read-only access to various server state and metadata:

- Session resources: Access session state and thought graphs
- Method resources: View method documentation and schemas
- Trace resources: Export OpenTelemetry-compatible traces
- Template resources: Retrieve built-in pipeline templates

Example:
    Register all resources with the MCP server:

    >>> from reasoning_mcp.server import mcp
    >>> from reasoning_mcp.resources import register_all_resources
    >>> register_all_resources(mcp)

    Access resources via MCP protocol:

    >>> # session://abc-123
    >>> # method://sequential-thinking
    >>> # trace://abc-123
    >>> # template://deep-analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# Import resource registration functions from submodules
from reasoning_mcp.resources.session import register_session_resources
from reasoning_mcp.resources.method import register_method_resources
from reasoning_mcp.resources.trace import register_trace_resources
from reasoning_mcp.resources.template import register_template_resources


def register_all_resources(mcp: FastMCP) -> None:
    """Register all MCP resources with the FastMCP server.

    This function registers all resource types with the provided FastMCP
    server instance. Resources include session state, method documentation,
    execution traces, and pipeline templates.

    Resources are registered using the @mcp.resource decorator pattern
    in their respective modules and are automatically available once
    registered.

    Args:
        mcp: The FastMCP server instance to register resources with

    Example:
        >>> from reasoning_mcp.server import mcp
        >>> from reasoning_mcp.resources import register_all_resources
        >>> register_all_resources(mcp)
        >>> # Resources are now available via MCP protocol

    Note:
        This function should be called during server initialization,
        typically in the main server.py module after the FastMCP
        instance is created.
    """
    # Register session resources (TASK-060)
    register_session_resources(mcp)

    # Register method resources (TASK-061)
    register_method_resources(mcp)

    # Register trace resources (TASK-062)
    register_trace_resources(mcp)

    # Register template resources (TASK-063)
    register_template_resources(mcp)


__all__ = [
    "register_all_resources",
    "register_session_resources",
    "register_method_resources",
    "register_trace_resources",
    "register_template_resources",
]
