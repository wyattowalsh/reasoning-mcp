"""Prompts module for reasoning-mcp.

This module contains MCP prompt functions for guided reasoning and pipeline workflows.
"""

from reasoning_mcp.prompts.guided import register_guided_prompts
from reasoning_mcp.prompts.pipelines import register_pipeline_prompts


def register_all_prompts(mcp):
    """Register all MCP prompts with the FastMCP server.

    This function registers all prompt categories with the provided FastMCP instance:
    - Guided reasoning prompts for step-by-step workflows
    - Pipeline prompts for multi-stage reasoning workflows

    Args:
        mcp: The FastMCP server instance to register prompts with

    Example:
        >>> from reasoning_mcp.server import mcp
        >>> register_all_prompts(mcp)
    """
    register_guided_prompts(mcp)
    register_pipeline_prompts(mcp)


__all__ = [
    "register_all_prompts",
    "register_guided_prompts",
    "register_pipeline_prompts",
]
