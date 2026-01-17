"""Component type enumeration for reasoning-mcp plugin system.

This module defines the ComponentType enum which categorizes all types of
components that can be registered by plugins in the unified component registry.
"""

from __future__ import annotations

from enum import Enum


class ComponentType(str, Enum):
    """Types of components that can be registered by plugins.

    The ComponentType enum defines all categories of components that plugins
    can register with the ComponentRegistry. This unified type system replaces
    the previous specialized registries with a single, extensible registry.

    Example:
        >>> from reasoning_mcp.models.component import ComponentType
        >>> component_type = ComponentType.METHOD
        >>> print(component_type.value)
        'method'
    """

    METHOD = "method"
    """Reasoning methods - core algorithms for reasoning and problem-solving."""

    EXECUTOR = "executor"
    """Pipeline executors - components that execute pipeline stages."""

    MIDDLEWARE = "middleware"
    """Middleware components - request/response processing and transformation."""

    STORAGE = "storage"
    """Storage backends - persistence and retrieval of sessions and data."""

    EVALUATOR = "evaluator"
    """Evaluators - components that assess reasoning quality and correctness."""

    SELECTOR = "selector"
    """Selectors - components that choose appropriate methods for queries."""

    PIPELINE = "pipeline"
    """Pipeline definitions - orchestration patterns for reasoning workflows."""

    MCP_TOOL = "mcp_tool"
    """MCP tools - Model Context Protocol tool definitions."""

    MCP_RESOURCE = "mcp_resource"
    """MCP resources - Model Context Protocol resource handlers."""
