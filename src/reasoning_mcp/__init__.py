"""
reasoning-mcp: A unified MCP server aggregating 30+ reasoning methodologies.

This package provides a comprehensive Model Context Protocol (MCP) server that
integrates multiple advanced reasoning techniques including Chain of Thought,
Tree of Thoughts, Graph of Thoughts, and many more. It enables LLMs to leverage
sophisticated reasoning strategies for complex problem-solving tasks.

The server exposes these reasoning methodologies through standardized MCP tools,
making them accessible to any MCP-compatible client.
"""

__version__ = "0.1.0"

# Future exports will be added here as modules are implemented
__all__ = []

# ============================================================================
# Core Models
# ============================================================================
# from .models import ReasoningRequest, ReasoningResponse, ReasoningStep
# from .models import ChainOfThoughtModel, TreeOfThoughtsModel, etc.

# ============================================================================
# Reasoning Methods
# ============================================================================
# from .methods import ChainOfThought, TreeOfThoughts, GraphOfThoughts
# from .methods import ReasoningMethod, BaseReasoner

# ============================================================================
# MCP Tools
# ============================================================================
# from .tools import register_reasoning_tools, ReasoningToolRegistry
# from .tools import create_mcp_server

# ============================================================================
# Server Components
# ============================================================================
# from .server import ReasoningMCPServer, ServerConfig

# ============================================================================
# Utilities
# ============================================================================
# from .utils import format_reasoning_output, validate_reasoning_input
# from .utils import ReasoningLogger, ConfigManager
