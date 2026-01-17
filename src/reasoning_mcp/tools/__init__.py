"""Tools module for reasoning-mcp.

This module contains MCP tool functions for session management and reasoning operations.
"""

from reasoning_mcp.tools.compose import compose
from reasoning_mcp.tools.debug import (
    AnalyzeTraceInput,
    DebugToolInput,
    GetTraceInput,
    ListTracesInput,
    analyze_trace,
    enable_tracing,
    get_trace,
    list_traces,
    set_trace_storage,
)
from reasoning_mcp.tools.evaluate import evaluate
from reasoning_mcp.tools.methods import methods_compare, methods_list, methods_recommend
from reasoning_mcp.tools.reason import reason
from reasoning_mcp.tools.register import register_tools
from reasoning_mcp.tools.session import (
    session_branch,
    session_continue,
    session_inspect,
    session_merge,
)

__all__ = [
    # Core reasoning tool
    "reason",
    # Method tools
    "methods_list",
    "methods_recommend",
    "methods_compare",
    # Session tools
    "session_continue",
    "session_branch",
    "session_inspect",
    "session_merge",
    # Pipeline tools
    "compose",
    "evaluate",
    # Debug tools
    "enable_tracing",
    "get_trace",
    "list_traces",
    "analyze_trace",
    "set_trace_storage",
    "DebugToolInput",
    "GetTraceInput",
    "ListTracesInput",
    "AnalyzeTraceInput",
    # Registration
    "register_tools",
]
