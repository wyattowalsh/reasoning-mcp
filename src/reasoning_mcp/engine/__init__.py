"""Pipeline execution engine.

This package provides the core execution engine for reasoning pipelines,
including executors for all pipeline stage types.
"""

from reasoning_mcp.engine.executor import (
    ExecutionContext,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.engine.loop import LoopExecutor
from reasoning_mcp.engine.parallel import ParallelExecutor
from reasoning_mcp.engine.registry import get_executor_for_stage
from reasoning_mcp.engine.sequence import SequenceExecutor
from reasoning_mcp.engine.switch import SwitchExecutor

# These will be implemented in future tasks as needed
# from reasoning_mcp.engine.conditional import ConditionalExecutor
# from reasoning_mcp.engine.method import MethodExecutor

__all__ = [
    "ExecutionContext",
    "PipelineExecutor",
    "StageResult",
    "LoopExecutor",
    "ParallelExecutor",
    "SequenceExecutor",
    "SwitchExecutor",
    "get_executor_for_stage",
]
