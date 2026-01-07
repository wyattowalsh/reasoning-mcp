"""Executor registry for dispatching pipeline stages to appropriate executors.

This module provides utilities for mapping pipeline stage types to their
corresponding executor implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reasoning_mcp.engine.executor import PipelineExecutor
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import (
    ConditionalPipeline,
    LoopPipeline,
    MethodStage,
    ParallelPipeline,
    Pipeline,
    SequencePipeline,
    SwitchPipeline,
)

if TYPE_CHECKING:
    pass  # For future executor imports


def get_executor_for_stage(stage: Pipeline) -> PipelineExecutor:
    """Get the appropriate executor for a pipeline stage.

    Dispatches to stage-specific executors based on the stage type.
    This is the main entry point for pipeline execution.

    Args:
        stage: Pipeline stage to execute

    Returns:
        Executor instance configured for the stage

    Raises:
        ValueError: If stage type is not recognized or executor not available
    """
    from reasoning_mcp.engine.switch import SwitchExecutor

    # Switch stage
    if isinstance(stage, SwitchPipeline):
        return SwitchExecutor(stage)

    # Method stage
    if isinstance(stage, MethodStage):
        # Import here to avoid circular dependency
        try:
            from reasoning_mcp.engine.method import MethodExecutor

            return MethodExecutor(stage)
        except ImportError:
            raise ValueError(
                "MethodExecutor not yet implemented. "
                "This will be available in TASK-090."
            )

    # Sequence stage
    if isinstance(stage, SequencePipeline):
        try:
            from reasoning_mcp.engine.sequence import SequenceExecutor

            return SequenceExecutor(stage)
        except ImportError:
            raise ValueError(
                "SequenceExecutor not yet implemented. "
                "This will be available in TASK-091."
            )

    # Parallel stage
    if isinstance(stage, ParallelPipeline):
        try:
            from reasoning_mcp.engine.parallel import ParallelExecutor

            return ParallelExecutor(stage)
        except ImportError:
            raise ValueError(
                "ParallelExecutor not yet implemented. "
                "This will be available in TASK-092."
            )

    # Conditional stage
    if isinstance(stage, ConditionalPipeline):
        try:
            from reasoning_mcp.engine.conditional import ConditionalExecutor

            return ConditionalExecutor(stage)
        except ImportError:
            raise ValueError(
                "ConditionalExecutor not yet implemented. "
                "This will be available in TASK-093."
            )

    # Loop stage
    if isinstance(stage, LoopPipeline):
        try:
            from reasoning_mcp.engine.loop import LoopExecutor

            return LoopExecutor(stage)
        except ImportError:
            raise ValueError(
                "LoopExecutor not yet implemented. "
                "This will be available in TASK-094."
            )

    # Unknown stage type
    raise ValueError(
        f"Unknown pipeline stage type: {type(stage).__name__}. "
        f"Stage type: {getattr(stage, 'stage_type', 'unknown')}"
    )


__all__ = [
    "get_executor_for_stage",
]
