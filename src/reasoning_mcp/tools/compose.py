"""Pipeline composition and execution tool.

This module provides the compose() function for orchestrating complex reasoning
workflows by executing pipelines of reasoning methods with various control flows.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import structlog

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.engine.registry import get_executor_for_stage
from reasoning_mcp.models.pipeline import (
    Pipeline,
    PipelineTrace,
    StageMetrics,
    StageTrace,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.tools import ComposeOutput
from reasoning_mcp.registry import MethodRegistry

logger = structlog.get_logger(__name__)


async def compose(
    pipeline: Pipeline,
    input: str,
    session_id: str | None = None,
    registry: MethodRegistry | None = None,
) -> ComposeOutput:
    """Execute a pipeline of reasoning methods.

    This function orchestrates the execution of a complete reasoning pipeline,
    handling sequential, parallel, conditional, loop, and switch control flows.
    It tracks execution progress, manages errors, and produces a comprehensive
    trace of the execution.

    Args:
        pipeline: The Pipeline to execute (from reasoning_mcp.models.pipeline).
            Can be any pipeline stage type: MethodStage, SequencePipeline,
            ParallelPipeline, ConditionalPipeline, LoopPipeline, or SwitchPipeline.
        input: The input text/problem to reason about. This is used as the
            initial thought content for the pipeline execution.
        session_id: Optional session ID to use. If None, a new session will
            be created for this pipeline execution.
        registry: Optional MethodRegistry for looking up methods. If None,
            attempts to get it from the app context.

    Returns:
        ComposeOutput containing:
            - session_id: The session used for execution
            - pipeline_id: The ID of the executed pipeline
            - success: Whether the pipeline completed successfully
            - final_thoughts: List of final ThoughtNodes produced
            - trace: Complete PipelineTrace with execution details
            - error: Error message if execution failed

    Examples:
        Execute a simple method stage:
        >>> from reasoning_mcp.models.pipeline import MethodStage
        >>> from reasoning_mcp.models.core import MethodIdentifier
        >>> stage = MethodStage(
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     name="analysis",
        ...     max_thoughts=10
        ... )
        >>> result = await compose(
        ...     pipeline=stage,
        ...     input="What are the ethical implications of AI?"
        ... )
        >>> assert result.success is True
        >>> assert len(result.final_thoughts) > 0

        Execute a sequence pipeline:
        >>> from reasoning_mcp.models.pipeline import SequencePipeline
        >>> sequence = SequencePipeline(
        ...     name="multi_stage_analysis",
        ...     stages=[
        ...         MethodStage(
        ...             method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...             name="initial_analysis"
        ...         ),
        ...         MethodStage(
        ...             method_id=MethodIdentifier.SELF_REFLECTION,
        ...             name="self_critique"
        ...         )
        ...     ]
        ... )
        >>> result = await compose(
        ...     pipeline=sequence,
        ...     input="Analyze this problem step by step"
        ... )

        Execute with existing session:
        >>> result = await compose(
        ...     pipeline=stage,
        ...     input="Continue previous analysis",
        ...     session_id="existing-session-123"
        ... )
    """
    start_time = datetime.now()
    result_session_id = session_id or str(uuid4())
    result_pipeline_id = pipeline.id

    # Get or create registry
    if registry is None:
        try:
            from reasoning_mcp.server import get_app_context

            ctx = get_app_context()
            registry = ctx.registry
        except (ImportError, RuntimeError, AttributeError) as e:
            # Create a minimal registry for standalone use (e.g., when running without server)
            logger.debug("compose_using_standalone_registry", reason=str(e))
            registry = MethodRegistry()

    # Create session
    session = Session(id=result_session_id)
    session.start()

    # Create execution context
    context = ExecutionContext(
        session=session,
        registry=registry,
        input_data={"input": input},
        variables={},
        thought_ids=[],
        metadata={
            "pipeline_id": result_pipeline_id,
            "pipeline_type": str(pipeline.stage_type),
        },
    )

    try:
        # Get the appropriate executor for the pipeline type
        executor = get_executor_for_stage(pipeline)

        # Execute the pipeline
        result: StageResult = await executor.execute(context)

        # Collect final thoughts from the session
        final_thoughts = list(session.graph.nodes.values())

        # Create pipeline trace
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        # Create stage metrics
        metrics = StageMetrics(
            stage_id=result_pipeline_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            thoughts_generated=len(final_thoughts),
            errors_count=0 if result.success else 1,
            retries_count=0,
            metadata={
                "stage_type": str(pipeline.stage_type),
            },
        )

        # Create root trace from the result
        root_trace = StageTrace(
            stage_id=result_pipeline_id,
            stage_type=pipeline.stage_type,
            status="completed" if result.success else "failed",
            input_thought_ids=[],
            output_thought_ids=result.output_thought_ids,
            metrics=metrics,
            error=result.error,
            children=[result.trace] if result.trace else [],
            metadata=result.metadata,
        )

        # Create pipeline trace
        trace = PipelineTrace(
            pipeline_id=result_pipeline_id,
            session_id=result_session_id,
            started_at=start_time,
            completed_at=end_time,
            status="completed" if result.success else "failed",
            root_trace=root_trace,
            metadata={
                "duration_seconds": duration_seconds,
                "thoughts_generated": len(final_thoughts),
                "pipeline_type": str(pipeline.stage_type),
            },
        )

        # Complete or fail the session
        if result.success:
            session.complete()
        else:
            session.fail(result.error or "Pipeline execution failed")

        return ComposeOutput(
            session_id=result_session_id,
            pipeline_id=result_pipeline_id,
            success=result.success,
            final_thoughts=final_thoughts,
            trace=trace,
            error=result.error,
        )

    except Exception as e:
        # Handle unexpected errors
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        # Mark session as failed
        session.fail(str(e))

        # Create error trace
        error_trace = PipelineTrace(
            pipeline_id=result_pipeline_id,
            session_id=result_session_id,
            started_at=start_time,
            completed_at=end_time,
            status="failed",
            root_trace=None,
            metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_seconds": duration_seconds,
            },
        )

        return ComposeOutput(
            session_id=result_session_id,
            pipeline_id=result_pipeline_id,
            success=False,
            final_thoughts=[],
            trace=error_trace,
            error=str(e),
        )


async def compose_background(
    pipeline: Pipeline,
    input: str,
    session_id: str | None = None,
    registry: MethodRegistry | None = None,
) -> str:
    """Execute a pipeline in the background and return the session ID.

    This is a convenience function that wraps compose() for use in
    background task scenarios. It returns immediately with the session ID,
    allowing the caller to check status asynchronously.

    Args:
        pipeline: The Pipeline to execute
        input: The input text/problem to reason about
        session_id: Optional session ID to use
        registry: Optional MethodRegistry for looking up methods

    Returns:
        The session ID that can be used to track the pipeline execution

    Note:
        In the current implementation, this function executes synchronously
        and waits for completion. In a full implementation, this would
        submit the task to a background worker and return immediately.
    """
    result_session_id = session_id or str(uuid4())

    # For now, execute synchronously and return the session ID
    # In a full implementation, this would submit to a background worker
    await compose(
        pipeline=pipeline,
        input=input,
        session_id=result_session_id,
        registry=registry,
    )

    return result_session_id


__all__ = ["compose", "compose_background"]
