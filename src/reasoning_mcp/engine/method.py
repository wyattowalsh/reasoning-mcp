"""Method executor for single reasoning method execution.

This module implements the MethodExecutor which executes a single reasoning
method stage within a pipeline.

Timeout/Cancellation Features:
    - Configurable timeout via ExecutionContext.timeout
    - Graceful handling of asyncio.TimeoutError
    - Proper propagation of asyncio.CancelledError
    - Detailed error messages for timeout scenarios
"""

from __future__ import annotations

import asyncio
import builtins
from asyncio import CancelledError
from datetime import datetime
from typing import TYPE_CHECKING

from reasoning_mcp.engine.executor import (
    DEFAULT_EXECUTION_TIMEOUT,
    ExecutionContext,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import MethodStage, Pipeline
from reasoning_mcp.telemetry.instrumentation import traced_executor

if TYPE_CHECKING:
    from reasoning_mcp.debug.collector import TraceCollector
    from reasoning_mcp.streaming.context import StreamingContext


class MethodExecutor(PipelineExecutor):
    """Executor for single method stages.

    MethodExecutor executes a single reasoning method and returns its results.
    This is the basic building block for pipeline execution.
    """

    def __init__(
        self,
        pipeline: MethodStage,
        streaming_context: StreamingContext | None = None,
        trace_collector: TraceCollector | None = None,
    ):
        """Initialize the method executor.

        Args:
            pipeline: MethodStage configuration to execute
            streaming_context: Optional streaming context for emitting real-time events
            trace_collector: Optional trace collector for debugging and monitoring
        """

        from reasoning_mcp.engine.executor import PipelineExecutor

        PipelineExecutor.__init__(self, streaming_context, trace_collector)
        if not isinstance(pipeline, MethodStage):
            raise TypeError(f"Expected MethodStage, got {type(pipeline)}")
        self.pipeline = pipeline
        self.method_stage = pipeline

    @traced_executor("method.execute")
    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the reasoning method with timeout enforcement.

        Args:
            context: Execution context with session and state

        Returns:
            StageResult with method outputs

        Raises:
            CancelledError: Re-raised to propagate cancellation to parent tasks

        Note:
            - Uses asyncio.timeout() for execution timeout enforcement
            - Timeout is configurable via context.timeout (default: 300s)
            - Stage-specific timeout_seconds takes precedence if set
            - CancelledError is re-raised to allow graceful cancellation propagation
        """
        start_time = datetime.now()
        stage = self.method_stage

        # Determine timeout: stage-specific timeout takes precedence
        timeout_seconds = stage.timeout_seconds or context.timeout or DEFAULT_EXECUTION_TIMEOUT

        # Start tracing span if collector is available
        span_id = None
        if hasattr(self, "_trace_collector") and self._trace_collector:
            from reasoning_mcp.models.debug import SpanStatus

            span_id = self._trace_collector.start_span(
                f"MethodExecutor: {stage.method_id}",
                attributes={
                    "method_id": str(stage.method_id),
                    "stage_id": stage.id,
                    "timeout_seconds": timeout_seconds,
                },
            )

        try:
            async with asyncio.timeout(timeout_seconds):
                input_text = context.input_data.get("input", "")
                method = context.registry.get(stage.method_id)
                if method is None:
                    raise ValueError(f"Method '{stage.method_id}' not registered")

                if not context.registry.is_initialized(stage.method_id):
                    await context.registry.initialize(stage.method_id)

                method_context = {
                    **context.input_data,
                    **context.variables,
                    **stage.metadata,
                    "max_thoughts": stage.max_thoughts,
                    "timeout_seconds": timeout_seconds,
                }
                thought = await method.execute(
                    context.session,
                    input_text,
                    context=method_context,
                    execution_context=context,  # Pass ExecutionContext for sampling
                )

                if thought.id not in context.session.graph.nodes:
                    context.session.add_thought(thought)

            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                thoughts_generated=1,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="completed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[thought.id],
                metrics=metrics,
            )

            # End tracing span on success
            if span_id and self._trace_collector:
                from reasoning_mcp.models.debug import SpanStatus

                self._trace_collector.end_span(span_id, SpanStatus.COMPLETED)

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[thought.id],
                output_data={
                    "content": thought.content,
                    "thought_id": thought.id,
                    "method_id": str(thought.method_id),
                    "confidence": thought.confidence,
                    "stage_name": stage.name,
                },
                trace=trace,
            )

        except builtins.TimeoutError:
            # Handle timeout - create detailed error result
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            error_msg = (
                f"Method execution timed out after {elapsed:.2f}s "
                f"(timeout: {timeout_seconds}s, method: {stage.method_id}, stage: {stage.name})"
            )

            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                errors_count=1,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="timeout",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],
                metrics=metrics,
                error=error_msg,
            )

            # End tracing span on timeout
            if span_id and self._trace_collector:
                from reasoning_mcp.models.debug import SpanStatus

                self._trace_collector.end_span(span_id, SpanStatus.FAILED)

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=False,
                error=error_msg,
                trace=trace,
                metadata={"timeout": True, "timeout_seconds": timeout_seconds},
            )

        except CancelledError:
            # Re-raise CancelledError to propagate cancellation to parent tasks
            # This ensures graceful shutdown when the pipeline is cancelled
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            # End tracing span on cancellation
            if span_id and self._trace_collector:
                from reasoning_mcp.models.debug import SpanStatus

                self._trace_collector.end_span(span_id, SpanStatus.FAILED)

            # Log cancellation for debugging but re-raise
            raise

        except Exception as e:
            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                errors_count=1,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="failed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],
                metrics=metrics,
                error=str(e),
            )

            # End tracing span on failure
            if span_id and self._trace_collector:
                from reasoning_mcp.models.debug import SpanStatus

                self._trace_collector.end_span(span_id, SpanStatus.FAILED)

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=False,
                error=str(e),
                trace=trace,
            )

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the method stage configuration.

        Checks that the stage is a valid MethodStage with a method_id.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(stage, MethodStage):
            errors.append(f"Expected MethodStage, got {type(stage).__name__}")
            return errors

        if not stage.method_id:
            errors.append("MethodStage must have a method_id")

        return errors
