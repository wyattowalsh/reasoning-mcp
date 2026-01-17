"""Instrumentation decorators and utilities for tracing executors.

This module provides decorators and helpers for adding OpenTelemetry
spans to pipeline executors.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from reasoning_mcp.telemetry.attributes import (
    ATTR_DURATION_MS,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_TYPE,
    ATTR_SESSION_ID,
    ATTR_STAGE_ID,
    ATTR_STAGE_NAME,
    ATTR_STAGE_TYPE,
)
from reasoning_mcp.telemetry.provider import _OTEL_AVAILABLE, get_tracer

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext

if _OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
else:
    trace = None
    Status = None
    StatusCode = None


F = TypeVar("F", bound=Callable[..., Any])


def traced_executor(
    span_name: str,
    *,
    include_stage_attrs: bool = True,
) -> Callable[[F], F]:
    """Decorator for tracing executor methods.

    Creates a span around the decorated async method, automatically
    adding session and stage attributes from the ExecutionContext.

    Args:
        span_name: Name for the span (e.g., "method.execute", "parallel.execute").
        include_stage_attrs: Whether to include stage attributes from self.stage.

    Returns:
        Decorated function with tracing.

    Example:
        @traced_executor("method.execute")
        async def execute(self, context: ExecutionContext) -> StageResult:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(self: Any, context: ExecutionContext, *args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer("reasoning-mcp.engine")

            # Use context manager for span lifecycle
            with tracer.start_as_current_span(span_name) as span:
                start_time = time.perf_counter()

                # Add session context
                if hasattr(context, "session") and context.session:
                    span.set_attribute(ATTR_SESSION_ID, context.session.id)

                # Add stage/pipeline attributes if available
                # Executors may use self.stage or self.pipeline
                if include_stage_attrs:
                    pipeline = None
                    if hasattr(self, "stage") and self.stage:
                        pipeline = self.stage
                    elif hasattr(self, "pipeline") and self.pipeline:
                        pipeline = self.pipeline

                    if pipeline:
                        # Try various attribute names for ID
                        stage_id = getattr(pipeline, "stage_id", None) or getattr(
                            pipeline, "id", None
                        )
                        if stage_id:
                            span.set_attribute(ATTR_STAGE_ID, str(stage_id))

                        # Try various attribute names for type
                        stage_type = getattr(pipeline, "stage_type", None) or getattr(
                            pipeline, "pipeline_type", None
                        )
                        if stage_type:
                            type_value = (
                                stage_type.value
                                if hasattr(stage_type, "value")
                                else str(stage_type)
                            )
                            span.set_attribute(ATTR_STAGE_TYPE, type_value)

                        # Name attribute
                        if hasattr(pipeline, "name") and pipeline.name:
                            span.set_attribute(ATTR_STAGE_NAME, pipeline.name)

                try:
                    result = await func(self, context, *args, **kwargs)

                    # Record success
                    if hasattr(result, "success"):
                        span.set_attribute("success", result.success)
                        if not result.success and hasattr(result, "error"):
                            span.set_attribute(ATTR_ERROR_MESSAGE, str(result.error))
                            set_span_error(span, str(result.error))
                        else:
                            set_span_ok(span)
                    else:
                        set_span_ok(span)

                    return result

                except Exception as e:
                    record_exception(span, e)
                    raise

                finally:
                    # Record duration
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(ATTR_DURATION_MS, duration_ms)

        return wrapper  # type: ignore[return-value]

    return decorator


def set_span_ok(span: Any) -> None:
    """Set span status to OK.

    Args:
        span: The span to update (can be NoOpSpan).
    """
    if _OTEL_AVAILABLE and Status is not None:
        span.set_status(Status(StatusCode.OK))


def set_span_error(span: Any, description: str | None = None) -> None:
    """Set span status to ERROR.

    Args:
        span: The span to update (can be NoOpSpan).
        description: Optional error description.
    """
    if _OTEL_AVAILABLE and Status is not None:
        span.set_status(Status(StatusCode.ERROR, description=description))


def record_exception(span: Any, exception: BaseException) -> None:
    """Record an exception on a span.

    Args:
        span: The span to record on (can be NoOpSpan).
        exception: The exception to record.
    """
    span.record_exception(exception)
    span.set_attribute(ATTR_ERROR_TYPE, type(exception).__name__)
    span.set_attribute(ATTR_ERROR_MESSAGE, str(exception))
    set_span_error(span, str(exception))


def add_span_event(
    span: Any,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Add an event to a span.

    Args:
        span: The span to add event to (can be NoOpSpan).
        name: Event name.
        attributes: Optional event attributes.
    """
    span.add_event(name, attributes=attributes)


__all__ = [
    "traced_executor",
    "set_span_ok",
    "set_span_error",
    "record_exception",
    "add_span_event",
]
