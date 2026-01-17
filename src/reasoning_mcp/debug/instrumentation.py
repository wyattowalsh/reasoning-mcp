"""Instrumentation for automatic tracing of reasoning execution."""

from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from reasoning_mcp.models.debug import SpanStatus, TraceLevel, TraceStepType

F = TypeVar("F", bound=Callable[..., Any])


class InstrumentationConfig(BaseModel):
    """Configuration for instrumentation."""

    trace_level: TraceLevel = Field(default=TraceLevel.STANDARD)
    capture_inputs: bool = Field(default=True)
    capture_outputs: bool = Field(default=True)
    capture_thoughts: bool = Field(default=True)
    max_content_length: int = Field(default=1000)


def instrument(
    name: str | None = None, capture_args: bool = True, capture_result: bool = True
) -> Callable[[F], F]:
    """Decorator to instrument a function with tracing.

    Args:
        name: Custom span name (defaults to function name)
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get collector from first arg if it's an instrumented object
            collector = None
            if args and hasattr(args[0], "_trace_collector"):
                collector = args[0]._trace_collector

            if collector is None:
                # No collector, just run the function
                return func(*args, **kwargs)

            # Start span
            attributes = {}
            if capture_args:
                attributes["args"] = str(args[1:])[:500]  # Skip self
                attributes["kwargs"] = str(kwargs)[:500]

            span_id = collector.start_span(span_name, attributes=attributes)

            try:
                result = func(*args, **kwargs)

                if capture_result:
                    collector.add_step(TraceStepType.OUTPUT, f"Result: {str(result)[:500]}")

                collector.end_span(span_id)
                return result
            except Exception as e:
                collector.record_error(e)
                collector.end_span(span_id, SpanStatus.FAILED)
                raise

        return wrapper  # type: ignore

    return decorator


@runtime_checkable
class HasTraceCollector(Protocol):
    """Protocol for objects that have a trace collector."""

    _trace_collector: Any


class InstrumentedExecutor:
    """Base class for adding tracing to executors."""

    def __init__(self, trace_collector: Any = None) -> None:
        self._trace_collector = trace_collector

    def set_trace_collector(self, collector: Any) -> None:
        """Set the trace collector."""
        self._trace_collector = collector

    def _start_span(self, name: str, **attributes: Any) -> str | None:
        """Start a span if collector is available."""
        if self._trace_collector is None:
            return None
        result: str | None = self._trace_collector.start_span(name, attributes=attributes)
        return result

    def _end_span(self, span_id: str | None, status: Any = None) -> None:
        """End a span if collector is available."""
        if self._trace_collector is None or span_id is None:
            return
        self._trace_collector.end_span(span_id, status or SpanStatus.COMPLETED)

    def _add_step(self, step_type: Any, content: str, **metadata: Any) -> None:
        """Add a step if collector is available."""
        if self._trace_collector is None:
            return
        self._trace_collector.add_step(step_type, content, metadata=metadata)


class InstrumentedMethodExecutor(InstrumentedExecutor):
    """Wraps method execution with tracing."""

    def __init__(self, method_name: str, trace_collector: Any = None) -> None:
        super().__init__(trace_collector)
        self.method_name = method_name

    async def execute_with_tracing(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function with tracing."""
        span_id = self._start_span(
            f"method:{self.method_name}", args=str(args)[:200], kwargs=str(kwargs)[:200]
        )

        try:
            self._add_step(TraceStepType.METHOD_START, f"Starting {self.method_name}")

            result = await func(*args, **kwargs) if callable(func) else func

            self._add_step(TraceStepType.METHOD_END, f"Completed {self.method_name}")
            self._end_span(span_id)

            return result
        except Exception as e:
            self._add_step(TraceStepType.ERROR, str(e))
            self._end_span(span_id, SpanStatus.FAILED)
            raise


class InstrumentedSequenceExecutor(InstrumentedExecutor):
    """Wraps sequence execution with tracing."""

    def __init__(self, sequence_name: str = "sequence", trace_collector: Any = None) -> None:
        super().__init__(trace_collector)
        self.sequence_name = sequence_name

    async def execute_sequence_with_tracing(
        self, steps: list[Callable[..., Any]], *args: Any, **kwargs: Any
    ) -> list[Any]:
        """Execute a sequence of steps with tracing."""
        parent_span_id = self._start_span(f"sequence:{self.sequence_name}", step_count=len(steps))

        results = []
        try:
            for i, step in enumerate(steps):
                step_span_id = self._start_span(f"step:{i}", step_index=i)

                try:
                    result = await step(*args, **kwargs) if callable(step) else step
                    results.append(result)
                    self._end_span(step_span_id)
                except Exception:
                    self._end_span(step_span_id, SpanStatus.FAILED)
                    raise

            self._end_span(parent_span_id)
            return results
        except Exception:
            self._end_span(parent_span_id, SpanStatus.FAILED)
            raise
