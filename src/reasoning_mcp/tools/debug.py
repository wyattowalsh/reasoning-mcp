"""Debug tools for trace visualization and analysis."""

from typing import Any

from pydantic import BaseModel, Field

from reasoning_mcp.models.debug import TraceLevel

# Input Models


class DebugToolInput(BaseModel):
    """Input for enabling tracing."""

    trace_level: TraceLevel = Field(default=TraceLevel.STANDARD)
    export_format: str = Field(default="json")


class GetTraceInput(BaseModel):
    """Input for retrieving a trace."""

    trace_id: str = Field(..., description="ID of the trace to retrieve")
    format: str = Field(default="json", description="Export format")


class ListTracesInput(BaseModel):
    """Input for listing traces."""

    session_id: str = Field(..., description="Session ID to list traces for")


class AnalyzeTraceInput(BaseModel):
    """Input for analyzing a trace."""

    trace_id: str = Field(..., description="ID of the trace to analyze")


# Tool Functions

# These are placeholder implementations that will work with the storage and export layers
_trace_storage = None  # Will be set by server initialization


def set_trace_storage(storage: Any) -> None:
    """Set the trace storage backend."""
    global _trace_storage
    _trace_storage = storage


async def enable_tracing(input: DebugToolInput) -> dict[str, Any]:
    """Enable tracing for the current session.

    Returns configuration confirmation.
    """
    return {
        "enabled": True,
        "trace_level": input.trace_level.value,
        "export_format": input.export_format,
        "message": f"Tracing enabled at {input.trace_level.value} level",
    }


async def get_trace(input: GetTraceInput) -> dict[str, Any]:
    """Retrieve and export a trace.

    Returns the trace in the requested format.
    """
    if _trace_storage is None:
        return {"error": "Trace storage not initialized"}

    trace = _trace_storage.load(input.trace_id)
    if trace is None:
        return {"error": f"Trace {input.trace_id} not found"}

    # Import here to avoid circular imports
    from reasoning_mcp.debug.export import get_exporter

    try:
        exporter = get_exporter(input.format)
        exported = exporter.export(trace)
        return {"trace_id": input.trace_id, "format": input.format, "content": exported}
    except ValueError as e:
        return {"error": str(e)}


async def list_traces(input: ListTracesInput) -> dict[str, Any]:
    """List traces for a session.

    Returns list of trace IDs and metadata.
    """
    if _trace_storage is None:
        return {"error": "Trace storage not initialized", "traces": []}

    trace_ids = _trace_storage.list_traces(input.session_id)
    traces = []
    for trace_id in trace_ids:
        trace = _trace_storage.load(trace_id)
        if trace:
            traces.append(
                {
                    "trace_id": trace_id,
                    "session_id": trace.session_id,
                    "span_count": len(trace.spans),
                    "error_count": len(trace.errors),
                }
            )

    return {"session_id": input.session_id, "trace_count": len(traces), "traces": traces}


async def analyze_trace(input: AnalyzeTraceInput) -> dict[str, Any]:
    """Analyze a trace for bottlenecks and issues.

    Returns analysis results.
    """
    if _trace_storage is None:
        return {"error": "Trace storage not initialized"}

    trace = _trace_storage.load(input.trace_id)
    if trace is None:
        return {"error": f"Trace {input.trace_id} not found"}

    # Import here to avoid circular imports
    from reasoning_mcp.debug.analyzer import TraceAnalyzer

    analyzer = TraceAnalyzer(trace)

    return {
        "trace_id": input.trace_id,
        "timing": analyzer.get_timing_breakdown(),
        "bottlenecks": analyzer.find_bottlenecks(),
        "errors": [{"type": e.error_type, "message": e.message} for e in analyzer.find_errors()],
        "failed_spans": [s.name for s in analyzer.find_failed_spans()],
        "summary": analyzer.generate_summary(),
    }
