from datetime import datetime

import pytest

from reasoning_mcp.debug.storage import MemoryTraceStorage
from reasoning_mcp.models.debug import SpanStatus, Trace, TraceLevel, TraceSpan
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


@pytest.fixture
def sample_trace():
    root_span = TraceSpan(
        span_id="span-1",
        parent_id=None,
        name="root",
        start_time=datetime.now(),
        end_time=datetime.now(),
        status=SpanStatus.COMPLETED,
        attributes={},
    )
    return Trace(
        trace_id="trace-1",
        session_id="session-1",
        root_span=root_span,
        spans=[root_span],
        steps=[],
        decisions=[],
        errors=[],
    )


class TestDebugToolInput:
    def test_default_values(self):
        input = DebugToolInput()
        assert input.trace_level == TraceLevel.STANDARD
        assert input.export_format == "json"

    def test_custom_values(self):
        input = DebugToolInput(trace_level=TraceLevel.VERBOSE, export_format="html")
        assert input.trace_level == TraceLevel.VERBOSE
        assert input.export_format == "html"


class TestEnableTracing:
    async def test_returns_confirmation(self):
        input = DebugToolInput()
        result = await enable_tracing(input)
        assert result["enabled"] is True
        assert result["trace_level"] == "standard"


class TestGetTrace:
    async def test_returns_trace(self, sample_trace):
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        set_trace_storage(storage)

        input = GetTraceInput(trace_id="trace-1")
        result = await get_trace(input)
        assert result["trace_id"] == "trace-1"
        assert "content" in result

    async def test_returns_error_for_missing_trace(self):
        storage = MemoryTraceStorage()
        set_trace_storage(storage)

        input = GetTraceInput(trace_id="nonexistent")
        result = await get_trace(input)
        assert "error" in result


class TestListTraces:
    async def test_lists_traces(self, sample_trace):
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        set_trace_storage(storage)

        input = ListTracesInput(session_id="session-1")
        result = await list_traces(input)
        assert result["trace_count"] == 1
        assert result["traces"][0]["trace_id"] == "trace-1"


class TestAnalyzeTrace:
    async def test_analyzes_trace(self, sample_trace):
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        set_trace_storage(storage)

        input = AnalyzeTraceInput(trace_id="trace-1")
        result = await analyze_trace(input)
        assert result["trace_id"] == "trace-1"
        assert "timing" in result
        assert "summary" in result
