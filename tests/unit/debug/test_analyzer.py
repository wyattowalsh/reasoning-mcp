from datetime import datetime, timedelta

import pytest

from reasoning_mcp.debug.analyzer import TraceAnalyzer
from reasoning_mcp.models.debug import (
    SpanStatus,
    Trace,
    TraceError,
    TraceSpan,
    TraceStep,
    TraceStepType,
)


@pytest.fixture
def simple_trace():
    now = datetime.now()
    root_span = TraceSpan(
        span_id="root",
        parent_id=None,
        name="root_operation",
        start_time=now,
        end_time=now + timedelta(seconds=2),
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


@pytest.fixture
def complex_trace():
    now = datetime.now()
    root_span = TraceSpan(
        span_id="root",
        parent_id=None,
        name="root_operation",
        start_time=now,
        end_time=now + timedelta(seconds=5),
        status=SpanStatus.COMPLETED,
        attributes={},
    )
    child_span = TraceSpan(
        span_id="child",
        parent_id="root",
        name="child_operation",
        start_time=now + timedelta(seconds=0.5),
        end_time=now + timedelta(seconds=3),
        status=SpanStatus.COMPLETED,
        attributes={},
    )
    failed_span = TraceSpan(
        span_id="failed",
        parent_id="root",
        name="failed_operation",
        start_time=now + timedelta(seconds=3),
        end_time=now + timedelta(seconds=4),
        status=SpanStatus.FAILED,
        attributes={},
    )
    step = TraceStep(
        step_id="step-1",
        span_id="root",
        step_type=TraceStepType.THOUGHT,
        timestamp=now + timedelta(seconds=1),
        content="thinking",
        metadata={},
    )
    error = TraceError(
        error_id="error-1",
        span_id="failed",
        error_type="ValueError",
        message="Something went wrong",
        stack_trace="...",
        recoverable=False,
        recovery_action=None,
    )
    return Trace(
        trace_id="trace-2",
        session_id="session-1",
        root_span=root_span,
        spans=[root_span, child_span, failed_span],
        steps=[step],
        decisions=[],
        errors=[error],
    )


class TestFindBottlenecks:
    def test_finds_slow_spans(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        bottlenecks = analyzer.find_bottlenecks(threshold_seconds=2.0)
        assert len(bottlenecks) >= 1
        assert any(b["name"] == "root_operation" for b in bottlenecks)

    def test_empty_when_all_fast(self, simple_trace):
        analyzer = TraceAnalyzer(simple_trace)
        bottlenecks = analyzer.find_bottlenecks(threshold_seconds=10.0)
        assert len(bottlenecks) == 0


class TestFindErrors:
    def test_finds_errors(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        errors = analyzer.find_errors()
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"

    def test_empty_when_no_errors(self, simple_trace):
        analyzer = TraceAnalyzer(simple_trace)
        errors = analyzer.find_errors()
        assert len(errors) == 0


class TestFindFailedSpans:
    def test_finds_failed_spans(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        failed = analyzer.find_failed_spans()
        assert len(failed) == 1
        assert failed[0].name == "failed_operation"

    def test_empty_when_all_succeed(self, simple_trace):
        analyzer = TraceAnalyzer(simple_trace)
        failed = analyzer.find_failed_spans()
        assert len(failed) == 0


class TestGetTimingBreakdown:
    def test_returns_total_duration(self, simple_trace):
        analyzer = TraceAnalyzer(simple_trace)
        timing = analyzer.get_timing_breakdown()
        assert timing["total_duration_seconds"] == pytest.approx(2.0, rel=0.1)

    def test_counts_spans_and_steps(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        timing = analyzer.get_timing_breakdown()
        assert timing["span_count"] == 3
        assert timing["step_count"] == 1
        assert timing["error_count"] == 1


class TestGetSpanHierarchy:
    def test_builds_tree(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        tree = analyzer.get_span_hierarchy()
        assert tree["name"] == "root_operation"
        assert len(tree["children"]) == 2


class TestGenerateSummary:
    def test_generates_summary_string(self, simple_trace):
        analyzer = TraceAnalyzer(simple_trace)
        summary = analyzer.generate_summary()
        assert "Trace Summary:" in summary
        assert "trace-1" in summary
        assert "SUCCESS" in summary

    def test_includes_errors_in_summary(self, complex_trace):
        analyzer = TraceAnalyzer(complex_trace)
        summary = analyzer.generate_summary()
        assert "Errors:" in summary
        assert "ValueError" in summary
        assert "FAILED" in summary
