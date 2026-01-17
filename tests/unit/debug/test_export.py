import json
from datetime import datetime

import pytest

from reasoning_mcp.debug.export import (
    HTMLExporter,
    JSONExporter,
    MermaidExporter,
    OpenTelemetryExporter,
    get_exporter,
)
from reasoning_mcp.models.debug import SpanStatus, Trace, TraceSpan, TraceStep, TraceStepType


@pytest.fixture
def sample_trace():
    root_span = TraceSpan(
        span_id="span-1",
        parent_id=None,
        name="root",
        start_time=datetime.now(),
        end_time=datetime.now(),
        status=SpanStatus.COMPLETED,
        attributes={"method": "test"},
    )
    step = TraceStep(
        step_id="step-1",
        span_id="span-1",
        step_type=TraceStepType.THOUGHT,
        timestamp=datetime.now(),
        content="Test thought",
        metadata={},
    )
    return Trace(
        trace_id="trace-1",
        session_id="session-1",
        root_span=root_span,
        spans=[root_span],
        steps=[step],
        decisions=[],
        errors=[],
    )


class TestJSONExporter:
    def test_exports_valid_json(self, sample_trace):
        exporter = JSONExporter()
        output = exporter.export(sample_trace)
        data = json.loads(output)
        assert data["trace_id"] == "trace-1"

    def test_respects_indent(self, sample_trace):
        exporter = JSONExporter(indent=4)
        output = exporter.export(sample_trace)
        assert "    " in output  # 4-space indent


class TestMermaidExporter:
    def test_exports_sequence_diagram(self, sample_trace):
        exporter = MermaidExporter()
        output = exporter.export(sample_trace)
        assert "sequenceDiagram" in output

    def test_includes_span_names(self, sample_trace):
        exporter = MermaidExporter()
        output = exporter.export(sample_trace)
        assert "root" in output


class TestHTMLExporter:
    def test_exports_valid_html(self, sample_trace):
        exporter = HTMLExporter()
        output = exporter.export(sample_trace)
        assert "<!DOCTYPE html>" in output
        assert "</html>" in output

    def test_includes_trace_id(self, sample_trace):
        exporter = HTMLExporter()
        output = exporter.export(sample_trace)
        assert "trace-1" in output

    def test_includes_spans(self, sample_trace):
        exporter = HTMLExporter()
        output = exporter.export(sample_trace)
        assert "root" in output

    def test_escapes_html(self):
        root_span = TraceSpan(
            span_id="span-1",
            parent_id=None,
            name="<script>alert('xss')</script>",
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=SpanStatus.COMPLETED,
            attributes={},
        )
        trace = Trace(
            trace_id="trace-1",
            session_id="session-1",
            root_span=root_span,
            spans=[root_span],
            steps=[],
            decisions=[],
            errors=[],
        )
        exporter = HTMLExporter()
        output = exporter.export(trace)
        assert "<script>" not in output
        assert "&lt;script&gt;" in output


class TestOpenTelemetryExporter:
    def test_exports_otlp_format(self, sample_trace):
        exporter = OpenTelemetryExporter()
        output = exporter.export(sample_trace)
        data = json.loads(output)
        assert "resourceSpans" in data

    def test_includes_spans(self, sample_trace):
        exporter = OpenTelemetryExporter()
        output = exporter.export(sample_trace)
        data = json.loads(output)
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "root"


class TestGetExporter:
    def test_json_exporter(self):
        exporter = get_exporter("json")
        assert isinstance(exporter, JSONExporter)

    def test_mermaid_exporter(self):
        exporter = get_exporter("mermaid")
        assert isinstance(exporter, MermaidExporter)

    def test_html_exporter(self):
        exporter = get_exporter("html")
        assert isinstance(exporter, HTMLExporter)

    def test_otlp_exporter(self):
        exporter = get_exporter("otlp")
        assert isinstance(exporter, OpenTelemetryExporter)

    def test_unknown_format(self):
        with pytest.raises(ValueError):
            get_exporter("unknown")
