"""Trace export functionality for reasoning-mcp.

This module provides exporters to convert trace data into various formats
including JSON, Mermaid diagrams, HTML visualizations, and OpenTelemetry format.
"""

import html as html_module
import json
from typing import Protocol

from reasoning_mcp.models.debug import SpanStatus, Trace


class TraceExporter(Protocol):
    """Protocol for trace exporters."""

    def export(self, trace: Trace) -> str:
        """Export trace to string format."""
        ...


class JSONExporter:
    """Exports trace as formatted JSON."""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def export(self, trace: Trace) -> str:
        return trace.model_dump_json(indent=self.indent)


class MermaidExporter:
    """Exports trace as Mermaid sequence diagram."""

    def export(self, trace: Trace) -> str:
        lines = ["sequenceDiagram"]

        # Build span hierarchy
        span_map = {s.span_id: s for s in trace.spans}

        for span in trace.spans:
            parent_name = "Client"
            if span.parent_id and span.parent_id in span_map:
                parent_name = span_map[span.parent_id].name

            lines.append(f"    {parent_name}->>+{span.name}: start")

            # Add steps for this span
            span_steps = [s for s in trace.steps if s.span_id == span.span_id]
            for step in span_steps:
                lines.append(
                    f"    Note right of {span.name}: {step.step_type.value}: {step.content[:30]}"
                )

            status_symbol = "+" if span.status == SpanStatus.COMPLETED else "x"
            lines.append(f"    {span.name}->>{status_symbol}{parent_name}: {span.status.value}")

        return "\n".join(lines)


class HTMLExporter:
    """Exports trace as interactive HTML visualization."""

    def export(self, trace: Trace) -> str:
        spans_html = self._render_spans(trace)
        steps_html = self._render_steps(trace)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Trace: {html_module.escape(trace.trace_id)}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 20px; }}
        .span {{ border: 1px solid #ccc; padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .span.completed {{ border-color: #4caf50; background: #e8f5e9; }}
        .span.failed {{ border-color: #f44336; background: #ffebee; }}
        .step {{ padding: 5px 10px; margin: 2px 0; font-size: 0.9em; }}
        .step-thought {{ background: #e3f2fd; }}
        .step-decision {{ background: #fff3e0; }}
        .step-error {{ background: #ffebee; }}
        h1, h2 {{ color: #333; }}
        .meta {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Trace: {html_module.escape(trace.trace_id)}</h1>
    <p class="meta">Session: {html_module.escape(trace.session_id)}</p>

    <h2>Spans</h2>
    {spans_html}

    <h2>Steps</h2>
    {steps_html}
</body>
</html>"""

    def _render_spans(self, trace: Trace) -> str:
        html_parts = []
        for span in trace.spans:
            status_class = (
                span.status.value
                if isinstance(span.status.value, str)
                else span.status.name.lower()
            )
            duration = ""
            if span.end_time and span.start_time:
                dur = (span.end_time - span.start_time).total_seconds()
                duration = f" ({dur:.3f}s)"
            html_parts.append(
                f"""<div class="span {status_class}">
                <strong>{html_module.escape(span.name)}</strong>{duration}
                <br><small>ID: {html_module.escape(span.span_id)}</small>
            </div>"""
            )
        return "\n".join(html_parts)

    def _render_steps(self, trace: Trace) -> str:
        html_parts = []
        for step in trace.steps:
            step_class = f"step-{step.step_type.value}"
            html_parts.append(
                f"""<div class="step {step_class}">
                <strong>{step.step_type.value}</strong>: {html_module.escape(step.content[:200])}
            </div>"""
            )
        return "\n".join(html_parts)


class OpenTelemetryExporter:
    """Exports trace in OpenTelemetry-compatible JSON format."""

    def export(self, trace: Trace) -> str:
        # Convert to OTLP-like format
        spans = []
        for span in trace.spans:
            otlp_span = {
                "traceId": trace.trace_id,
                "spanId": span.span_id,
                "parentSpanId": span.parent_id,
                "name": span.name,
                "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
                "endTimeUnixNano": int(span.end_time.timestamp() * 1e9) if span.end_time else None,
                "status": {"code": 1 if span.status == SpanStatus.COMPLETED else 2},
                "attributes": [
                    {"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()
                ],
            }
            spans.append(otlp_span)

        return json.dumps(
            {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": "session.id", "value": {"stringValue": trace.session_id}}
                            ]
                        },
                        "scopeSpans": [{"spans": spans}],
                    }
                ]
            },
            indent=2,
        )


def get_exporter(format: str = "json") -> TraceExporter:
    """Factory function to create exporters."""
    exporters: dict[str, type[TraceExporter]] = {
        "json": JSONExporter,
        "mermaid": MermaidExporter,
        "html": HTMLExporter,
        "otlp": OpenTelemetryExporter,
    }
    if format not in exporters:
        raise ValueError(f"Unknown export format: {format}. Available: {list(exporters.keys())}")
    exporter_class = exporters[format]
    return exporter_class()
