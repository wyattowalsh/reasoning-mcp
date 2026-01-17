"""Trace analysis utilities for reasoning-mcp."""

from typing import Any

from reasoning_mcp.models.debug import SpanStatus, Trace, TraceError, TraceSpan


class TraceAnalyzer:
    """Analyzes traces for bottlenecks, errors, and timing."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def find_bottlenecks(self, threshold_seconds: float = 1.0) -> list[dict[str, Any]]:
        """Find spans that took longer than threshold."""
        bottlenecks = []
        for span in self.trace.spans:
            if span.end_time and span.start_time:
                duration = (span.end_time - span.start_time).total_seconds()
                if duration > threshold_seconds:
                    bottlenecks.append(
                        {
                            "span_id": span.span_id,
                            "name": span.name,
                            "duration_seconds": duration,
                            "threshold": threshold_seconds,
                        }
                    )
        return sorted(bottlenecks, key=lambda x: x["duration_seconds"], reverse=True)

    def find_errors(self) -> list[TraceError]:
        """Find all errors in the trace."""
        return list(self.trace.errors)

    def find_failed_spans(self) -> list[TraceSpan]:
        """Find all spans with failed or cancelled status."""
        return [
            span
            for span in self.trace.spans
            if span.status in (SpanStatus.FAILED, SpanStatus.CANCELLED)
        ]

    def get_timing_breakdown(self) -> dict[str, Any]:
        """Get timing breakdown of the trace."""
        total_duration = 0.0
        span_durations: dict[str, float] = {}

        if self.trace.root_span.end_time and self.trace.root_span.start_time:
            total_duration = (
                self.trace.root_span.end_time - self.trace.root_span.start_time
            ).total_seconds()

        for span in self.trace.spans:
            if span.end_time and span.start_time:
                duration = (span.end_time - span.start_time).total_seconds()
                span_durations[span.name] = duration

        return {
            "total_duration_seconds": total_duration,
            "span_count": len(self.trace.spans),
            "step_count": len(self.trace.steps),
            "error_count": len(self.trace.errors),
            "decision_count": len(self.trace.decisions),
            "span_durations": span_durations,
        }

    def get_span_hierarchy(self) -> dict[str, Any]:
        """Get hierarchical view of spans."""
        span_map = {span.span_id: span for span in self.trace.spans}
        children: dict[str | None, list[str]] = {}

        for span in self.trace.spans:
            parent = span.parent_id
            if parent not in children:
                children[parent] = []
            children[parent].append(span.span_id)

        def build_tree(span_id: str) -> dict[str, Any]:
            span = span_map[span_id]
            duration = 0.0
            if span.end_time and span.start_time:
                duration = (span.end_time - span.start_time).total_seconds()

            node: dict[str, Any] = {
                "span_id": span_id,
                "name": span.name,
                "status": span.status.value,
                "duration_seconds": duration,
                "children": [],
            }

            for child_id in children.get(span_id, []):
                child_list: list[Any] = node["children"]
                child_list.append(build_tree(child_id))

            return node

        # Build from root
        root_spans = children.get(None, [])
        if root_spans:
            return build_tree(root_spans[0])
        return {}

    def generate_summary(self) -> str:
        """Generate human-readable trace summary."""
        timing = self.get_timing_breakdown()
        errors = self.find_errors()
        failed_spans = self.find_failed_spans()
        bottlenecks = self.find_bottlenecks()

        lines = [
            f"Trace Summary: {self.trace.trace_id}",
            f"Session: {self.trace.session_id}",
            "=" * 50,
            "",
            "Execution Overview:",
            f"  Total Duration: {timing['total_duration_seconds']:.3f}s",
            f"  Spans: {timing['span_count']}",
            f"  Steps: {timing['step_count']}",
            f"  Decisions: {timing['decision_count']}",
            f"  Errors: {timing['error_count']}",
            "",
        ]

        if bottlenecks:
            lines.append("Bottlenecks (>1s):")
            for b in bottlenecks[:5]:  # Top 5
                lines.append(f"  - {b['name']}: {b['duration_seconds']:.3f}s")
            lines.append("")

        if failed_spans:
            lines.append("Failed Spans:")
            for span in failed_spans:
                lines.append(f"  - {span.name} ({span.status.value})")
            lines.append("")

        if errors:
            lines.append("Errors:")
            for error in errors:
                lines.append(f"  - [{error.error_type}] {error.message[:50]}")
            lines.append("")

        # Status
        if not errors and not failed_spans:
            lines.append("Status: SUCCESS")
        elif errors:
            lines.append(f"Status: FAILED ({len(errors)} errors)")
        else:
            lines.append("Status: PARTIAL (some spans failed)")

        return "\n".join(lines)
