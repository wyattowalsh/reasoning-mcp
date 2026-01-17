"""End-to-end tests for the debugging feature."""

from reasoning_mcp.debug.analyzer import TraceAnalyzer
from reasoning_mcp.debug.collector import TraceCollector
from reasoning_mcp.debug.export import get_exporter
from reasoning_mcp.debug.storage import MemoryTraceStorage, get_storage
from reasoning_mcp.models.debug import SpanStatus, TraceStepType


class TestDebugEndToEnd:
    """End-to-end tests for the complete debug flow."""

    def test_full_tracing_flow(self):
        """Test complete flow: collect -> store -> retrieve -> analyze -> export."""
        # 1. Create collector and record trace data
        collector = TraceCollector(session_id="e2e-test-session")

        root_span = collector.start_span("root_operation", attributes={"test": True})
        collector.add_step(TraceStepType.INPUT, "Processing input")

        child_span = collector.start_span("child_operation")
        collector.add_step(TraceStepType.THOUGHT, "Analyzing data")
        collector.checkpoint("midpoint", {"progress": 50})
        collector.end_span(child_span)

        collector.add_step(TraceStepType.OUTPUT, "Completed processing")
        collector.end_span(root_span)

        # 2. Get trace
        trace = collector.get_trace()
        assert trace.trace_id == collector.trace_id
        assert len(trace.spans) == 2
        assert len(trace.steps) == 4  # INPUT, THOUGHT, CHECKPOINT, OUTPUT

        # 3. Store trace
        storage = MemoryTraceStorage()
        storage.save(trace)

        # 4. Retrieve trace
        loaded = storage.load(trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id

        # 5. Analyze trace
        analyzer = TraceAnalyzer(loaded)
        timing = analyzer.get_timing_breakdown()
        assert timing["span_count"] == 2
        assert timing["step_count"] == 4

        summary = analyzer.generate_summary()
        assert "SUCCESS" in summary

        # 6. Export trace
        for format in ["json", "html", "mermaid"]:
            exporter = get_exporter(format)
            output = exporter.export(loaded)
            assert len(output) > 0

    def test_error_tracing_flow(self):
        """Test tracing when errors occur."""
        collector = TraceCollector()

        span_id = collector.start_span("error_operation")
        collector.add_step(TraceStepType.INPUT, "Starting")

        # Record an error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            collector.record_error(e, recoverable=False)

        collector.end_span(span_id, SpanStatus.FAILED)

        trace = collector.get_trace()

        # Analyze
        analyzer = TraceAnalyzer(trace)
        errors = analyzer.find_errors()
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"

        failed = analyzer.find_failed_spans()
        assert len(failed) == 1

        summary = analyzer.generate_summary()
        assert "FAILED" in summary

    def test_storage_backends(self, tmp_path):
        """Test different storage backends."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        trace = collector.get_trace()

        # Memory storage
        mem_storage = get_storage("memory")
        mem_storage.save(trace)
        assert mem_storage.load(trace.trace_id) is not None

        # File storage
        file_storage = get_storage("file", tmp_path / "traces")
        file_storage.save(trace)
        assert file_storage.load(trace.trace_id) is not None

        # SQLite storage
        sqlite_storage = get_storage("sqlite", tmp_path / "traces.db")
        sqlite_storage.save(trace)
        assert sqlite_storage.load(trace.trace_id) is not None
