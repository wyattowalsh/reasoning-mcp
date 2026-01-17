"""Tests for streaming metrics module."""

from reasoning_mcp.streaming.metrics import MetricsCollector, StreamingMetrics


class TestStreamingMetrics:
    """Tests for StreamingMetrics model."""

    def test_metrics_model(self):
        """Test StreamingMetrics model defaults."""
        metrics = StreamingMetrics()
        assert metrics.events_emitted == 0
        assert metrics.events_dropped == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.backpressure_events == 0

    def test_metrics_model_custom(self):
        """Test StreamingMetrics with custom values."""
        metrics = StreamingMetrics(
            events_emitted=100,
            events_dropped=5,
            avg_latency_ms=2.5,
            backpressure_events=10,
        )
        assert metrics.events_emitted == 100
        assert metrics.events_dropped == 5


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_metrics_collector(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert metrics.events_emitted == 0
        assert metrics.events_dropped == 0

    def test_record_emit(self):
        """Test recording emits."""
        collector = MetricsCollector()
        collector.record_emit(1.5)
        collector.record_emit(2.5)

        metrics = collector.get_metrics()
        assert metrics.events_emitted == 2
        assert metrics.avg_latency_ms == 2.0  # (1.5 + 2.5) / 2

    def test_record_drop(self):
        """Test recording dropped events."""
        collector = MetricsCollector()
        collector.record_drop()
        collector.record_drop()

        metrics = collector.get_metrics()
        assert metrics.events_dropped == 2

    def test_record_backpressure(self):
        """Test recording backpressure events."""
        collector = MetricsCollector()
        collector.record_backpressure()

        metrics = collector.get_metrics()
        assert metrics.backpressure_events == 1

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.record_emit(1.0)
        collector.record_drop()
        collector.record_backpressure()

        collector.reset()
        metrics = collector.get_metrics()
        assert metrics.events_emitted == 0
        assert metrics.events_dropped == 0
        assert metrics.backpressure_events == 0
