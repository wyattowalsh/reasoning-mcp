"""Metrics collection for streaming operations."""

from pydantic import BaseModel


class StreamingMetrics(BaseModel):
    """Metrics about streaming performance."""

    events_emitted: int = 0
    events_dropped: int = 0
    avg_latency_ms: float = 0.0
    backpressure_events: int = 0


class MetricsCollector:
    """Collector for streaming metrics.

    Thread-safe metrics collection for monitoring streaming
    performance and health.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._events_emitted: int = 0
        self._events_dropped: int = 0
        self._total_latency_ms: float = 0.0
        self._backpressure_events: int = 0

    def record_emit(self, latency_ms: float) -> None:
        """Record a successful event emission.

        Args:
            latency_ms: Time taken to emit the event in milliseconds.
        """
        self._events_emitted += 1
        self._total_latency_ms += latency_ms

    def record_drop(self) -> None:
        """Record a dropped event due to backpressure."""
        self._events_dropped += 1

    def record_backpressure(self) -> None:
        """Record a backpressure event."""
        self._backpressure_events += 1

    def get_metrics(self) -> StreamingMetrics:
        """Get current metrics snapshot.

        Returns:
            StreamingMetrics with current values.
        """
        avg_latency = (
            self._total_latency_ms / self._events_emitted if self._events_emitted > 0 else 0.0
        )
        return StreamingMetrics(
            events_emitted=self._events_emitted,
            events_dropped=self._events_dropped,
            avg_latency_ms=avg_latency,
            backpressure_events=self._backpressure_events,
        )

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self._events_emitted = 0
        self._events_dropped = 0
        self._total_latency_ms = 0.0
        self._backpressure_events = 0
