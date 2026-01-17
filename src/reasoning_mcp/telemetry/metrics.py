"""OpenTelemetry metrics definitions for reasoning-mcp.

This module defines counters, histograms, and other metrics for
monitoring reasoning pipeline performance.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Track whether metrics are available
_METRICS_AVAILABLE = False
_meter: Any = None

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    _METRICS_AVAILABLE = True
except ImportError:
    metrics = None  # type: ignore[assignment]
    MeterProvider = None  # type: ignore[assignment, misc]
    ConsoleMetricExporter = None  # type: ignore[assignment, misc]
    PeriodicExportingMetricReader = None  # type: ignore[assignment, misc]


class NoOpCounter:
    """No-op counter for when metrics are unavailable."""

    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass


class NoOpHistogram:
    """No-op histogram for when metrics are unavailable."""

    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass


class NoOpUpDownCounter:
    """No-op up-down counter for when metrics are unavailable."""

    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass


# Metric instances (initialized lazily)
_method_executions_counter: Any = None
_method_errors_counter: Any = None
_thoughts_generated_counter: Any = None
_execution_duration_histogram: Any = None
_active_sessions_counter: Any = None
_parallel_branches_histogram: Any = None
_loop_iterations_histogram: Any = None


def init_metrics(service_name: str = "reasoning-mcp") -> bool:
    """Initialize OpenTelemetry metrics.

    Args:
        service_name: Name of the service for metrics.

    Returns:
        True if metrics were initialized, False otherwise.
    """
    global _meter, _method_executions_counter, _method_errors_counter
    global _thoughts_generated_counter, _execution_duration_histogram
    global _active_sessions_counter, _parallel_branches_histogram
    global _loop_iterations_histogram

    if not _METRICS_AVAILABLE:
        logger.warning("OpenTelemetry metrics not available")
        return False

    try:
        # Create meter provider with console exporter for now
        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=60000,  # Export every minute
        )
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

        _meter = metrics.get_meter(service_name)

        # Create counters
        _method_executions_counter = _meter.create_counter(
            name="reasoning.method.executions",
            description="Number of method executions",
            unit="1",
        )

        _method_errors_counter = _meter.create_counter(
            name="reasoning.method.errors",
            description="Number of method execution errors",
            unit="1",
        )

        _thoughts_generated_counter = _meter.create_counter(
            name="reasoning.thoughts.generated",
            description="Total thoughts generated",
            unit="1",
        )

        # Create histograms
        _execution_duration_histogram = _meter.create_histogram(
            name="reasoning.execution.duration",
            description="Execution duration in milliseconds",
            unit="ms",
        )

        _parallel_branches_histogram = _meter.create_histogram(
            name="reasoning.parallel.branches",
            description="Number of parallel branches executed",
            unit="1",
        )

        _loop_iterations_histogram = _meter.create_histogram(
            name="reasoning.loop.iterations",
            description="Number of loop iterations",
            unit="1",
        )

        # Create up-down counters
        _active_sessions_counter = _meter.create_up_down_counter(
            name="reasoning.sessions.active",
            description="Number of active sessions",
            unit="1",
        )

        logger.info("OpenTelemetry metrics initialized")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        return False


def get_method_executions_counter() -> Any:
    """Get the method executions counter."""
    return _method_executions_counter or NoOpCounter()


def get_method_errors_counter() -> Any:
    """Get the method errors counter."""
    return _method_errors_counter or NoOpCounter()


def get_thoughts_generated_counter() -> Any:
    """Get the thoughts generated counter."""
    return _thoughts_generated_counter or NoOpCounter()


def get_execution_duration_histogram() -> Any:
    """Get the execution duration histogram."""
    return _execution_duration_histogram or NoOpHistogram()


def get_active_sessions_counter() -> Any:
    """Get the active sessions counter."""
    return _active_sessions_counter or NoOpUpDownCounter()


def get_parallel_branches_histogram() -> Any:
    """Get the parallel branches histogram."""
    return _parallel_branches_histogram or NoOpHistogram()


def get_loop_iterations_histogram() -> Any:
    """Get the loop iterations histogram."""
    return _loop_iterations_histogram or NoOpHistogram()


# Convenience functions for recording metrics
def record_method_execution(method_id: str, success: bool, duration_ms: float) -> None:
    """Record a method execution metric.

    Args:
        method_id: The method identifier.
        success: Whether the execution succeeded.
        duration_ms: Execution duration in milliseconds.
    """
    attrs = {"method.id": method_id, "success": success}

    get_method_executions_counter().add(1, attrs)
    get_execution_duration_histogram().record(duration_ms, attrs)

    if not success:
        get_method_errors_counter().add(1, {"method.id": method_id})


def record_thoughts_generated(count: int, method_id: str) -> None:
    """Record thoughts generated metric.

    Args:
        count: Number of thoughts generated.
        method_id: The method that generated them.
    """
    get_thoughts_generated_counter().add(count, {"method.id": method_id})


def record_session_started() -> None:
    """Record a session start."""
    get_active_sessions_counter().add(1)


def record_session_ended() -> None:
    """Record a session end."""
    get_active_sessions_counter().add(-1)


def record_parallel_execution(branch_count: int) -> None:
    """Record parallel branch count.

    Args:
        branch_count: Number of parallel branches executed.
    """
    get_parallel_branches_histogram().record(branch_count)


def record_loop_execution(iteration_count: int) -> None:
    """Record loop iteration count.

    Args:
        iteration_count: Number of iterations executed.
    """
    get_loop_iterations_histogram().record(iteration_count)


__all__ = [
    "init_metrics",
    "get_method_executions_counter",
    "get_method_errors_counter",
    "get_thoughts_generated_counter",
    "get_execution_duration_histogram",
    "get_active_sessions_counter",
    "get_parallel_branches_histogram",
    "get_loop_iterations_histogram",
    "record_method_execution",
    "record_thoughts_generated",
    "record_session_started",
    "record_session_ended",
    "record_parallel_execution",
    "record_loop_execution",
]
