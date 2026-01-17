"""Tests for telemetry metrics module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reasoning_mcp.telemetry.metrics import (
    NoOpCounter,
    NoOpHistogram,
    NoOpUpDownCounter,
    get_active_sessions_counter,
    get_execution_duration_histogram,
    get_loop_iterations_histogram,
    get_method_errors_counter,
    get_method_executions_counter,
    get_parallel_branches_histogram,
    get_thoughts_generated_counter,
    init_metrics,
    record_loop_execution,
    record_method_execution,
    record_parallel_execution,
    record_session_ended,
    record_session_started,
    record_thoughts_generated,
)


class TestNoOpCounter:
    """Tests for NoOpCounter class."""

    def test_add_does_nothing(self) -> None:
        """Test add method is a no-op."""
        counter = NoOpCounter()
        # Should not raise
        counter.add(1)
        counter.add(10, {"key": "value"})
        counter.add(0)
        counter.add(-1)

    def test_add_accepts_any_attributes(self) -> None:
        """Test add accepts any attributes dict."""
        counter = NoOpCounter()
        counter.add(1, None)
        counter.add(1, {})
        counter.add(1, {"a": 1, "b": "two"})


class TestNoOpHistogram:
    """Tests for NoOpHistogram class."""

    def test_record_does_nothing(self) -> None:
        """Test record method is a no-op."""
        histogram = NoOpHistogram()
        # Should not raise
        histogram.record(1.0)
        histogram.record(100.5, {"key": "value"})
        histogram.record(0.0)
        histogram.record(-1.5)

    def test_record_accepts_any_attributes(self) -> None:
        """Test record accepts any attributes dict."""
        histogram = NoOpHistogram()
        histogram.record(1.0, None)
        histogram.record(1.0, {})
        histogram.record(1.0, {"method.id": "test", "success": True})


class TestNoOpUpDownCounter:
    """Tests for NoOpUpDownCounter class."""

    def test_add_does_nothing(self) -> None:
        """Test add method is a no-op."""
        counter = NoOpUpDownCounter()
        # Should not raise
        counter.add(1)
        counter.add(-1)
        counter.add(10, {"key": "value"})

    def test_add_accepts_negative_values(self) -> None:
        """Test add accepts negative values (up-down counter)."""
        counter = NoOpUpDownCounter()
        counter.add(-5)
        counter.add(-100, {"session": "test"})


class TestInitMetrics:
    """Tests for init_metrics function."""

    def test_returns_false_when_metrics_unavailable(self) -> None:
        """Test returns False when OpenTelemetry not available."""
        with patch(
            "reasoning_mcp.telemetry.metrics._METRICS_AVAILABLE", False
        ):
            result = init_metrics()
            assert result is False

    def test_returns_true_when_initialized_successfully(self) -> None:
        """Test returns True when metrics initialize successfully."""
        with patch(
            "reasoning_mcp.telemetry.metrics._METRICS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics.PeriodicExportingMetricReader"
            ) as mock_reader:
                with patch(
                    "reasoning_mcp.telemetry.metrics.MeterProvider"
                ) as mock_provider:
                    with patch(
                        "reasoning_mcp.telemetry.metrics.metrics"
                    ) as mock_metrics:
                        mock_reader.return_value = MagicMock()
                        mock_provider.return_value = MagicMock()
                        mock_meter = MagicMock()
                        mock_metrics.get_meter.return_value = mock_meter
                        mock_meter.create_counter.return_value = MagicMock()
                        mock_meter.create_histogram.return_value = MagicMock()
                        mock_meter.create_up_down_counter.return_value = MagicMock()

                        result = init_metrics("test-service")

                        assert result is True
                        mock_metrics.set_meter_provider.assert_called_once()
                        mock_metrics.get_meter.assert_called_once_with("test-service")

    def test_returns_false_on_initialization_error(self) -> None:
        """Test returns False when initialization fails."""
        with patch(
            "reasoning_mcp.telemetry.metrics._METRICS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics.PeriodicExportingMetricReader",
                side_effect=Exception("Init error"),
            ):
                result = init_metrics()
                assert result is False

    def test_uses_default_service_name(self) -> None:
        """Test uses 'reasoning-mcp' as default service name."""
        with patch(
            "reasoning_mcp.telemetry.metrics._METRICS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics.PeriodicExportingMetricReader"
            ) as mock_reader:
                with patch(
                    "reasoning_mcp.telemetry.metrics.MeterProvider"
                ) as mock_provider:
                    with patch(
                        "reasoning_mcp.telemetry.metrics.metrics"
                    ) as mock_metrics:
                        mock_reader.return_value = MagicMock()
                        mock_provider.return_value = MagicMock()
                        mock_meter = MagicMock()
                        mock_metrics.get_meter.return_value = mock_meter
                        mock_meter.create_counter.return_value = MagicMock()
                        mock_meter.create_histogram.return_value = MagicMock()
                        mock_meter.create_up_down_counter.return_value = MagicMock()

                        init_metrics()

                        mock_metrics.get_meter.assert_called_once_with("reasoning-mcp")


class TestGetterFunctions:
    """Tests for metric getter functions."""

    def test_get_method_executions_counter_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpCounter when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._method_executions_counter", None
        ):
            counter = get_method_executions_counter()
            assert isinstance(counter, NoOpCounter)

    def test_get_method_errors_counter_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpCounter when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._method_errors_counter", None
        ):
            counter = get_method_errors_counter()
            assert isinstance(counter, NoOpCounter)

    def test_get_thoughts_generated_counter_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpCounter when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._thoughts_generated_counter", None
        ):
            counter = get_thoughts_generated_counter()
            assert isinstance(counter, NoOpCounter)

    def test_get_execution_duration_histogram_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpHistogram when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._execution_duration_histogram", None
        ):
            histogram = get_execution_duration_histogram()
            assert isinstance(histogram, NoOpHistogram)

    def test_get_active_sessions_counter_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpUpDownCounter when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._active_sessions_counter", None
        ):
            counter = get_active_sessions_counter()
            assert isinstance(counter, NoOpUpDownCounter)

    def test_get_parallel_branches_histogram_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpHistogram when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._parallel_branches_histogram", None
        ):
            histogram = get_parallel_branches_histogram()
            assert isinstance(histogram, NoOpHistogram)

    def test_get_loop_iterations_histogram_returns_noop_when_uninitialized(self) -> None:
        """Test returns NoOpHistogram when not initialized."""
        with patch(
            "reasoning_mcp.telemetry.metrics._loop_iterations_histogram", None
        ):
            histogram = get_loop_iterations_histogram()
            assert isinstance(histogram, NoOpHistogram)


class TestRecordMethodExecution:
    """Tests for record_method_execution function."""

    def test_records_successful_execution(self) -> None:
        """Test records successful method execution."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_method_executions_counter",
            return_value=mock_counter,
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics.get_execution_duration_histogram",
                return_value=mock_histogram,
            ):
                with patch(
                    "reasoning_mcp.telemetry.metrics.get_method_errors_counter",
                    return_value=MagicMock(),
                ):
                    record_method_execution("chain_of_thought", True, 150.5)

                    mock_counter.add.assert_called_once()
                    call_args = mock_counter.add.call_args
                    assert call_args[0][0] == 1
                    assert call_args[0][1]["method.id"] == "chain_of_thought"
                    assert call_args[0][1]["success"] is True

                    mock_histogram.record.assert_called_once()

    def test_records_failed_execution(self) -> None:
        """Test records failed method execution with error counter."""
        mock_counter = MagicMock()
        mock_error_counter = MagicMock()
        mock_histogram = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_method_executions_counter",
            return_value=mock_counter,
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics.get_execution_duration_histogram",
                return_value=mock_histogram,
            ):
                with patch(
                    "reasoning_mcp.telemetry.metrics.get_method_errors_counter",
                    return_value=mock_error_counter,
                ):
                    record_method_execution("tree_of_thoughts", False, 50.0)

                    mock_error_counter.add.assert_called_once_with(
                        1, {"method.id": "tree_of_thoughts"}
                    )


class TestRecordThoughtsGenerated:
    """Tests for record_thoughts_generated function."""

    def test_records_thought_count(self) -> None:
        """Test records the number of thoughts generated."""
        mock_counter = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_thoughts_generated_counter",
            return_value=mock_counter,
        ):
            record_thoughts_generated(5, "chain_of_thought")

            mock_counter.add.assert_called_once_with(
                5, {"method.id": "chain_of_thought"}
            )


class TestRecordSessionStartedEnded:
    """Tests for session tracking functions."""

    def test_record_session_started_increments_counter(self) -> None:
        """Test session start increments active sessions counter."""
        mock_counter = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_active_sessions_counter",
            return_value=mock_counter,
        ):
            record_session_started()

            mock_counter.add.assert_called_once_with(1)

    def test_record_session_ended_decrements_counter(self) -> None:
        """Test session end decrements active sessions counter."""
        mock_counter = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_active_sessions_counter",
            return_value=mock_counter,
        ):
            record_session_ended()

            mock_counter.add.assert_called_once_with(-1)


class TestRecordParallelExecution:
    """Tests for record_parallel_execution function."""

    def test_records_branch_count(self) -> None:
        """Test records parallel branch count."""
        mock_histogram = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_parallel_branches_histogram",
            return_value=mock_histogram,
        ):
            record_parallel_execution(4)

            mock_histogram.record.assert_called_once_with(4)


class TestRecordLoopExecution:
    """Tests for record_loop_execution function."""

    def test_records_iteration_count(self) -> None:
        """Test records loop iteration count."""
        mock_histogram = MagicMock()

        with patch(
            "reasoning_mcp.telemetry.metrics.get_loop_iterations_histogram",
            return_value=mock_histogram,
        ):
            record_loop_execution(10)

            mock_histogram.record.assert_called_once_with(10)


class TestNoOpBehavior:
    """Tests for no-op behavior when metrics unavailable."""

    def test_record_method_execution_with_noop(self) -> None:
        """Test record_method_execution works with NoOp counters."""
        # Should not raise
        with patch(
            "reasoning_mcp.telemetry.metrics._method_executions_counter", None
        ):
            with patch(
                "reasoning_mcp.telemetry.metrics._execution_duration_histogram", None
            ):
                with patch(
                    "reasoning_mcp.telemetry.metrics._method_errors_counter", None
                ):
                    record_method_execution("test", True, 100.0)
                    record_method_execution("test", False, 50.0)

    def test_record_thoughts_generated_with_noop(self) -> None:
        """Test record_thoughts_generated works with NoOp counter."""
        with patch(
            "reasoning_mcp.telemetry.metrics._thoughts_generated_counter", None
        ):
            record_thoughts_generated(10, "test")

    def test_record_session_with_noop(self) -> None:
        """Test session tracking works with NoOp counter."""
        with patch(
            "reasoning_mcp.telemetry.metrics._active_sessions_counter", None
        ):
            record_session_started()
            record_session_ended()

    def test_record_parallel_execution_with_noop(self) -> None:
        """Test record_parallel_execution works with NoOp histogram."""
        with patch(
            "reasoning_mcp.telemetry.metrics._parallel_branches_histogram", None
        ):
            record_parallel_execution(3)

    def test_record_loop_execution_with_noop(self) -> None:
        """Test record_loop_execution works with NoOp histogram."""
        with patch(
            "reasoning_mcp.telemetry.metrics._loop_iterations_histogram", None
        ):
            record_loop_execution(5)
