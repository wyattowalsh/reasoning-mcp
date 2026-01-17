"""Tests for CircuitBreaker implementation."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from reasoning_mcp.reliability import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    def test_initialization(self) -> None:
        """Test circuit breaker initializes with correct defaults."""
        circuit = CircuitBreaker("test_circuit")

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.name == "test_circuit"
        assert metrics.state == CircuitBreakerState.CLOSED
        assert metrics.failure_count == 0
        assert metrics.success_count == 0
        assert metrics.total_calls == 0
        assert metrics.rejection_count == 0

    def test_custom_thresholds(self) -> None:
        """Test circuit breaker accepts custom configuration."""
        circuit = CircuitBreaker(
            "custom_circuit",
            failure_threshold=3,
            recovery_timeout=30.0,
            half_open_max_calls=2,
        )

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        assert circuit.get_metrics().name == "custom_circuit"


class TestCircuitBreakerClosedState:
    """Test circuit breaker behavior in CLOSED state."""

    async def test_successful_call(self) -> None:
        """Test successful call passes through in CLOSED state."""
        circuit = CircuitBreaker("test", failure_threshold=3)

        async def success_func() -> str:
            return "success"

        result = await circuit.call(success_func, timeout=1.0)

        assert result == "success"
        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.total_calls == 1
        assert metrics.failure_count == 0
        assert metrics.last_success_time is not None

    async def test_multiple_successful_calls(self) -> None:
        """Test multiple successful calls remain in CLOSED state."""
        circuit = CircuitBreaker("test", failure_threshold=3)

        async def success_func() -> int:
            return 42

        for _i in range(5):
            result = await circuit.call(success_func, timeout=1.0)
            assert result == 42

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.total_calls == 5
        assert metrics.failure_count == 0

    async def test_failure_below_threshold(self) -> None:
        """Test failures below threshold keep circuit CLOSED."""
        circuit = CircuitBreaker("test", failure_threshold=3)

        async def fail_func() -> None:
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError, match="test error"):
            await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.failure_count == 1
        assert metrics.last_failure_time is not None

        # Second failure
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        assert circuit.get_metrics().failure_count == 2


class TestCircuitBreakerOpenState:
    """Test circuit breaker behavior when opening."""

    async def test_threshold_triggers_open(self) -> None:
        """Test circuit opens after reaching failure threshold."""
        circuit = CircuitBreaker("test", failure_threshold=3)

        async def fail_func() -> None:
            raise ValueError("test error")

        # Trigger failures up to threshold
        for _i in range(3):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        # Circuit should now be OPEN
        assert circuit.get_state() == CircuitBreakerState.OPEN
        metrics = circuit.get_metrics()
        assert metrics.failure_count == 3

    async def test_open_rejects_calls(self) -> None:
        """Test OPEN circuit rejects calls immediately."""
        circuit = CircuitBreaker("test", failure_threshold=2)

        async def fail_func() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _i in range(2):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Next call should be rejected without executing
        async def should_not_execute() -> None:
            pytest.fail("This function should not be executed")

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await circuit.call(should_not_execute, timeout=1.0)

        assert exc_info.value.circuit_name == "test"
        assert circuit.get_metrics().rejection_count == 1

    async def test_open_error_message(self) -> None:
        """Test CircuitBreakerOpenError has informative message."""
        circuit = CircuitBreaker("test_service", failure_threshold=1)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        # Try to call when open
        async def dummy() -> None:
            pass

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await circuit.call(dummy, timeout=1.0)

        error_msg = str(exc_info.value)
        assert "test_service" in error_msg
        assert "OPEN" in error_msg
        assert "Failures: 1" in error_msg


class TestCircuitBreakerHalfOpenState:
    """Test circuit breaker HALF_OPEN state and recovery."""

    async def test_transition_to_half_open(self) -> None:
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        circuit = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms timeout
        )

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        for _i in range(2):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should transition to HALF_OPEN
        async def success_func() -> str:
            return "recovered"

        result = await circuit.call(success_func, timeout=1.0)

        assert result == "recovered"
        assert circuit.get_state() == CircuitBreakerState.HALF_OPEN

    async def test_successful_recovery(self) -> None:
        """Test successful recovery transitions back to CLOSED."""
        circuit = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=3,
        )

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        for _i in range(2):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Make successful calls in HALF_OPEN state
        async def success_func() -> str:
            return "ok"

        for _i in range(3):
            result = await circuit.call(success_func, timeout=1.0)
            assert result == "ok"

        # Circuit should now be CLOSED
        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.failure_count == 0
        assert metrics.success_count == 0  # Reset on close

    async def test_failed_recovery(self) -> None:
        """Test failed recovery transitions back to OPEN."""
        circuit = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=3,
        )

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        for _i in range(2):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First call succeeds (HALF_OPEN)
        async def success_func() -> str:
            return "ok"

        await circuit.call(success_func, timeout=1.0)
        assert circuit.get_state() == CircuitBreakerState.HALF_OPEN

        # Second call fails
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        # Circuit should reopen
        assert circuit.get_state() == CircuitBreakerState.OPEN

    async def test_half_open_limited_calls(self) -> None:
        """Test HALF_OPEN state limits number of calls."""
        circuit = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Make 2 successful calls
        async def success_func() -> str:
            return "ok"

        await circuit.call(success_func, timeout=1.0)
        await circuit.call(success_func, timeout=1.0)

        # Circuit should be CLOSED now
        assert circuit.get_state() == CircuitBreakerState.CLOSED


class TestCircuitBreakerTimeout:
    """Test timeout enforcement."""

    async def test_timeout_enforcement(self) -> None:
        """Test calls are cancelled after timeout."""
        circuit = CircuitBreaker("test")

        async def slow_func() -> None:
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await circuit.call(slow_func, timeout=0.1)

        # Timeout should count as failure
        assert circuit.get_metrics().failure_count == 1

    async def test_timeout_opens_circuit(self) -> None:
        """Test repeated timeouts open the circuit."""
        circuit = CircuitBreaker("test", failure_threshold=2)

        async def slow_func() -> None:
            await asyncio.sleep(10)

        # Trigger timeouts
        for _i in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await circuit.call(slow_func, timeout=0.1)

        # Circuit should be OPEN
        assert circuit.get_state() == CircuitBreakerState.OPEN


class TestCircuitBreakerManualControl:
    """Test manual reset and force_open methods."""

    async def test_manual_reset(self) -> None:
        """Test manual reset transitions to CLOSED."""
        circuit = CircuitBreaker("test", failure_threshold=1)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Manual reset
        circuit.reset()

        assert circuit.get_state() == CircuitBreakerState.CLOSED
        metrics = circuit.get_metrics()
        assert metrics.failure_count == 0
        assert metrics.last_failure_time is None

        # Should allow calls now
        async def success_func() -> str:
            return "ok"

        result = await circuit.call(success_func, timeout=1.0)
        assert result == "ok"

    async def test_force_open(self) -> None:
        """Test force_open immediately opens circuit."""
        circuit = CircuitBreaker("test")

        assert circuit.get_state() == CircuitBreakerState.CLOSED

        # Force open
        circuit.force_open()

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Should reject calls
        async def func() -> None:
            pass

        with pytest.raises(CircuitBreakerOpenError):
            await circuit.call(func, timeout=1.0)


class TestCircuitBreakerMetrics:
    """Test metrics tracking."""

    async def test_metrics_total_calls(self) -> None:
        """Test total_calls is tracked correctly."""
        circuit = CircuitBreaker("test", failure_threshold=5)

        async def success_func() -> str:
            return "ok"

        async def fail_func() -> None:
            raise ValueError("error")

        # Mix of success and failure
        await circuit.call(success_func, timeout=1.0)
        await circuit.call(success_func, timeout=1.0)

        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        metrics = circuit.get_metrics()
        assert metrics.total_calls == 3

    async def test_metrics_rejection_count(self) -> None:
        """Test rejection_count tracks OPEN rejections."""
        circuit = CircuitBreaker("test", failure_threshold=1)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        # Try multiple calls while OPEN
        async def dummy() -> None:
            pass

        for _i in range(5):
            with pytest.raises(CircuitBreakerOpenError):
                await circuit.call(dummy, timeout=1.0)

        metrics = circuit.get_metrics()
        assert metrics.rejection_count == 5

    async def test_metrics_timestamps(self) -> None:
        """Test last_failure_time and last_success_time are tracked."""
        circuit = CircuitBreaker("test")

        # Initially None
        metrics = circuit.get_metrics()
        assert metrics.last_failure_time is None
        assert metrics.last_success_time is None

        # Success
        async def success_func() -> str:
            return "ok"

        before_success = datetime.now()
        await circuit.call(success_func, timeout=1.0)
        after_success = datetime.now()

        metrics = circuit.get_metrics()
        assert metrics.last_success_time is not None
        assert before_success <= metrics.last_success_time <= after_success

        # Failure
        async def fail_func() -> None:
            raise ValueError("error")

        before_failure = datetime.now()
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)
        after_failure = datetime.now()

        metrics = circuit.get_metrics()
        assert metrics.last_failure_time is not None
        assert before_failure <= metrics.last_failure_time <= after_failure

    async def test_metrics_immutable(self) -> None:
        """Test metrics are immutable snapshots."""
        circuit = CircuitBreaker("test")

        metrics1 = circuit.get_metrics()

        async def success_func() -> str:
            return "ok"

        await circuit.call(success_func, timeout=1.0)

        metrics2 = circuit.get_metrics()

        # Original snapshot unchanged
        assert metrics1.total_calls == 0
        assert metrics2.total_calls == 1


class TestCircuitBreakerConcurrency:
    """Test concurrent calls to circuit breaker."""

    async def test_concurrent_successful_calls(self) -> None:
        """Test multiple concurrent successful calls work correctly."""
        circuit = CircuitBreaker("test")

        async def success_func(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        # Run 10 concurrent calls
        tasks = [circuit.call(success_func, i, timeout=1.0) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results == [i * 2 for i in range(10)]
        assert circuit.get_state() == CircuitBreakerState.CLOSED
        assert circuit.get_metrics().total_calls == 10

    async def test_concurrent_calls_with_failures(self) -> None:
        """Test concurrent calls handle failures correctly."""
        circuit = CircuitBreaker("test", failure_threshold=5)

        call_count = 0

        async def mixed_func(should_fail: bool) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if should_fail:
                raise ValueError("fail")
            return "ok"

        # Mix of success and failure
        tasks = [circuit.call(mixed_func, i % 2 == 0, timeout=1.0) for i in range(10)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check we got mix of results and exceptions
        success_count = sum(1 for r in results if r == "ok")
        failure_count = sum(1 for r in results if isinstance(r, ValueError))

        assert success_count == 5
        assert failure_count == 5
        assert circuit.get_metrics().total_calls == 10

    async def test_concurrent_calls_during_state_transition(self) -> None:
        """Test thread safety during state transitions."""
        circuit = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0.1)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        for _i in range(3):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Launch concurrent calls that should transition to HALF_OPEN
        async def success_func() -> str:
            await asyncio.sleep(0.01)
            return "ok"

        tasks = [circuit.call(success_func, timeout=1.0) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed, some might be rejected
        success_results = [r for r in results if r == "ok"]
        assert len(success_results) > 0

        # State should be either HALF_OPEN or CLOSED depending on timing
        state = circuit.get_state()
        assert state in [CircuitBreakerState.HALF_OPEN, CircuitBreakerState.CLOSED]


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions."""

    async def test_function_with_args_and_kwargs(self) -> None:
        """Test circuit breaker works with args and kwargs."""
        circuit = CircuitBreaker("test")

        async def complex_func(a: int, b: str, c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"

        result = await circuit.call(complex_func, 42, "hello", c=3.14, timeout=1.0)

        assert result == "42-hello-3.14"

    async def test_success_resets_failure_count_in_closed(self) -> None:
        """Test success resets failure count in CLOSED state."""
        circuit = CircuitBreaker("test", failure_threshold=5)

        async def fail_func() -> None:
            raise ValueError("error")

        async def success_func() -> str:
            return "ok"

        # Some failures
        for _i in range(3):
            with pytest.raises(ValueError):
                await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_metrics().failure_count == 3

        # Success should reset failure count
        await circuit.call(success_func, timeout=1.0)

        assert circuit.get_metrics().failure_count == 0
        assert circuit.get_state() == CircuitBreakerState.CLOSED

    async def test_zero_recovery_timeout(self) -> None:
        """Test circuit with zero recovery timeout transitions immediately."""
        circuit = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        assert circuit.get_state() == CircuitBreakerState.OPEN

        # Should immediately transition to HALF_OPEN on next call
        async def success_func() -> str:
            return "ok"

        result = await circuit.call(success_func, timeout=1.0)

        assert result == "ok"
        assert circuit.get_state() == CircuitBreakerState.HALF_OPEN


class TestCircuitBreakerExceptions:
    """Test exception handling and types."""

    def test_circuit_breaker_error_hierarchy(self) -> None:
        """Test exception hierarchy is correct."""
        assert issubclass(CircuitBreakerError, Exception)
        assert issubclass(CircuitBreakerOpenError, CircuitBreakerError)

    async def test_open_error_contains_metrics(self) -> None:
        """Test CircuitBreakerOpenError includes metrics."""
        circuit = CircuitBreaker("test", failure_threshold=1)

        async def fail_func() -> None:
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await circuit.call(fail_func, timeout=1.0)

        # Get rejection error
        async def dummy() -> None:
            pass

        try:
            await circuit.call(dummy, timeout=1.0)
            pytest.fail("Should have raised CircuitBreakerOpenError")
        except CircuitBreakerOpenError as e:
            assert e.circuit_name == "test"
            assert isinstance(e.metrics, CircuitBreakerMetrics)
            assert e.metrics.state == CircuitBreakerState.OPEN
            assert e.metrics.failure_count == 1
