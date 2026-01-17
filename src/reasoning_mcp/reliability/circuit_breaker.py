"""Circuit breaker pattern implementation for reliability.

This module provides a circuit breaker pattern to protect against cascading failures
when plugins or components fail. The circuit breaker tracks failures and automatically
transitions between CLOSED, OPEN, and HALF_OPEN states to prevent overload during
failures and allow for graceful recovery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    """Circuit breaker states.

    - CLOSED: Normal operation, all requests pass through
    - OPEN: Failing, requests are rejected immediately to prevent cascading failures
    - HALF_OPEN: Testing recovery, limited requests are allowed to test if the
      system has recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit is open and rejecting calls."""

    def __init__(self, circuit_name: str, metrics: CircuitBreakerMetrics) -> None:
        """Initialize the error with circuit details.

        Args:
            circuit_name: Name of the circuit breaker
            metrics: Current metrics of the circuit breaker
        """
        self.circuit_name = circuit_name
        self.metrics = metrics
        super().__init__(
            f"Circuit breaker '{circuit_name}' is OPEN. "
            f"Failures: {metrics.failure_count}, "
            f"Last failure: {metrics.last_failure_time}"
        )


@dataclass(frozen=True)
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring.

    This dataclass provides a snapshot of the circuit breaker's current state
    and historical performance metrics.
    """

    name: str
    """Name of the circuit breaker."""

    state: CircuitBreakerState
    """Current state of the circuit breaker."""

    failure_count: int
    """Number of consecutive failures in the current state."""

    success_count: int
    """Number of consecutive successes in the current state."""

    last_failure_time: datetime | None
    """Timestamp of the last recorded failure."""

    last_success_time: datetime | None
    """Timestamp of the last recorded success."""

    total_calls: int
    """Total number of calls attempted through this circuit breaker."""

    rejection_count: int
    """Number of calls rejected while in OPEN state."""


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    The circuit breaker pattern prevents cascading failures by tracking failures
    and automatically transitioning between states:

    1. CLOSED: Normal operation, all calls pass through
    2. OPEN: Too many failures, all calls are rejected immediately
    3. HALF_OPEN: Testing recovery, limited calls are allowed

    Example:
        >>> circuit = CircuitBreaker("my_service", failure_threshold=3)
        >>> async def risky_operation():
        ...     # Some operation that might fail
        ...     pass
        >>> result = await circuit.call(risky_operation, timeout=5.0)

    Thread Safety:
        This class is thread-safe and uses asyncio.Lock for state transitions.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker for identification
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery (HALF_OPEN)
            half_open_max_calls: Maximum successful calls in HALF_OPEN before closing
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._total_calls = 0
        self._rejection_count = 0
        self._lock = asyncio.Lock()

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: object,
        timeout: float = 30.0,
        **kwargs: object,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            timeout: Timeout in seconds for the function execution
            **kwargs: Keyword arguments to pass to func

        Returns:
            The result of the function call

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and rejecting calls
            asyncio.TimeoutError: If the function execution exceeds timeout
            Exception: Any exception raised by the function

        Example:
            >>> circuit = CircuitBreaker("api")
            >>> result = await circuit.call(fetch_data, url="...", timeout=10.0)
        """
        async with self._lock:
            self._total_calls += 1

            # Check if we should attempt recovery from OPEN state
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                else:
                    self._rejection_count += 1
                    raise CircuitBreakerOpenError(self._name, self.get_metrics())

            # In HALF_OPEN, only allow limited calls
            if (
                self._state == CircuitBreakerState.HALF_OPEN
                and self._success_count >= self._half_open_max_calls
            ):
                self._rejection_count += 1
                raise CircuitBreakerOpenError(self._name, self.get_metrics())

        # Execute the function with timeout
        try:
            async with asyncio.timeout(timeout):
                result = await func(*args, **kwargs)

            # Record success
            async with self._lock:
                self._record_success()

            return result

        except Exception as e:
            # Record failure
            async with self._lock:
                self._record_failure(e)
            raise

    def get_state(self) -> CircuitBreakerState:
        """Get the current state of the circuit breaker.

        Returns:
            Current circuit breaker state (CLOSED, OPEN, or HALF_OPEN)
        """
        return self._state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics snapshot.

        Returns:
            Immutable snapshot of current circuit breaker metrics
        """
        return CircuitBreakerMetrics(
            name=self._name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            total_calls=self._total_calls,
            rejection_count=self._rejection_count,
        )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        This resets all counters and transitions to CLOSED state, allowing
        normal operation to resume. Use with caution, as this bypasses the
        automatic recovery mechanism.
        """
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None

    def force_open(self) -> None:
        """Manually force the circuit breaker to OPEN state.

        This immediately opens the circuit and rejects all calls. Useful for
        maintenance windows or when you know a service is down.
        """
        self._state = CircuitBreakerState.OPEN
        self._failure_count = self._failure_threshold

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery.

        Returns:
            True if recovery timeout has elapsed, False otherwise
        """
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self._recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call and update state if necessary.

        In CLOSED state: Reset failure count
        In HALF_OPEN state: Increment success count, close circuit if threshold met
        """
        self._last_success_time = datetime.now()

        if self._state == CircuitBreakerState.CLOSED:
            self._failure_count = 0

        elif self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._half_open_max_calls:
                # Recovery successful, transition to CLOSED
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._success_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call and update state if necessary.

        Args:
            error: The exception that caused the failure

        In CLOSED state: Increment failure count, open circuit if threshold met
        In HALF_OPEN state: Immediately reopen circuit (recovery failed)
        """
        self._last_failure_time = datetime.now()
        self._failure_count += 1

        if self._state == CircuitBreakerState.CLOSED:
            if self._failure_count >= self._failure_threshold:
                # Too many failures, open the circuit
                self._state = CircuitBreakerState.OPEN

        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Recovery failed, reopen the circuit
            self._state = CircuitBreakerState.OPEN
            self._success_count = 0
