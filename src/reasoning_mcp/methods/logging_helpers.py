"""Logging helpers for reasoning methods.

This module provides consistent logging patterns for exception handling
across all native reasoning methods. It ensures that silent exception handlers
are replaced with properly logged fallbacks for debugging and monitoring.

Example usage:
    from reasoning_mcp.methods.logging_helpers import log_sampling_fallback

    try:
        result = await self._sample_generate(input_text)
    except Exception:
        log_sampling_fallback(
            method_name=self.__class__.__name__,
            operation="generate_candidates",
            step=str(self._step_counter),
            phase=self._current_phase,
        )
        result = self._heuristic_generate(input_text)
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from reasoning_mcp.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def log_sampling_fallback(
    method_name: str,
    operation: str,
    *,
    step: str | None = None,
    phase: str | None = None,
    input_summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a sampling failure with fallback to heuristic.

    Use for expected failures when LLM sampling is unavailable or fails
    gracefully. Logs at WARNING level.

    Args:
        method_name: Name of the reasoning method (use self.__class__.__name__)
        operation: The operation that failed (e.g., "generate_candidates")
        step: Optional step name or number
        phase: Optional phase name (e.g., "generate", "discriminate")
        input_summary: Optional truncated input for context
        extra: Optional additional context fields

    Example:
        except Exception:
            log_sampling_fallback(
                method_name=self.__class__.__name__,
                operation="generate_candidates",
                step="1",
                phase="generate",
            )
            return self._heuristic_generate(...)
    """
    log_extra: dict[str, Any] = {
        "method": method_name,
        "operation": operation,
        "fallback_type": "heuristic",
    }
    if step is not None:
        log_extra["step"] = step
    if phase is not None:
        log_extra["phase"] = phase
    if input_summary is not None:
        # Truncate to prevent log bloat
        log_extra["input_summary"] = input_summary[:200]
    if extra:
        log_extra.update(extra)

    logger.warning(
        "LLM sampling failed, using heuristic fallback",
        exc_info=True,
        extra=log_extra,
    )


def log_parse_fallback(
    method_name: str,
    operation: str,
    raw_response: str | None = None,
    *,
    expected_type: str = "unknown",
    fallback_value: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a parsing failure with default value fallback.

    Use for parsing LLM output that may be malformed. Logs at WARNING level.

    Args:
        method_name: Name of the reasoning method
        operation: The parse operation that failed (e.g., "parse_confidence")
        raw_response: The raw response that failed to parse (will be truncated)
        expected_type: What type was expected (e.g., "float", "json", "list")
        fallback_value: The fallback value being used
        extra: Optional additional context fields

    Example:
        except ValueError:
            log_parse_fallback(
                method_name=self.__class__.__name__,
                operation="parse_confidence_score",
                raw_response=confidence_text,
                expected_type="float",
                fallback_value=0.5,
            )
            return default_confidence
    """
    log_extra: dict[str, Any] = {
        "method": method_name,
        "operation": operation,
        "expected_type": expected_type,
        "fallback_value": str(fallback_value)[:100],
    }
    if raw_response is not None:
        # Truncate raw response to prevent log bloat
        log_extra["raw_response_preview"] = raw_response[:100]
    if extra:
        log_extra.update(extra)

    logger.warning(
        f"Failed to parse response as {expected_type}, using fallback",
        exc_info=True,
        extra=log_extra,
    )


def log_critical_fallback(
    method_name: str,
    operation: str,
    *,
    context_id: str | None = None,
    impact: str = "degraded_quality",
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a critical operation failure.

    Use for failures in critical paths that may affect reasoning quality.
    Logs at ERROR level.

    Args:
        method_name: Name of the reasoning method
        operation: The critical operation that failed
        context_id: Optional ID for the affected context (node, session, etc.)
        impact: Description of the impact (e.g., "reduced_exploration")
        extra: Optional additional context fields

    Example:
        except Exception:
            log_critical_fallback(
                method_name=self.__class__.__name__,
                operation="expand_tree_node",
                context_id=str(node.id),
                impact="reduced_exploration",
            )
            return self._create_fallback_node(...)
    """
    log_extra: dict[str, Any] = {
        "method": method_name,
        "operation": operation,
        "impact": impact,
    }
    if context_id is not None:
        log_extra["context_id"] = context_id
    if extra:
        log_extra.update(extra)

    logger.error(
        f"Critical operation '{operation}' failed in {method_name}",
        exc_info=True,
        extra=log_extra,
    )


class SamplingMetrics:
    """Track sampling success/failure metrics for circuit breaker integration.

    This class maintains a sliding window of recent sampling operations
    and can be used to implement circuit breaker patterns to avoid
    repeated failing calls.

    Example:
        metrics = SamplingMetrics("ChainOfThought")

        try:
            result = await sample(...)
            metrics.record_success()
        except Exception:
            metrics.record_failure()
            if metrics.should_circuit_break:
                # Skip sampling entirely for a while
                return heuristic_result()
    """

    def __init__(self, method_name: str, window_size: int = 100) -> None:
        """Initialize metrics tracker.

        Args:
            method_name: Name of the method being tracked
            window_size: Number of recent operations to track
        """
        self.method_name = method_name
        self.window_size = window_size
        self._successes: list[float] = []
        self._failures: list[float] = []

    def record_success(self) -> None:
        """Record a successful sampling operation."""
        now = time.time()
        self._successes.append(now)
        self._trim_old_entries()

    def record_failure(self) -> None:
        """Record a failed sampling operation."""
        now = time.time()
        self._failures.append(now)
        self._trim_old_entries()
        logger.debug(
            "Sampling failure recorded",
            extra={
                "method": self.method_name,
                "failure_rate": self.failure_rate,
                "total_failures": len(self._failures),
            },
        )

    def _trim_old_entries(self) -> None:
        """Remove entries beyond window size."""
        if len(self._successes) > self.window_size:
            self._successes = self._successes[-self.window_size :]
        if len(self._failures) > self.window_size:
            self._failures = self._failures[-self.window_size :]

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate.

        Returns:
            Failure rate as float between 0.0 and 1.0
        """
        total = len(self._successes) + len(self._failures)
        if total == 0:
            return 0.0
        return len(self._failures) / total

    @property
    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should activate.

        Circuit breaker activates when:
        - Failure rate exceeds 50%
        - At least 5 failures recorded (to avoid false positives)

        Returns:
            True if circuit breaker should activate
        """
        return self.failure_rate > 0.5 and len(self._failures) >= 5

    def reset(self) -> None:
        """Reset all metrics."""
        self._successes.clear()
        self._failures.clear()


def with_sampling_fallback(
    fallback_func: Callable[..., R],
    *,
    operation: str = "sampling",
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Decorator for async functions with sampling fallback.

    Wraps an async sampling function to automatically log failures
    and call a fallback function.

    Args:
        fallback_func: The fallback function to call on failure
        operation: Name of the operation for logging

    Returns:
        Decorated function that logs and falls back on failure

    Example:
        @with_sampling_fallback(self._heuristic_generate, operation="generate")
        async def _sample_generate(self, input_text: str) -> str:
            return await self._execution_context.sample(...)
    """

    def decorator(
        func: Callable[..., Awaitable[R]],
    ) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            # Extract self if this is a method call
            self_obj = args[0] if args else None
            method_name = (
                self_obj.__class__.__name__
                if self_obj and hasattr(self_obj, "__class__")
                else "Unknown"
            )
            try:
                return await func(*args, **kwargs)
            except Exception:
                log_sampling_fallback(
                    method_name=method_name,
                    operation=operation,
                )
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "log_sampling_fallback",
    "log_parse_fallback",
    "log_critical_fallback",
    "SamplingMetrics",
    "with_sampling_fallback",
]
