"""OpenTelemetry integration for reasoning-mcp.

This module provides distributed tracing capabilities for debugging
complex reasoning pipelines. Telemetry is optional and gracefully
disabled when OpenTelemetry packages are not installed.

Usage:
    from reasoning_mcp.telemetry import init_telemetry, get_tracer

    # Initialize in server lifespan
    init_telemetry(settings)

    # Get tracer for instrumentation
    tracer = get_tracer("reasoning-mcp.engine")

Example with spans:
    from reasoning_mcp.telemetry import traced_executor

    @traced_executor("method.execute")
    async def execute(self, context):
        ...
"""

from __future__ import annotations

from reasoning_mcp.telemetry.attributes import (
    ATTR_BRANCH_COUNT,
    ATTR_ITERATION,
    ATTR_MAX_CONCURRENCY,
    ATTR_MAX_ITERATIONS,
    ATTR_MERGE_STRATEGY,
    ATTR_METHOD_ID,
    ATTR_METHOD_TIMEOUT,
    ATTR_PIPELINE_ID,
    ATTR_SESSION_ID,
    ATTR_STAGE_ID,
    ATTR_STAGE_NAME,
    ATTR_STAGE_TYPE,
    ATTR_TERMINATION_REASON,
    ATTR_THOUGHTS_GENERATED,
)
from reasoning_mcp.telemetry.instrumentation import (
    record_exception,
    set_span_error,
    set_span_ok,
    traced_executor,
)
from reasoning_mcp.telemetry.provider import (
    get_tracer,
    init_telemetry,
    shutdown_telemetry,
)

__all__ = [
    # Provider
    "init_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    # Instrumentation
    "traced_executor",
    "set_span_ok",
    "set_span_error",
    "record_exception",
    # Attributes
    "ATTR_SESSION_ID",
    "ATTR_STAGE_ID",
    "ATTR_STAGE_TYPE",
    "ATTR_STAGE_NAME",
    "ATTR_PIPELINE_ID",
    "ATTR_METHOD_ID",
    "ATTR_METHOD_TIMEOUT",
    "ATTR_THOUGHTS_GENERATED",
    "ATTR_BRANCH_COUNT",
    "ATTR_MAX_CONCURRENCY",
    "ATTR_MERGE_STRATEGY",
    "ATTR_ITERATION",
    "ATTR_MAX_ITERATIONS",
    "ATTR_TERMINATION_REASON",
]
