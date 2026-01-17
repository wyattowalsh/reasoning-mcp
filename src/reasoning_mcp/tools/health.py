"""Health check tools for reasoning-mcp.

This module provides MCP tools for:
- Checking server health and readiness
- Getting component status
- Retrieving server version and startup errors
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from reasoning_mcp.server import AppContext


class HealthCheckOutput(BaseModel):
    """Output from health check tool."""

    status: str = Field(description="Overall health status (healthy, degraded, unhealthy)")
    version: str = Field(description="Server version")
    components: list[dict[str, Any]] = Field(description="Component health details")
    startup_errors: list[str] = Field(default_factory=list, description="Non-fatal startup errors")


class ReadinessCheckOutput(BaseModel):
    """Output from readiness check tool."""

    ready: bool = Field(description="Whether server is ready to accept requests")
    reason: str | None = Field(default=None, description="Reason if not ready")
    components_ready: dict[str, bool] = Field(description="Individual component readiness")


class LivenessCheckOutput(BaseModel):
    """Output from liveness check tool."""

    alive: bool = Field(description="Whether server is alive")
    uptime_seconds: float | None = Field(default=None, description="Server uptime in seconds")


async def health_check_handler(ctx: AppContext) -> HealthCheckOutput:
    """Perform comprehensive health check.

    Checks all server components and returns their status.

    Args:
        ctx: Application context

    Returns:
        HealthCheckOutput with status and component details
    """
    from reasoning_mcp.server import ComponentHealth, HealthStatus, ServerHealth

    components: list[ComponentHealth] = []
    overall_status = HealthStatus.HEALTHY

    # Check registry
    try:
        start = time.perf_counter()
        method_count = len(ctx.registry.list_all())
        latency = (time.perf_counter() - start) * 1000
        components.append(
            ComponentHealth(
                name="registry",
                status=HealthStatus.HEALTHY if method_count > 0 else HealthStatus.DEGRADED,
                message=f"{method_count} methods registered",
                latency_ms=latency,
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="registry",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
        )
        overall_status = HealthStatus.UNHEALTHY

    # Check session manager
    try:
        start = time.perf_counter()
        session_count = len(await ctx.session_manager.list_sessions())
        latency = (time.perf_counter() - start) * 1000
        components.append(
            ComponentHealth(
                name="session_manager",
                status=HealthStatus.HEALTHY,
                message=f"{session_count} active sessions",
                latency_ms=latency,
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="session_manager",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
        )
        overall_status = HealthStatus.UNHEALTHY

    # Check hybrid session manager if configured
    if ctx.hybrid_session_manager is not None:
        try:
            start = time.perf_counter()
            # Simple health check - just verify we can list sessions
            await ctx.hybrid_session_manager.list_sessions()
            latency = (time.perf_counter() - start) * 1000
            components.append(
                ComponentHealth(
                    name="persistence",
                    status=HealthStatus.HEALTHY,
                    message="Disk persistence active",
                    latency_ms=latency,
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="persistence",
                    status=HealthStatus.DEGRADED,
                    message=f"Persistence degraded: {e}",
                )
            )
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

    # Check middleware if configured
    if ctx.middleware is not None:
        try:
            metrics = ctx.middleware.get_metrics()
            components.append(
                ComponentHealth(
                    name="middleware",
                    status=HealthStatus.HEALTHY,
                    message=f"{metrics.tool_calls} calls, {metrics.errors} errors",
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="middleware",
                    status=HealthStatus.DEGRADED,
                    message=str(e),
                )
            )

    # Check cache if configured
    if ctx.cache is not None:
        try:
            cache_metrics = ctx.cache.get_metrics()
            components.append(
                ComponentHealth(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message=f"Hit rate: {cache_metrics.hit_rate:.1%}",
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    message=str(e),
                )
            )

    # Build health object
    health = ServerHealth(
        status=overall_status,
        components=components,
        version="0.1.0",
    )

    return HealthCheckOutput(
        status=health.status.value,
        version=health.version,
        components=[
            {
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
                "latency_ms": c.latency_ms,
            }
            for c in health.components
        ],
        startup_errors=ctx.startup_errors,
    )


async def readiness_check_handler(ctx: AppContext) -> ReadinessCheckOutput:
    """Check if server is ready to accept requests.

    Verifies that all critical components are initialized and operational.

    Args:
        ctx: Application context

    Returns:
        ReadinessCheckOutput with readiness status
    """
    components_ready: dict[str, bool] = {}
    not_ready_reasons: list[str] = []

    # Check if context is initialized
    if not ctx.initialized:
        return ReadinessCheckOutput(
            ready=False,
            reason="Server context not initialized",
            components_ready={"context": False},
        )
    components_ready["context"] = True

    # Check registry
    try:
        method_count = len(ctx.registry.list_all())
        components_ready["registry"] = method_count > 0
        if method_count == 0:
            not_ready_reasons.append("No methods registered")
    except Exception as e:
        components_ready["registry"] = False
        not_ready_reasons.append(f"Registry error: {e}")

    # Check session manager
    try:
        await ctx.session_manager.list_sessions()
        components_ready["session_manager"] = True
    except Exception as e:
        components_ready["session_manager"] = False
        not_ready_reasons.append(f"Session manager error: {e}")

    # Determine overall readiness
    ready = all(components_ready.values())
    reason = "; ".join(not_ready_reasons) if not_ready_reasons else None

    return ReadinessCheckOutput(
        ready=ready,
        reason=reason,
        components_ready=components_ready,
    )


async def liveness_check_handler(ctx: AppContext) -> LivenessCheckOutput:
    """Check if server is alive.

    Simple liveness probe that returns quickly.

    Args:
        ctx: Application context

    Returns:
        LivenessCheckOutput with liveness status
    """
    # Server is alive if we can execute this handler
    return LivenessCheckOutput(
        alive=True,
        uptime_seconds=None,  # Could track startup time in ctx if needed
    )


__all__ = [
    "HealthCheckOutput",
    "ReadinessCheckOutput",
    "LivenessCheckOutput",
    "health_check_handler",
    "readiness_check_handler",
    "liveness_check_handler",
]
