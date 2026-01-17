"""MCP server core for reasoning-mcp.

This module provides the FastMCP server instance and application context
for managing reasoning methods, sessions, and server lifecycle.

FastMCP v2.14+ Features Used:
- Server lifespans for initialization/cleanup (v2.13+)
- MCP Middleware for logging and metrics (v2.9+)
- Context state management (v2.11+)
- Sampling support ready (v2.14+)
- Elicitation support ready (v2.10+)
- Background task support ready (v2.14+)

Production Hardening Features:
- Startup validation for external services (registry, telemetry, storage)
- Session persistence with HybridSessionManager
- Structured error handling with specific exception classes
- Health check support for readiness/liveness probes
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from reasoning_mcp.config import Settings
from reasoning_mcp.telemetry import init_telemetry, shutdown_telemetry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from reasoning_mcp.middleware import ReasoningMiddleware, ResponseCacheMiddleware
    from reasoning_mcp.registry import MethodRegistry
    from reasoning_mcp.sessions import SessionManager
    from reasoning_mcp.storage.base import StorageBackend
    from reasoning_mcp.storage.hybrid import HybridSessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# Exception Classes for Production Error Handling
# =============================================================================


class ServerError(Exception):
    """Base exception for server-related errors."""

    pass


class StartupValidationError(ServerError):
    """Raised when startup validation fails for a component."""

    def __init__(self, component: str, reason: str, recoverable: bool = False) -> None:
        self.component = component
        self.reason = reason
        self.recoverable = recoverable
        super().__init__(
            f"Startup validation failed for {component}: {reason}"
            + (" (recoverable)" if recoverable else " (fatal)")
        )


class ServiceUnavailableError(ServerError):
    """Raised when an external service is unavailable."""

    def __init__(self, service: str, details: str | None = None) -> None:
        self.service = service
        self.details = details
        msg = f"External service unavailable: {service}"
        if details:
            msg += f" - {details}"
        super().__init__(msg)


class SessionPersistenceError(ServerError):
    """Raised when session persistence operations fail."""

    def __init__(self, operation: str, session_id: str | None = None, cause: Exception | None = None) -> None:
        self.operation = operation
        self.session_id = session_id
        self.cause = cause
        msg = f"Session persistence error during {operation}"
        if session_id:
            msg += f" for session {session_id}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


# =============================================================================
# Health Check Support
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None


@dataclass
class ServerHealth:
    """Overall server health status."""

    status: HealthStatus
    components: list[ComponentHealth] = field(default_factory=list)
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "version": self.version,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.components
            ],
        }

# Global application context - set during server lifespan
_APP_CONTEXT: AppContext | None = None


class AppContextNotInitializedError(RuntimeError):
    """Raised when AppContext is accessed before server initialization."""

    def __init__(self) -> None:
        super().__init__(
            "AppContext not initialized. This typically means the server "
            "lifespan has not started or has already ended. Ensure tools "
            "are called within an active MCP session."
        )


def get_app_context() -> AppContext:
    """Get the current application context.

    This function provides access to the AppContext that was created
    during server initialization. It allows tools and other components
    to access the shared registry, session manager, and settings without
    relying on global singletons.

    Returns:
        The current AppContext instance

    Raises:
        AppContextNotInitializedError: If called before server initialization
            or after server shutdown

    Example:
        >>> from reasoning_mcp.server import get_app_context
        >>> ctx = get_app_context()
        >>> method = ctx.registry.get("chain_of_thought")
        >>> session = await ctx.session_manager.create()
    """
    if _APP_CONTEXT is None:
        raise AppContextNotInitializedError()
    return _APP_CONTEXT


@dataclass
class AppContext:
    """Application context for server lifespan.

    This context holds all the stateful components needed throughout the
    server's lifetime, including the method registry, session manager,
    middleware, and configuration settings.

    FastMCP v2.14+ Features:
    - Middleware reference for metrics access
    - Response cache for expensive operations
    - JWT authentication middleware for MCPaaS
    - API key authentication middleware for MCPaaS
    - Shared state dict for cross-tool communication
    - Settings with sampling/elicitation configuration

    Production Hardening Features:
    - Session persistence via HybridSessionManager
    - Storage backend reference for direct access
    - Startup validation results tracking

    Attributes:
        registry: Registry for managing reasoning methods
        session_manager: Manager for session lifecycle and state (memory-only)
        settings: Server configuration settings
        middleware: MCP middleware for logging and metrics (optional)
        cache: Response cache middleware for expensive operations (optional)
        jwt_middleware: JWT authentication middleware (optional, MCPaaS feature)
        api_key_middleware: API key authentication middleware (optional, MCPaaS feature)
        rate_limiter: Rate limiting middleware (optional, MCPaaS feature)
        shared_state: Shared state dict for cross-tool communication
        initialized: Whether the context has been initialized
        hybrid_session_manager: Hybrid manager with persistence (optional)
        storage_backend: Storage backend for session persistence (optional)
        startup_errors: List of non-fatal startup errors encountered
    """

    registry: MethodRegistry
    session_manager: SessionManager
    settings: Settings
    middleware: ReasoningMiddleware | None = None
    cache: ResponseCacheMiddleware | None = None
    jwt_middleware: Any = None  # JWTAuthMiddleware (optional)
    api_key_middleware: Any = None  # APIKeyAuthMiddleware (optional)
    rate_limiter: Any = None  # RateLimitMiddleware (optional)
    shared_state: dict[str, Any] = field(default_factory=dict)
    initialized: bool = False
    # Production hardening fields
    hybrid_session_manager: HybridSessionManager | None = None
    storage_backend: StorageBackend | None = None
    startup_errors: list[str] = field(default_factory=list)


async def _setup_session_persistence(
    settings: Settings,
    session_manager: SessionManager,
) -> tuple[StorageBackend | None, HybridSessionManager | None]:
    """Set up session persistence with disk storage and hybrid manager.

    Args:
        settings: Application settings with persistence configuration
        session_manager: In-memory session manager (used for config like max_sessions)

    Returns:
        Tuple of (storage_backend, hybrid_session_manager), both may be None on failure
    """
    from reasoning_mcp.storage.disk import DiskSessionStorage
    from reasoning_mcp.storage.hybrid import HybridSessionManager

    # Create disk storage backend
    storage_backend = DiskSessionStorage(
        cache_dir=settings.session_cache_dir,
    )

    # Initialize storage
    await storage_backend.initialize()

    # Create hybrid session manager with same config as in-memory manager
    hybrid_manager = HybridSessionManager(
        storage=storage_backend,
        max_sessions=settings.max_sessions,
        cleanup_interval=settings.session_cleanup_interval,
    )

    return storage_backend, hybrid_manager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle and context.

    This async context manager handles (FastMCP v2.13+ lifespan pattern):
    - Creating and initializing core components (registry, session manager)
    - Registering native reasoning methods
    - Setting up MCP middleware for logging and metrics (v2.9+)
    - Validating external services on startup
    - Setting up session persistence if enabled
    - Providing the AppContext to server tools/resources
    - Cleanup on server shutdown

    Args:
        server: The FastMCP server instance

    Yields:
        AppContext: The initialized application context

    Raises:
        StartupValidationError: If a critical component fails validation

    Example:
        >>> async with app_lifespan(mcp) as ctx:
        ...     method = ctx.registry.get("chain_of_thought")
        ...     session = await ctx.session_manager.create()
        ...     metrics = ctx.middleware.get_metrics() if ctx.middleware else None
    """
    # Track non-fatal startup errors
    startup_errors: list[str] = []

    # Create settings
    settings = Settings()

    # Initialize telemetry (OpenTelemetry tracing)
    try:
        init_telemetry(settings)
        logger.debug("Telemetry initialized successfully")
    except Exception as e:
        # Telemetry failure is non-fatal
        error_msg = f"Telemetry initialization failed: {e}"
        startup_errors.append(error_msg)
        logger.warning(error_msg)

    # Import MethodRegistry here to avoid circular imports
    from reasoning_mcp.registry import MethodRegistry

    # Create registry
    registry = MethodRegistry()

    # Register native methods with the registry
    from reasoning_mcp.methods.native import register_all_native_methods

    try:
        registration_results = register_all_native_methods(registry)
        successful = sum(registration_results.values())
        total = len(registration_results)
        logger.info(f"Registered {successful}/{total} native reasoning methods")

        # Validate that at least some methods were registered (critical)
        if successful == 0:
            raise StartupValidationError(
                component="MethodRegistry",
                reason="No reasoning methods were successfully registered",
                recoverable=False,
            )
        elif successful < total * 0.5:
            # More than 50% failed - log warning but continue
            error_msg = f"Only {successful}/{total} methods registered successfully"
            startup_errors.append(error_msg)
            logger.warning(error_msg)
    except StartupValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        raise StartupValidationError(
            component="MethodRegistry",
            reason=f"Failed to register native methods: {e}",
            recoverable=False,
        ) from e

    # Import SessionManager here to avoid circular imports
    from reasoning_mcp.sessions import SessionManager

    # Create session manager (always create for fallback)
    session_manager = SessionManager(
        max_sessions=settings.max_sessions,
        cleanup_interval=settings.session_cleanup_interval,
    )

    # Set up session persistence if enabled
    storage_backend = None
    hybrid_session_manager = None

    if settings.enable_session_persistence:
        try:
            storage_backend, hybrid_session_manager = await _setup_session_persistence(
                settings, session_manager
            )
            if hybrid_session_manager:
                logger.info(
                    f"Session persistence enabled "
                    f"(backend={settings.session_storage_backend}, "
                    f"cache_dir={settings.session_cache_dir})"
                )
        except Exception as e:
            # Persistence failure is non-fatal - fall back to memory-only
            error_msg = f"Session persistence setup failed, using memory-only: {e}"
            startup_errors.append(error_msg)
            logger.warning(error_msg)

    # Set up middleware if enabled (FastMCP v2.9+)
    middleware = None
    if settings.enable_middleware:
        from reasoning_mcp.middleware import ReasoningMiddleware

        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }

        middleware = ReasoningMiddleware(
            enable_logging=True,
            enable_metrics=settings.enable_middleware_metrics,
            log_level=log_level_map.get(settings.middleware_log_level, logging.DEBUG),
        )
        server.add_middleware(middleware)
        logger.info("MCP middleware registered with FastMCP server")

    # Set up response cache if enabled (FastMCP v2.14+)
    cache = None
    if settings.enable_cache:
        from reasoning_mcp.middleware import ResponseCacheMiddleware

        cache = ResponseCacheMiddleware(
            max_entries=settings.cache_max_entries,
            default_ttl=settings.cache_default_ttl_seconds,
        )
        logger.info(
            f"Response cache initialized (max_entries={settings.cache_max_entries}, "
            f"default_ttl={settings.cache_default_ttl_seconds}s)"
        )

    # Set up JWT authentication middleware if enabled (MCPaaS feature)
    jwt_middleware = None
    if settings.jwt_enabled:
        from reasoning_mcp.auth import create_middleware_from_settings

        jwt_middleware = create_middleware_from_settings(settings)
        if jwt_middleware:
            jwt_middleware.register(server)
            logger.info(
                f"JWT authentication enabled "
                f"(algorithm={settings.jwt_algorithm}, "
                f"expire_minutes={settings.jwt_expire_minutes})"
            )

    # Set up API key authentication middleware if enabled (MCPaaS feature)
    api_key_middleware = None
    if settings.api_key_enabled:
        from reasoning_mcp.auth import create_api_key_middleware_from_settings

        api_key_middleware = create_api_key_middleware_from_settings(settings)
        if api_key_middleware:
            api_key_middleware.register(server)
            logger.info(f"API key authentication enabled (header={settings.api_key_header})")

    # Set up rate limiting middleware if enabled (MCPaaS feature)
    rate_limiter = None
    if settings.enable_rate_limiting:
        from reasoning_mcp.middleware import RateLimitMiddleware

        rate_limiter = RateLimitMiddleware(settings)
        rate_limiter.register(server)
        logger.info(
            f"Rate limiting middleware registered "
            f"(per-minute: {settings.rate_limit_requests_per_minute}, "
            f"per-hour: {settings.rate_limit_requests_per_hour}, "
            f"burst: {settings.rate_limit_burst_size})"
        )

    # Create context
    ctx = AppContext(
        registry=registry,
        session_manager=session_manager,
        settings=settings,
        middleware=middleware,
        cache=cache,
        jwt_middleware=jwt_middleware,
        api_key_middleware=api_key_middleware,
        rate_limiter=rate_limiter,
        shared_state={},
        initialized=False,
        # Production hardening fields
        hybrid_session_manager=hybrid_session_manager,
        storage_backend=storage_backend,
        startup_errors=startup_errors,
    )

    # Initialize components
    try:
        await registry.initialize()
    except Exception as e:
        raise StartupValidationError(
            component="MethodRegistry",
            reason=f"Registry initialization failed: {e}",
            recoverable=False,
        ) from e

    # Initialize hybrid session manager if configured
    if hybrid_session_manager:
        try:
            await hybrid_session_manager.initialize()
        except Exception as e:
            error_msg = f"Hybrid session manager initialization failed: {e}"
            startup_errors.append(error_msg)
            logger.warning(error_msg)
            # Fall back to memory-only by clearing the hybrid manager reference
            ctx.hybrid_session_manager = None

    # Mark as initialized
    ctx.initialized = True

    # Store context globally for tool access
    global _APP_CONTEXT
    _APP_CONTEXT = ctx

    # Log startup summary
    persistence_status = "enabled" if hybrid_session_manager else "disabled"
    logger.info(
        f"reasoning-mcp server initialized (FastMCP v2.14+ features enabled: "
        f"middleware={settings.enable_middleware}, "
        f"cache={settings.enable_cache}, "
        f"jwt_auth={settings.jwt_enabled}, "
        f"api_key_auth={settings.api_key_enabled}, "
        f"rate_limiting={settings.enable_rate_limiting}, "
        f"sampling={settings.enable_sampling}, "
        f"elicitation={settings.enable_elicitation}, "
        f"background_tasks={settings.enable_background_tasks}, "
        f"session_persistence={persistence_status})"
    )

    # Log any startup warnings
    if startup_errors:
        logger.warning(
            f"Server started with {len(startup_errors)} non-fatal issue(s): "
            f"{'; '.join(startup_errors)}"
        )

    try:
        # Yield context to server
        yield ctx
    finally:
        # Clear global context first
        _APP_CONTEXT = None
        # Cleanup on shutdown
        logger.info("Shutting down reasoning-mcp server...")

        # Clear all sessions
        await session_manager.clear()

        # Reset middleware metrics
        if middleware:
            final_metrics = middleware.get_metrics()
            logger.info(
                f"Final metrics: {final_metrics.tool_calls} tool calls, "
                f"{final_metrics.resource_reads} resource reads, "
                f"{final_metrics.errors} errors"
            )
            middleware.reset_metrics()

        # Log cache metrics and clear cache
        if cache:
            cache_metrics = cache.get_metrics()
            logger.info(
                f"Final cache metrics: {cache_metrics.hits} hits, "
                f"{cache_metrics.misses} misses, "
                f"hit_rate={cache_metrics.hit_rate:.2%}, "
                f"{cache_metrics.evictions} evictions"
            )
            await cache.clear()

        # Shutdown rate limiter and log metrics
        if rate_limiter:
            rate_limit_metrics = rate_limiter.get_metrics()
            logger.info(
                f"Final rate limit metrics: {rate_limit_metrics.requests_allowed} allowed, "
                f"{rate_limit_metrics.requests_rejected} rejected, "
                f"rejection_rate={rate_limit_metrics.rejection_rate:.2%}, "
                f"{rate_limit_metrics.unique_clients} unique clients"
            )
            await rate_limiter.shutdown()

        # Shutdown telemetry (flush and close exporters)
        shutdown_telemetry()

        ctx.initialized = False
        logger.info("reasoning-mcp server shutdown complete")


def get_sampling_handler(settings: Settings | None = None) -> Any:
    """Get the sampling configuration for the server.

    Creates and returns sampling configuration based on settings.
    FastMCP v2.14+ uses SamplingTool decorator and Context.sample() method
    rather than explicit handler classes.

    Args:
        settings: Server settings (uses global settings if None)

    Returns:
        Sampling configuration dict or None if sampling disabled

    Note:
        In FastMCP v2.14+, sampling is handled via:
        - @mcp.sampling_tool() decorator for registering sampling tools
        - Context.sample() method for invoking sampling within tools
        The AnthropicSamplingHandler/OpenAISamplingHandler classes have been
        replaced with a unified sampling interface.
    """
    if settings is None:
        from reasoning_mcp.config import get_settings

        settings = get_settings()

    if not settings.enable_sampling:
        return None

    # Return sampling configuration for the server
    # In FastMCP v2.14+, sampling is configured via environment variables
    # (OPENAI_API_KEY, ANTHROPIC_API_KEY) and uses Context.sample()
    sampling_config: dict[str, Any] = {
        "enabled": True,
        "provider": settings.sampling_provider,
    }

    logger.info(
        f"Sampling enabled (provider={settings.sampling_provider or 'auto'}). "
        "Use Context.sample() in tools to invoke LLM sampling."
    )

    return sampling_config


def get_auth_provider(settings: Settings | None = None) -> Any:
    """Get the authentication provider for the server.

    Creates and returns an auth provider based on settings.
    Supports JWT and API key authentication (FastMCP v2.12+).

    Args:
        settings: Server settings (uses global settings if None)

    Returns:
        Configured auth provider or None if auth disabled
    """
    if settings is None:
        from reasoning_mcp.config import get_settings

        settings = get_settings()

    # Check if JWT or API key authentication is enabled
    if settings.jwt_enabled:
        from reasoning_mcp.auth import create_middleware_from_settings

        return create_middleware_from_settings(settings)
    elif settings.api_key_enabled:
        from reasoning_mcp.auth import create_api_key_middleware_from_settings

        return create_api_key_middleware_from_settings(settings)

    # Auth is optional - return None by default
    return None


# Create the FastMCP server instance with v2.14+ features
mcp = FastMCP(
    name="reasoning-mcp",
    lifespan=app_lifespan,
)

# Register MCP tools with titles and meta parameters (v2.10+)
from reasoning_mcp.tools.register import register_tools

register_tools(mcp)

# Register background task management tools (FastMCP v2.14+)
from reasoning_mcp.tasks import register_background_tools

register_background_tools(mcp)

# Register MCP resources (session, method, template, trace)
from reasoning_mcp.resources import register_all_resources

register_all_resources(mcp)

logger.info("MCP tools and resources registered with FastMCP server (FastMCP v2.14+)")


__all__ = [
    "mcp",
    "AppContext",
    "AppContextNotInitializedError",
    "app_lifespan",
    "get_app_context",
    "get_sampling_handler",
    "get_auth_provider",
]
