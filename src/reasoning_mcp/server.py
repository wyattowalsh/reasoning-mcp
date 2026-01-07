"""MCP server core for reasoning-mcp.

This module provides the FastMCP server instance and application context
for managing reasoning methods, sessions, and server lifecycle.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from mcp.server.fastmcp import FastMCP

from reasoning_mcp.config import Settings
from reasoning_mcp.registry import MethodRegistry

if TYPE_CHECKING:
    # Import for type hints only - actual import happens in app_lifespan
    from reasoning_mcp.sessions import SessionManager


@dataclass
class AppContext:
    """Application context for server lifespan.

    This context holds all the stateful components needed throughout the
    server's lifetime, including the method registry, session manager,
    and configuration settings.

    Attributes:
        registry: Registry for managing reasoning methods
        session_manager: Manager for session lifecycle and state
        settings: Server configuration settings
        initialized: Whether the context has been initialized
    """

    registry: MethodRegistry
    session_manager: SessionManager
    settings: Settings
    initialized: bool = False


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle and context.

    This async context manager handles:
    - Creating and initializing core components (registry, session manager)
    - Providing the AppContext to server tools/resources
    - Cleanup on server shutdown

    Args:
        server: The FastMCP server instance

    Yields:
        AppContext: The initialized application context

    Example:
        >>> async with app_lifespan(mcp) as ctx:
        ...     method = ctx.registry.get("chain_of_thought")
        ...     session = await ctx.session_manager.create()
    """
    # Create settings
    settings = Settings()

    # Create registry
    registry = MethodRegistry()

    # Import SessionManager here to avoid circular imports
    from reasoning_mcp.sessions import SessionManager

    # Create session manager
    session_manager = SessionManager(
        max_sessions=settings.max_sessions,
        cleanup_interval=settings.session_cleanup_interval,
    )

    # Create context
    ctx = AppContext(
        registry=registry,
        session_manager=session_manager,
        settings=settings,
        initialized=False,
    )

    # Initialize components
    await registry.initialize()
    # SessionManager doesn't need explicit initialization

    # Mark as initialized
    ctx.initialized = True

    try:
        # Yield context to server
        yield ctx
    finally:
        # Cleanup on shutdown
        # Clear all sessions
        await session_manager.clear()

        # Registry cleanup (if needed in future)
        # await registry.cleanup()

        ctx.initialized = False


def register_middleware() -> None:
    """Register middleware components with the server.

    This is a placeholder for future middleware registration.
    Middleware may include:
    - Request logging and timing
    - Rate limiting
    - Authentication/authorization
    - Error handling and recovery
    - Telemetry and tracing
    """
    # TODO: Implement middleware registration in future batches
    pass


def get_sampling_handler() -> None:
    """Get the sampling handler for the server.

    This is a placeholder that returns None. In future implementations,
    this may return a configured sampling handler based on settings.

    Returns:
        None (placeholder for future implementation)
    """
    # TODO: Implement sampling handler configuration in TASK-042
    return None


def get_auth_provider() -> None:
    """Get the authentication provider for the server.

    This is a placeholder that returns None. In future implementations,
    this may return a JWT verifier or other auth provider based on settings.

    Returns:
        None (placeholder for future implementation)
    """
    # TODO: Implement auth provider configuration in TASK-043
    return None


# Create the FastMCP server instance
mcp = FastMCP(
    name="reasoning-mcp",
    lifespan=app_lifespan,
)

# Register middleware (placeholder)
register_middleware()


__all__ = [
    "mcp",
    "AppContext",
    "app_lifespan",
    "register_middleware",
    "get_sampling_handler",
    "get_auth_provider",
]
