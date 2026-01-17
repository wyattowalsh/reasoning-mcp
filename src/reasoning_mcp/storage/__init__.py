"""Storage backends for persistent session management.

This module provides storage backends for persisting reasoning sessions
across server restarts. The default in-memory SessionManager can be extended
with disk-based persistence using DiskSessionStorage, or use HybridSessionManager
for the best of both worlds.

Quick Start:
    >>> from reasoning_mcp.storage import HybridSessionManager, DiskSessionStorage
    >>>
    >>> # Create a hybrid manager with disk persistence
    >>> storage = DiskSessionStorage(cache_dir="~/.reasoning-mcp/sessions")
    >>> manager = HybridSessionManager(storage=storage)
    >>> await manager.initialize()

Components:
    - StorageBackend: Abstract interface for storage implementations
    - DiskSessionStorage: SQLite-backed disk storage using diskcache
    - HybridSessionManager: Combined memory + disk session management

Optional Dependencies:
    Disk persistence requires the 'cache' optional dependency:
    >>> pip install reasoning-mcp[cache]
"""

from reasoning_mcp.storage.base import StorageBackend

__all__ = [
    "StorageBackend",
    "DiskSessionStorage",
    "HybridSessionManager",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional dependencies."""
    if name == "DiskSessionStorage":
        from reasoning_mcp.storage.disk import DiskSessionStorage

        return DiskSessionStorage

    if name == "HybridSessionManager":
        from reasoning_mcp.storage.hybrid import HybridSessionManager

        return HybridSessionManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
