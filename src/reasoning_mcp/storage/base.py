"""Base storage backend interface for reasoning-mcp.

This module defines the abstract interface for session storage backends,
enabling persistence of reasoning sessions across server restarts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session
    from reasoning_mcp.models.thought import ThoughtGraph


class StorageBackend(ABC):
    """Abstract base class for session storage backends.

    StorageBackend defines the interface that all storage implementations must follow.
    This enables swapping between in-memory, disk-based, or hybrid storage strategies
    without changing the session management code.

    Implementations must handle:
    - Session serialization and deserialization
    - Thread-safe concurrent access
    - Graceful handling of missing data
    - Proper cleanup on close

    Examples:
        Create a custom storage backend:
        >>> class MyStorage(StorageBackend):
        ...     async def save_session(self, session: Session) -> bool:
        ...         # Implementation here
        ...         return True
        ...     # ... implement other methods
    """

    @abstractmethod
    async def save_session(self, session: Session) -> bool:
        """Save a session to persistent storage.

        Args:
            session: The Session object to persist

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> Session | None:
        """Load a session from persistent storage.

        Args:
            session_id: The unique identifier of the session to load

        Returns:
            The loaded Session object, or None if not found
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from persistent storage.

        Args:
            session_id: The unique identifier of the session to delete

        Returns:
            True if deleted successfully, False if not found
        """
        pass

    @abstractmethod
    async def list_session_ids(self) -> list[str]:
        """List all session IDs in storage.

        Returns:
            List of session ID strings
        """
        pass

    @abstractmethod
    async def save_graph(self, session_id: str, graph: ThoughtGraph) -> bool:
        """Save a thought graph separately (for lazy loading).

        This allows storing large graphs separately from session metadata
        to enable lazy loading and reduce memory usage.

        Args:
            session_id: The session ID this graph belongs to
            graph: The ThoughtGraph to persist

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def load_graph(self, session_id: str) -> ThoughtGraph | None:
        """Load a thought graph from storage.

        Args:
            session_id: The session ID whose graph to load

        Returns:
            The loaded ThoughtGraph, or None if not found
        """
        pass

    @abstractmethod
    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata without loading the full graph.

        This allows quick access to session info (status, created_at, etc.)
        without deserializing the potentially large thought graph.

        Args:
            session_id: The session ID to get metadata for

        Returns:
            Dictionary with session metadata, or None if not found
        """
        pass

    @abstractmethod
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists in storage.

        Args:
            session_id: The session ID to check

        Returns:
            True if session exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_session_size(self, session_id: str) -> int:
        """Get the approximate size of a session in bytes.

        Useful for deciding whether to lazy-load graphs or keep in memory.

        Args:
            session_id: The session ID to measure

        Returns:
            Approximate size in bytes, or 0 if not found
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources.

        This method should be called during shutdown to ensure
        all pending writes are flushed and connections are closed.
        """
        pass

    async def __aenter__(self) -> StorageBackend:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - ensures close is called."""
        await self.close()
