"""Session management for reasoning-mcp.

This module provides the SessionManager class for creating, storing, and managing
reasoning sessions with thread-safe async operations.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from reasoning_mcp.models.core import SessionStatus
from reasoning_mcp.models.session import Session, SessionConfig


class SessionManager:
    """Thread-safe manager for reasoning sessions.

    SessionManager provides CRUD operations and utilities for managing multiple
    reasoning sessions with automatic cleanup and filtering capabilities. All
    operations are async and thread-safe using asyncio.Lock.

    Examples:
        Create a manager:
        >>> manager = SessionManager(max_sessions=100, cleanup_interval=3600)
        >>> assert await manager.count() == 0

        Create and manage sessions:
        >>> session = await manager.create()
        >>> assert session.status == SessionStatus.CREATED
        >>> assert await manager.count() == 1
        >>>
        >>> retrieved = await manager.get(session.id)
        >>> assert retrieved is not None
        >>> assert retrieved.id == session.id

        Update a session:
        >>> session.start()
        >>> updated = await manager.update(session.id, session)
        >>> assert updated is True
        >>> retrieved = await manager.get(session.id)
        >>> assert retrieved.status == SessionStatus.ACTIVE

        List sessions with filters:
        >>> active = await manager.list_sessions(status=SessionStatus.ACTIVE)
        >>> assert len(active) == 1
        >>> all_sessions = await manager.list_sessions(limit=10)
        >>> assert len(all_sessions) == 1

        Cleanup old sessions:
        >>> # Sessions older than 24 hours will be removed
        >>> removed = await manager.cleanup_expired(max_age_seconds=86400)
        >>> assert removed == 0  # No old sessions yet

        Delete specific session:
        >>> deleted = await manager.delete(session.id)
        >>> assert deleted is True
        >>> assert await manager.count() == 0

        Clear all sessions:
        >>> await manager.create()
        >>> await manager.create()
        >>> assert await manager.count() == 2
        >>> await manager.clear()
        >>> assert await manager.count() == 0
    """

    def __init__(self, max_sessions: int = 100, cleanup_interval: int = 3600):
        """Initialize the session manager.

        Args:
            max_sessions: Maximum number of sessions to store (default: 100)
            cleanup_interval: Interval in seconds for automatic cleanup (default: 3600)

        Examples:
            >>> manager = SessionManager()
            >>> assert manager._max_sessions == 100
            >>> assert manager._cleanup_interval == 3600
            >>>
            >>> custom_manager = SessionManager(max_sessions=50, cleanup_interval=1800)
            >>> assert custom_manager._max_sessions == 50
        """
        self._max_sessions = max_sessions
        self._cleanup_interval = cleanup_interval
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, config: SessionConfig | None = None) -> Session:
        """Create a new session.

        Args:
            config: Optional session configuration. If None, uses default config.

        Returns:
            The newly created Session

        Raises:
            RuntimeError: If max_sessions limit is reached

        Examples:
            >>> manager = SessionManager()
            >>> session = await manager.create()
            >>> assert session.status == SessionStatus.CREATED
            >>> assert session.config is not None
            >>>
            >>> custom_config = SessionConfig(max_depth=20, timeout_seconds=600.0)
            >>> session2 = await manager.create(config=custom_config)
            >>> assert session2.config.max_depth == 20
            >>> assert session2.config.timeout_seconds == 600.0
        """
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Maximum session limit reached ({self._max_sessions})"
                )

            session = Session(config=config) if config else Session()
            self._sessions[session.id] = session
            return session

    async def get(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The unique identifier of the session

        Returns:
            The Session if found, None otherwise

        Examples:
            >>> manager = SessionManager()
            >>> session = await manager.create()
            >>> retrieved = await manager.get(session.id)
            >>> assert retrieved is not None
            >>> assert retrieved.id == session.id
            >>>
            >>> not_found = await manager.get("non-existent-id")
            >>> assert not_found is None
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def update(self, session_id: str, session: Session) -> bool:
        """Update an existing session.

        Args:
            session_id: The unique identifier of the session to update
            session: The updated Session object

        Returns:
            True if the session was updated, False if not found

        Examples:
            >>> manager = SessionManager()
            >>> session = await manager.create()
            >>> session.start()
            >>> updated = await manager.update(session.id, session)
            >>> assert updated is True
            >>> retrieved = await manager.get(session.id)
            >>> assert retrieved.status == SessionStatus.ACTIVE
            >>>
            >>> not_updated = await manager.update("non-existent-id", session)
            >>> assert not_updated is False
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id] = session
            return True

    async def delete(self, session_id: str) -> bool:
        """Remove a session by ID.

        Args:
            session_id: The unique identifier of the session to delete

        Returns:
            True if the session was deleted, False if not found

        Examples:
            >>> manager = SessionManager()
            >>> session = await manager.create()
            >>> assert await manager.count() == 1
            >>> deleted = await manager.delete(session.id)
            >>> assert deleted is True
            >>> assert await manager.count() == 0
            >>>
            >>> not_deleted = await manager.delete("non-existent-id")
            >>> assert not_deleted is False
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            return True

    async def list_sessions(
        self,
        *,
        status: SessionStatus | None = None,
        limit: int = 100,
    ) -> list[Session]:
        """List sessions with optional status filter.

        Args:
            status: Optional status filter. If None, returns all sessions.
            limit: Maximum number of sessions to return (default: 100)

        Returns:
            List of sessions matching the criteria, sorted by creation time descending

        Examples:
            >>> manager = SessionManager()
            >>> session1 = await manager.create()
            >>> session2 = await manager.create()
            >>> session2.start()
            >>> await manager.update(session2.id, session2)
            >>>
            >>> all_sessions = await manager.list_sessions()
            >>> assert len(all_sessions) == 2
            >>>
            >>> active = await manager.list_sessions(status=SessionStatus.ACTIVE)
            >>> assert len(active) == 1
            >>> assert active[0].id == session2.id
            >>>
            >>> created = await manager.list_sessions(status=SessionStatus.CREATED)
            >>> assert len(created) == 1
            >>> assert created[0].id == session1.id
            >>>
            >>> limited = await manager.list_sessions(limit=1)
            >>> assert len(limited) == 1
        """
        async with self._lock:
            sessions = list(self._sessions.values())

            # Filter by status if provided
            if status is not None:
                sessions = [s for s in sessions if s.status == status]

            # Sort by created_at descending (most recent first)
            sessions.sort(key=lambda s: s.created_at, reverse=True)

            # Apply limit
            return sessions[:limit]

    async def cleanup_expired(self, max_age_seconds: int = 86400) -> int:
        """Remove sessions older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default: 86400 = 24 hours)

        Returns:
            Number of sessions removed

        Examples:
            >>> import time
            >>> manager = SessionManager()
            >>> session = await manager.create()
            >>>
            >>> # No sessions older than 24 hours
            >>> removed = await manager.cleanup_expired(max_age_seconds=86400)
            >>> assert removed == 0
            >>> assert await manager.count() == 1
            >>>
            >>> # Remove sessions older than 0 seconds (all of them)
            >>> removed = await manager.cleanup_expired(max_age_seconds=0)
            >>> assert removed == 1
            >>> assert await manager.count() == 0
        """
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=max_age_seconds)

            to_remove = [
                session_id
                for session_id, session in self._sessions.items()
                if session.created_at < cutoff
            ]

            for session_id in to_remove:
                del self._sessions[session_id]

            return len(to_remove)

    async def count(self) -> int:
        """Return the number of sessions currently stored.

        Returns:
            Total number of sessions

        Examples:
            >>> manager = SessionManager()
            >>> assert await manager.count() == 0
            >>> await manager.create()
            >>> assert await manager.count() == 1
            >>> await manager.create()
            >>> assert await manager.count() == 2
        """
        async with self._lock:
            return len(self._sessions)

    async def clear(self) -> None:
        """Remove all sessions.

        Examples:
            >>> manager = SessionManager()
            >>> await manager.create()
            >>> await manager.create()
            >>> await manager.create()
            >>> assert await manager.count() == 3
            >>> await manager.clear()
            >>> assert await manager.count() == 0
        """
        async with self._lock:
            self._sessions.clear()
