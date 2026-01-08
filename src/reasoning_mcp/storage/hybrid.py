"""Hybrid session manager combining in-memory and disk storage.

This module implements a hybrid session management strategy that keeps hot
(active) sessions in memory for fast access while persisting all sessions
to disk for recovery across server restarts.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from reasoning_mcp.models.core import SessionStatus
from reasoning_mcp.sessions import SessionManager
from reasoning_mcp.storage.base import StorageBackend

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session, SessionConfig

logger = logging.getLogger(__name__)


class HybridSessionManager:
    """Hybrid session manager with memory + disk storage.

    HybridSessionManager combines the fast access of in-memory storage with
    the persistence of disk storage. Features include:

    - Hot sessions kept in memory for fast access
    - Cold sessions on disk, lazy-loaded on demand
    - Automatic persistence of dirty sessions
    - Session recovery on startup
    - Configurable lazy loading threshold

    The manager wraps the existing SessionManager for in-memory operations
    and delegates to a StorageBackend for persistence.

    Examples:
        Create a hybrid manager:
        >>> from reasoning_mcp.storage.disk import DiskSessionStorage
        >>> storage = DiskSessionStorage()
        >>> manager = HybridSessionManager(
        ...     storage=storage,
        ...     max_sessions=100,
        ...     lazy_load_threshold_kb=50,
        ... )

        Create and persist a session:
        >>> session = await manager.create()
        >>> await manager.persist(session.id)  # Explicitly persist

        Recover sessions on startup:
        >>> recovered = await manager.recover_sessions(max_age_hours=24)
        >>> print(f"Recovered {recovered} sessions")
    """

    def __init__(
        self,
        storage: StorageBackend | None = None,
        max_sessions: int = 100,
        cleanup_interval: int = 3600,
        lazy_load_threshold_kb: int = 50,
        auto_persist_interval: int = 60,
        recovery_on_startup: bool = True,
        max_recovery_age_hours: int = 24,
    ):
        """Initialize the hybrid session manager.

        Args:
            storage: StorageBackend for persistence (None to disable persistence)
            max_sessions: Maximum number of sessions in memory (default: 100)
            cleanup_interval: Interval for auto-cleanup in seconds (default: 3600)
            lazy_load_threshold_kb: Size threshold for lazy loading in KB (default: 50)
            auto_persist_interval: Interval for auto-persist in seconds (default: 60)
            recovery_on_startup: Whether to recover sessions on startup (default: True)
            max_recovery_age_hours: Max age for recovered sessions in hours (default: 24)
        """
        # In-memory session manager
        self._memory = SessionManager(
            max_sessions=max_sessions,
            cleanup_interval=cleanup_interval,
        )

        # Disk storage backend
        self._storage = storage

        # Configuration
        self._lazy_load_threshold = lazy_load_threshold_kb * 1024  # Convert to bytes
        self._auto_persist_interval = auto_persist_interval
        self._recovery_on_startup = recovery_on_startup
        self._max_recovery_age_hours = max_recovery_age_hours

        # Track dirty sessions (modified since last persist)
        self._dirty_sessions: set[str] = set()
        self._dirty_lock = asyncio.Lock()  # Lock for dirty session tracking

        # Auto-persist task
        self._persist_task: asyncio.Task | None = None
        self._shutdown = False

    async def initialize(self) -> None:
        """Initialize the manager and optionally recover sessions.

        This should be called during server startup to recover any
        persisted sessions from the previous run.
        """
        if self._recovery_on_startup and self._storage:
            await self.recover_sessions(max_age_hours=self._max_recovery_age_hours)

        # Start auto-persist task if storage is available
        if self._storage and self._auto_persist_interval > 0:
            self._persist_task = asyncio.create_task(self._auto_persist_loop())
            logger.info(
                f"Auto-persist task started (interval={self._auto_persist_interval}s)"
            )

    async def _auto_persist_loop(self) -> None:
        """Background task to periodically persist dirty sessions."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._auto_persist_interval)
                if self._shutdown:
                    break
                await self.persist_dirty()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-persist loop: {e}")

    async def create(self, config: SessionConfig | None = None) -> Session:
        """Create a new session.

        Args:
            config: Optional session configuration

        Returns:
            The newly created Session
        """
        session = await self._memory.create(config)

        # Mark as dirty for persistence
        async with self._dirty_lock:
            self._dirty_sessions.add(session.id)

        return session

    async def get(self, session_id: str) -> Session | None:
        """Get a session by ID, loading from disk if needed.

        Args:
            session_id: The unique identifier of the session

        Returns:
            The Session if found, None otherwise
        """
        # First check memory
        session = await self._memory.get(session_id)
        if session is not None:
            return session

        # Try loading from disk if storage is available
        if self._storage:
            session = await self._load_from_storage(session_id)
            if session is not None:
                # Add to memory cache
                try:
                    # Temporarily increase limit to add this session
                    self._memory._sessions[session_id] = session
                except Exception:
                    pass  # Memory full, but return the session anyway
                return session

        return None

    async def _load_from_storage(self, session_id: str) -> Session | None:
        """Load a session from storage, with lazy loading for large graphs."""
        if not self._storage:
            return None

        # Check size for lazy loading decision
        size = await self._storage.get_session_size(session_id)

        if size > self._lazy_load_threshold:
            # Load session without full graph (lazy load)
            # For now, load full session - lazy loading can be optimized later
            logger.debug(
                f"Loading large session {session_id} ({size / 1024:.1f}KB)"
            )

        return await self._storage.load_session(session_id)

    async def update(self, session_id: str, session: Session) -> bool:
        """Update an existing session.

        Args:
            session_id: The unique identifier of the session
            session: The updated Session object

        Returns:
            True if updated, False if not found
        """
        # Update in memory
        updated = await self._memory.update(session_id, session)

        if updated:
            # Mark as dirty
            async with self._dirty_lock:
                self._dirty_sessions.add(session_id)

        return updated

    async def delete(self, session_id: str) -> bool:
        """Delete a session from memory and storage.

        Args:
            session_id: The unique identifier of the session

        Returns:
            True if deleted, False if not found
        """
        # Delete from memory
        deleted_memory = await self._memory.delete(session_id)

        # Delete from storage
        deleted_storage = False
        if self._storage:
            deleted_storage = await self._storage.delete_session(session_id)

        # Remove from dirty set
        async with self._dirty_lock:
            self._dirty_sessions.discard(session_id)

        return deleted_memory or deleted_storage

    async def persist(self, session_id: str) -> bool:
        """Explicitly persist a session to disk.

        Args:
            session_id: The session ID to persist

        Returns:
            True if persisted, False if not found or no storage
        """
        if not self._storage:
            return False

        session = await self._memory.get(session_id)
        if session is None:
            return False

        success = await self._storage.save_session(session)

        if success:
            async with self._dirty_lock:
                self._dirty_sessions.discard(session_id)

        return success

    async def persist_dirty(self) -> int:
        """Persist all dirty sessions to disk.

        Returns:
            Number of sessions persisted
        """
        if not self._storage:
            return 0

        async with self._dirty_lock:
            dirty_ids = list(self._dirty_sessions)

        persisted = 0
        for session_id in dirty_ids:
            if await self.persist(session_id):
                persisted += 1

        if persisted > 0:
            logger.debug(f"Persisted {persisted} dirty sessions")

        return persisted

    async def persist_all(self) -> int:
        """Persist all sessions in memory to disk.

        Returns:
            Number of sessions persisted
        """
        if not self._storage:
            return 0

        sessions = await self._memory.list_sessions(limit=self._memory._max_sessions)
        persisted = 0

        for session in sessions:
            if await self._storage.save_session(session):
                persisted += 1
                async with self._dirty_lock:
                    self._dirty_sessions.discard(session.id)

        logger.info(f"Persisted {persisted}/{len(sessions)} sessions")
        return persisted

    async def recover_sessions(self, max_age_hours: int = 24) -> int:
        """Recover sessions from disk storage.

        Loads sessions from disk that are not older than max_age_hours
        and adds them to the in-memory manager.

        Args:
            max_age_hours: Maximum age in hours for recovered sessions

        Returns:
            Number of sessions recovered
        """
        if not self._storage:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        recovered = 0

        session_ids = await self._storage.list_session_ids()
        logger.info(f"Found {len(session_ids)} sessions in storage for recovery")

        for session_id in session_ids:
            try:
                # Get metadata first to check age
                metadata = await self._storage.get_session_metadata(session_id)
                if metadata is None:
                    continue

                # Check age
                created_at_str = metadata.get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    # Normalize to UTC for comparison
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    if created_at < cutoff:
                        logger.debug(
                            f"Skipping old session {session_id} "
                            f"(created {created_at})"
                        )
                        continue

                # Check if already in memory
                if await self._memory.get(session_id) is not None:
                    continue

                # Load the session
                session = await self._storage.load_session(session_id)
                if session is None:
                    continue

                # Add to memory (bypass normal create to keep same ID)
                async with self._memory._lock:
                    if len(self._memory._sessions) < self._memory._max_sessions:
                        self._memory._sessions[session_id] = session
                        recovered += 1
                        logger.debug(f"Recovered session {session_id}")
                    else:
                        logger.warning(
                            f"Memory limit reached, cannot recover {session_id}"
                        )
                        break

            except Exception as e:
                logger.error(f"Failed to recover session {session_id}: {e}")

        logger.info(f"Recovered {recovered} sessions from storage")
        return recovered

    async def list_sessions(
        self,
        *,
        status: SessionStatus | None = None,
        limit: int = 100,
        include_disk: bool = False,
    ) -> list[Session]:
        """List sessions with optional status filter.

        Args:
            status: Optional status filter
            limit: Maximum number of sessions to return
            include_disk: Whether to include sessions only on disk

        Returns:
            List of sessions matching the criteria
        """
        # Get from memory
        sessions = await self._memory.list_sessions(status=status, limit=limit)

        if include_disk and self._storage and len(sessions) < limit:
            # Get additional sessions from disk
            memory_ids = {s.id for s in sessions}
            disk_ids = await self._storage.list_session_ids()

            for session_id in disk_ids:
                if session_id in memory_ids:
                    continue
                if len(sessions) >= limit:
                    break

                session = await self._storage.load_session(session_id)
                if session is not None:
                    if status is None or session.status == status:
                        sessions.append(session)

        return sessions[:limit]

    async def count(self, include_disk: bool = False) -> int:
        """Get the number of sessions.

        Args:
            include_disk: Whether to include sessions only on disk

        Returns:
            Total session count
        """
        memory_count = await self._memory.count()

        if include_disk and self._storage:
            disk_ids = await self._storage.list_session_ids()
            memory_ids = set(self._memory._sessions.keys())
            disk_only = len(set(disk_ids) - memory_ids)
            return memory_count + disk_only

        return memory_count

    async def cleanup_expired(self, max_age_seconds: int = 86400) -> int:
        """Remove sessions older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of sessions removed
        """
        # Clean up memory
        removed = await self._memory.cleanup_expired(max_age_seconds)

        # Clean up dirty tracking
        async with self._dirty_lock:
            # Get current memory session IDs
            current_ids = set(self._memory._sessions.keys())
            # Remove deleted sessions from dirty set
            self._dirty_sessions &= current_ids

        return removed

    async def clear(self) -> None:
        """Remove all sessions from memory and optionally from disk."""
        await self._memory.clear()

        async with self._dirty_lock:
            self._dirty_sessions.clear()

    async def shutdown(self) -> None:
        """Shutdown the manager, persisting all dirty sessions.

        This should be called during server shutdown to ensure
        all sessions are persisted.
        """
        self._shutdown = True

        # Cancel auto-persist task
        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass

        # Final persist
        if self._storage:
            await self.persist_all()
            await self._storage.close()

        logger.info("HybridSessionManager shutdown complete")

    # Expose underlying memory manager for compatibility
    @property
    def _sessions(self) -> dict:
        """Access internal sessions dict (for compatibility)."""
        return self._memory._sessions
