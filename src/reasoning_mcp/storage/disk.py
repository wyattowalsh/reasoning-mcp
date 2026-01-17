"""Disk-based session storage using diskcache.

This module implements persistent session storage using the diskcache library,
enabling sessions to survive server restarts. Supports separate caching of
sessions and graphs for lazy loading of large thought graphs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reasoning_mcp.storage.base import StorageBackend

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session
    from reasoning_mcp.models.thought import ThoughtGraph

logger = logging.getLogger(__name__)

# Prefixes for cache keys to separate sessions from graphs
SESSION_PREFIX = "session:"
GRAPH_PREFIX = "graph:"
METADATA_PREFIX = "meta:"


class DiskSessionStorage(StorageBackend):
    """Disk-based session storage using diskcache.

    DiskSessionStorage persists sessions to disk using SQLite-backed caching,
    enabling recovery across server restarts. Features include:
    - Separate caches for sessions vs. graphs (lazy loading)
    - TTL-based expiration
    - Size-limited with LRU eviction
    - Thread-safe via executor

    The storage uses two logical partitions:
    - Session metadata (small, always loaded)
    - Thought graphs (potentially large, lazy loaded)

    Examples:
        Create a disk storage backend:
        >>> storage = DiskSessionStorage(
        ...     cache_dir=Path("~/.reasoning-mcp/sessions"),
        ...     size_limit_mb=500,
        ...     default_ttl=86400,  # 24 hours
        ... )

        Save and load a session:
        >>> session = Session()
        >>> await storage.save_session(session)
        >>> loaded = await storage.load_session(session.id)
        >>> assert loaded.id == session.id
    """

    def __init__(
        self,
        cache_dir: Path | str = "~/.reasoning-mcp/sessions",
        size_limit_mb: int = 500,
        default_ttl: int = 86400,
        eviction_policy: str = "least-recently-used",
    ):
        """Initialize disk storage.

        Args:
            cache_dir: Directory for cache files (default: ~/.reasoning-mcp/sessions)
            size_limit_mb: Maximum cache size in megabytes (default: 500)
            default_ttl: Default time-to-live in seconds (default: 86400 = 24h)
            eviction_policy: Eviction policy (default: "least-recently-used")
        """
        self._cache_dir = Path(cache_dir).expanduser()
        self._size_limit = size_limit_mb * 1024 * 1024  # Convert to bytes
        self._default_ttl = default_ttl
        self._eviction_policy = eviction_policy

        # Thread pool for running blocking diskcache operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="disk_storage")

        # Lazy-initialized cache
        self._cache: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the cache is initialized (lazy initialization)."""
        if self._initialized:
            return

        try:
            import diskcache

            # Create cache directory if needed
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Initialize cache with settings
            self._cache = diskcache.Cache(
                str(self._cache_dir),
                size_limit=self._size_limit,
                eviction_policy=self._eviction_policy,
            )
            self._initialized = True
            logger.info(
                f"DiskSessionStorage initialized at {self._cache_dir} "
                f"(size_limit={self._size_limit / (1024*1024):.0f}MB, "
                f"ttl={self._default_ttl}s)"
            )
        except ImportError:
            logger.warning(
                "diskcache not available. Install with: pip install reasoning-mcp[cache]"
            )
            raise

    async def _run_in_executor(self, func, *args):
        """Run a blocking function in the thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    def _serialize_session(self, session: Session) -> str:
        """Serialize a session to JSON (without the graph)."""
        # Get session data but exclude the graph for separate storage
        data = session.model_dump(mode="json")
        # Store graph separately - just keep a marker
        data["_graph_stored_separately"] = True
        data["graph"] = {"nodes": {}, "edges": {}}  # Empty placeholder
        return json.dumps(data)

    def _deserialize_session(self, data: str) -> Session:
        """Deserialize a session from JSON."""
        from reasoning_mcp.models.session import Session

        parsed = json.loads(data)
        # Remove marker if present
        parsed.pop("_graph_stored_separately", None)
        return Session.model_validate(parsed)

    def _serialize_graph(self, graph: ThoughtGraph) -> str:
        """Serialize a thought graph to JSON."""
        return json.dumps(graph.model_dump(mode="json"))

    def _deserialize_graph(self, data: str) -> ThoughtGraph:
        """Deserialize a thought graph from JSON."""
        from reasoning_mcp.models.thought import ThoughtGraph

        return ThoughtGraph.model_validate(json.loads(data))

    def _extract_metadata(self, session: Session) -> dict:
        """Extract session metadata for quick access."""
        return {
            "id": session.id,
            "status": session.status.value if hasattr(session.status, "value") else str(session.status),
            "created_at": session.created_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "thought_count": session.thought_count,
            "current_depth": session.current_depth,
            "error": session.error,
        }

    def _save_session_sync(self, session: Session) -> bool:
        """Synchronous session save (runs in executor)."""
        self._ensure_initialized()

        try:
            session_key = f"{SESSION_PREFIX}{session.id}"
            graph_key = f"{GRAPH_PREFIX}{session.id}"
            meta_key = f"{METADATA_PREFIX}{session.id}"

            # Serialize and store session (without graph)
            session_data = self._serialize_session(session)
            self._cache.set(session_key, session_data, expire=self._default_ttl)

            # Serialize and store graph separately
            graph_data = self._serialize_graph(session.graph)
            self._cache.set(graph_key, graph_data, expire=self._default_ttl)

            # Store metadata for quick access
            metadata = self._extract_metadata(session)
            self._cache.set(meta_key, json.dumps(metadata), expire=self._default_ttl)

            return True
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")
            return False

    async def save_session(self, session: Session) -> bool:
        """Save a session to disk storage."""
        return await self._run_in_executor(self._save_session_sync, session)

    def _load_session_sync(self, session_id: str) -> Session | None:
        """Synchronous session load (runs in executor)."""
        self._ensure_initialized()

        try:
            session_key = f"{SESSION_PREFIX}{session_id}"
            graph_key = f"{GRAPH_PREFIX}{session_id}"

            # Load session data
            session_data = self._cache.get(session_key)
            if session_data is None:
                return None

            # Deserialize session
            session = self._deserialize_session(session_data)

            # Load and attach graph
            graph_data = self._cache.get(graph_key)
            if graph_data:
                session.graph = self._deserialize_graph(graph_data)

            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def load_session(self, session_id: str) -> Session | None:
        """Load a session from disk storage."""
        return await self._run_in_executor(self._load_session_sync, session_id)

    def _delete_session_sync(self, session_id: str) -> bool:
        """Synchronous session delete (runs in executor)."""
        self._ensure_initialized()

        try:
            session_key = f"{SESSION_PREFIX}{session_id}"
            graph_key = f"{GRAPH_PREFIX}{session_id}"
            meta_key = f"{METADATA_PREFIX}{session_id}"

            # Delete all related keys
            deleted = False
            if session_key in self._cache:
                del self._cache[session_key]
                deleted = True
            if graph_key in self._cache:
                del self._cache[graph_key]
            if meta_key in self._cache:
                del self._cache[meta_key]

            return deleted
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk storage."""
        return await self._run_in_executor(self._delete_session_sync, session_id)

    def _list_session_ids_sync(self) -> list[str]:
        """Synchronous list session IDs (runs in executor)."""
        self._ensure_initialized()

        try:
            session_ids = []
            for key in self._cache:
                if isinstance(key, str) and key.startswith(SESSION_PREFIX):
                    session_id = key[len(SESSION_PREFIX):]
                    session_ids.append(session_id)
            return session_ids
        except Exception as e:
            logger.error(f"Failed to list session IDs: {e}")
            return []

    async def list_session_ids(self) -> list[str]:
        """List all session IDs in storage."""
        return await self._run_in_executor(self._list_session_ids_sync)

    def _save_graph_sync(self, session_id: str, graph: ThoughtGraph) -> bool:
        """Synchronous graph save (runs in executor)."""
        self._ensure_initialized()

        try:
            graph_key = f"{GRAPH_PREFIX}{session_id}"
            graph_data = self._serialize_graph(graph)
            self._cache.set(graph_key, graph_data, expire=self._default_ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to save graph for session {session_id}: {e}")
            return False

    async def save_graph(self, session_id: str, graph: ThoughtGraph) -> bool:
        """Save a thought graph separately."""
        return await self._run_in_executor(self._save_graph_sync, session_id, graph)

    def _load_graph_sync(self, session_id: str) -> ThoughtGraph | None:
        """Synchronous graph load (runs in executor)."""
        self._ensure_initialized()

        try:
            graph_key = f"{GRAPH_PREFIX}{session_id}"
            graph_data = self._cache.get(graph_key)
            if graph_data is None:
                return None
            return self._deserialize_graph(graph_data)
        except Exception as e:
            logger.error(f"Failed to load graph for session {session_id}: {e}")
            return None

    async def load_graph(self, session_id: str) -> ThoughtGraph | None:
        """Load a thought graph from storage."""
        return await self._run_in_executor(self._load_graph_sync, session_id)

    def _get_session_metadata_sync(self, session_id: str) -> dict | None:
        """Synchronous metadata get (runs in executor)."""
        self._ensure_initialized()

        try:
            meta_key = f"{METADATA_PREFIX}{session_id}"
            meta_data = self._cache.get(meta_key)
            if meta_data is None:
                return None
            return json.loads(meta_data)
        except Exception as e:
            logger.error(f"Failed to get metadata for session {session_id}: {e}")
            return None

    async def get_session_metadata(self, session_id: str) -> dict | None:
        """Get session metadata without loading the full graph."""
        return await self._run_in_executor(self._get_session_metadata_sync, session_id)

    def _session_exists_sync(self, session_id: str) -> bool:
        """Synchronous existence check (runs in executor)."""
        self._ensure_initialized()

        session_key = f"{SESSION_PREFIX}{session_id}"
        return session_key in self._cache

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists in storage."""
        return await self._run_in_executor(self._session_exists_sync, session_id)

    def _get_session_size_sync(self, session_id: str) -> int:
        """Synchronous size check (runs in executor)."""
        self._ensure_initialized()

        try:
            session_key = f"{SESSION_PREFIX}{session_id}"
            graph_key = f"{GRAPH_PREFIX}{session_id}"

            total_size = 0

            session_data = self._cache.get(session_key)
            if session_data:
                total_size += len(session_data.encode("utf-8") if isinstance(session_data, str) else session_data)

            graph_data = self._cache.get(graph_key)
            if graph_data:
                total_size += len(graph_data.encode("utf-8") if isinstance(graph_data, str) else graph_data)

            return total_size
        except Exception as e:
            logger.error(f"Failed to get size for session {session_id}: {e}")
            return 0

    async def get_session_size(self, session_id: str) -> int:
        """Get the approximate size of a session in bytes."""
        return await self._run_in_executor(self._get_session_size_sync, session_id)

    def _close_sync(self) -> None:
        """Synchronous close (runs in executor)."""
        if self._cache is not None:
            try:
                self._cache.close()
                logger.info("DiskSessionStorage closed")
            except Exception as e:
                logger.error(f"Error closing cache: {e}")

    async def close(self) -> None:
        """Close the storage backend and release resources."""
        await self._run_in_executor(self._close_sync)
        self._executor.shutdown(wait=True)
        self._initialized = False

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics (size, count, hits, misses, etc.)
        """
        self._ensure_initialized()

        try:
            return {
                "size": self._cache.size,
                "volume": self._cache.volume(),
                "count": len(self._cache),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def clear(self) -> None:
        """Clear all cached sessions.

        Warning: This will delete all persisted sessions!
        """
        self._ensure_initialized()

        def _clear():
            self._cache.clear()

        try:
            await self._run_in_executor(_clear)
            logger.warning("All sessions cleared from disk storage")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
