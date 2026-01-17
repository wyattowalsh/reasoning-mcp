"""Disk-based session storage using diskcache.

This module implements persistent session storage using the diskcache library,
enabling sessions to survive server restarts. Supports separate caching of
sessions and graphs for lazy loading of large thought graphs.

Features:
- Error recovery with configurable retry logic
- Operation timeouts to prevent blocking
- Graceful degradation on transient failures
- Detailed error classification for debugging
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from reasoning_mcp.storage.base import StorageBackend

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session
    from reasoning_mcp.models.thought import ThoughtGraph

logger = logging.getLogger(__name__)

# Prefixes for cache keys to separate sessions from graphs
SESSION_PREFIX = "session:"
GRAPH_PREFIX = "graph:"
METADATA_PREFIX = "meta:"

# Type variable for generic return types
T = TypeVar("T")


class StorageError(Exception):
    """Base exception for storage-related errors."""

    def __init__(self, message: str, operation: str | None = None) -> None:
        """Initialize storage error.

        Args:
            message: Error description
            operation: The operation that failed (e.g., "save_session", "load_graph")
        """
        super().__init__(message)
        self.operation = operation


class StorageTimeoutError(StorageError):
    """Raised when a storage operation times out."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Error description
            operation: The operation that timed out
            timeout_seconds: The timeout value that was exceeded
        """
        super().__init__(message, operation)
        self.timeout_seconds = timeout_seconds


class StorageIOError(StorageError):
    """Raised when a storage I/O operation fails."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize I/O error.

        Args:
            message: Error description
            operation: The operation that failed
            original_error: The underlying exception
        """
        super().__init__(message, operation)
        self.original_error = original_error


class StorageCorruptionError(StorageError):
    """Raised when stored data is corrupted or invalid."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize corruption error.

        Args:
            message: Error description
            operation: The operation that detected corruption
            session_id: The affected session ID
        """
        super().__init__(message, operation)
        self.session_id = session_id


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        exponential_base: float = 2.0,
    ) -> None:
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: The current retry attempt number (0-indexed)

        Returns:
            Delay in seconds before the next retry
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


class DiskSessionStorage(StorageBackend):
    """Disk-based session storage using diskcache.

    DiskSessionStorage persists sessions to disk using SQLite-backed caching,
    enabling recovery across server restarts. Features include:
    - Separate caches for sessions vs. graphs (lazy loading)
    - TTL-based expiration
    - Size-limited with LRU eviction
    - Thread-safe via executor
    - Configurable operation timeouts
    - Automatic retry with exponential backoff
    - Graceful error recovery

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

        Configure custom timeouts and retries:
        >>> storage = DiskSessionStorage(
        ...     operation_timeout=10.0,
        ...     retry_config=RetryConfig(max_retries=5, base_delay=0.2),
        ... )
    """

    # Default retry configuration
    DEFAULT_RETRY_CONFIG = RetryConfig(max_retries=3, base_delay=0.1, max_delay=5.0)

    # Default operation timeout in seconds
    DEFAULT_OPERATION_TIMEOUT = 30.0

    def __init__(
        self,
        cache_dir: Path | str = "~/.reasoning-mcp/sessions",
        size_limit_mb: int = 500,
        default_ttl: int = 86400,
        eviction_policy: str = "least-recently-used",
        operation_timeout: float | None = None,
        retry_config: RetryConfig | None = None,
        raise_on_error: bool = False,
    ):
        """Initialize disk storage.

        Args:
            cache_dir: Directory for cache files (default: ~/.reasoning-mcp/sessions)
            size_limit_mb: Maximum cache size in megabytes (default: 500)
            default_ttl: Default time-to-live in seconds (default: 86400 = 24h)
            eviction_policy: Eviction policy (default: "least-recently-used")
            operation_timeout: Timeout for individual operations in seconds (default: 30.0)
            retry_config: Configuration for retry behavior (default: 3 retries with exponential backoff)
            raise_on_error: If True, raise exceptions instead of returning False/None (default: False)
        """
        self._cache_dir = Path(cache_dir).expanduser()
        self._size_limit = size_limit_mb * 1024 * 1024  # Convert to bytes
        self._default_ttl = default_ttl
        self._eviction_policy = eviction_policy
        self._operation_timeout = operation_timeout or self.DEFAULT_OPERATION_TIMEOUT
        self._retry_config = retry_config or self.DEFAULT_RETRY_CONFIG
        self._raise_on_error = raise_on_error

        # Thread pool for running blocking diskcache operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="disk_storage")

        # Lazy-initialized cache
        self._cache: Any = None
        self._initialized = False

        # Error statistics for monitoring
        self._error_counts: dict[str, int] = {}
        self._last_error_time: float | None = None

    def _ensure_initialized(self) -> None:
        """Ensure the cache is initialized (lazy initialization).

        Raises:
            ImportError: If diskcache is not installed
            StorageIOError: If the cache directory cannot be created or accessed
        """
        if self._initialized:
            return

        try:
            import diskcache
        except ImportError:
            logger.warning(
                "diskcache not available. Install with: pip install reasoning-mcp[cache]"
            )
            raise

        try:
            # Create cache directory if needed
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            error_msg = f"Permission denied creating cache directory: {self._cache_dir}"
            logger.error(error_msg)
            raise StorageIOError(
                error_msg, operation="initialize", original_error=e
            ) from e
        except OSError as e:
            error_msg = f"Failed to create cache directory {self._cache_dir}: {e}"
            logger.error(error_msg)
            raise StorageIOError(
                error_msg, operation="initialize", original_error=e
            ) from e

        try:
            # Initialize cache with settings
            self._cache = diskcache.Cache(
                str(self._cache_dir),
                size_limit=self._size_limit,
                eviction_policy=self._eviction_policy,
            )
            self._initialized = True
            logger.info(
                f"DiskSessionStorage initialized at {self._cache_dir} "
                f"(size_limit={self._size_limit / (1024 * 1024):.0f}MB, "
                f"ttl={self._default_ttl}s, timeout={self._operation_timeout}s)"
            )
        except Exception as e:
            error_msg = f"Failed to initialize diskcache at {self._cache_dir}: {e}"
            logger.error(error_msg)
            raise StorageIOError(
                error_msg, operation="initialize", original_error=e
            ) from e

    def _record_error(self, operation: str, error: Exception) -> None:
        """Record an error for monitoring purposes.

        Args:
            operation: The operation that failed
            error: The exception that occurred
        """
        self._error_counts[operation] = self._error_counts.get(operation, 0) + 1
        self._last_error_time = time.time()
        logger.debug(
            f"Error recorded for {operation}: {error} "
            f"(total errors for this operation: {self._error_counts[operation]})"
        )

    def _handle_error(
        self,
        operation: str,
        error: Exception,
        default_return: T,
        session_id: str | None = None,
    ) -> T:
        """Handle an error based on configuration.

        Args:
            operation: The operation that failed
            error: The exception that occurred
            default_return: Value to return if not raising
            session_id: Optional session ID for context

        Returns:
            default_return if not raising

        Raises:
            StorageError: If raise_on_error is True
        """
        self._record_error(operation, error)

        if self._raise_on_error:
            if isinstance(error, StorageError):
                raise error
            elif isinstance(error, json.JSONDecodeError):
                raise StorageCorruptionError(
                    f"Corrupted data in {operation}: {error}",
                    operation=operation,
                    session_id=session_id,
                ) from error
            else:
                raise StorageIOError(
                    f"I/O error in {operation}: {error}",
                    operation=operation,
                    original_error=error,
                ) from error

        return default_return

    async def _run_in_executor(self, func: Callable[..., T], *args: Any) -> T:
        """Run a blocking function in the thread pool executor with timeout.

        Args:
            func: The blocking function to execute
            *args: Arguments to pass to the function

        Returns:
            The result of the function

        Raises:
            StorageTimeoutError: If the operation exceeds the timeout
            asyncio.CancelledError: If the operation is cancelled
        """
        loop = asyncio.get_running_loop()
        try:
            result: T = await asyncio.wait_for(
                loop.run_in_executor(self._executor, func, *args),
                timeout=self._operation_timeout,
            )
            return result
        except TimeoutError as e:
            func_name = getattr(func, "__name__", str(func))
            error_msg = (
                f"Operation {func_name} timed out after {self._operation_timeout}s"
            )
            logger.error(error_msg)
            raise StorageTimeoutError(
                error_msg,
                operation=func_name,
                timeout_seconds=self._operation_timeout,
            ) from e
        except FuturesTimeoutError as e:
            func_name = getattr(func, "__name__", str(func))
            error_msg = (
                f"Executor operation {func_name} timed out after {self._operation_timeout}s"
            )
            logger.error(error_msg)
            raise StorageTimeoutError(
                error_msg,
                operation=func_name,
                timeout_seconds=self._operation_timeout,
            ) from e

    async def _run_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        operation_name: str | None = None,
    ) -> T:
        """Run a function with retry logic and timeout.

        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            operation_name: Name of the operation for logging

        Returns:
            The result of the function

        Raises:
            StorageError: If all retries are exhausted
        """
        op_name = operation_name or getattr(func, "__name__", "unknown")
        last_error: Exception | None = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                return await self._run_in_executor(func, *args)
            except StorageTimeoutError:
                # Don't retry timeouts - they indicate a fundamental problem
                raise
            except asyncio.CancelledError:
                # Don't retry cancellations
                raise
            except Exception as e:
                last_error = e
                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        f"Operation {op_name} failed (attempt {attempt + 1}/"
                        f"{self._retry_config.max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation {op_name} failed after "
                        f"{self._retry_config.max_retries + 1} attempts: {e}"
                    )

        # All retries exhausted
        assert last_error is not None
        raise StorageIOError(
            f"Operation {op_name} failed after {self._retry_config.max_retries + 1} attempts",
            operation=op_name,
            original_error=last_error,
        ) from last_error

    def _serialize_session(self, session: Session) -> str:
        """Serialize a session to JSON (without the graph).

        Args:
            session: The session to serialize

        Returns:
            JSON string representation of the session

        Raises:
            StorageIOError: If serialization fails
        """
        try:
            # Get session data but exclude the graph for separate storage
            data = session.model_dump(mode="json")
            # Store graph separately - just keep a marker
            data["_graph_stored_separately"] = True
            data["graph"] = {"nodes": {}, "edges": {}}  # Empty placeholder
            return json.dumps(data)
        except (TypeError, ValueError) as e:
            raise StorageIOError(
                f"Failed to serialize session {session.id}: {e}",
                operation="serialize_session",
                original_error=e,
            ) from e

    def _deserialize_session(self, data: str, session_id: str | None = None) -> Session:
        """Deserialize a session from JSON.

        Args:
            data: JSON string to deserialize
            session_id: Optional session ID for error context

        Returns:
            Deserialized Session object

        Raises:
            StorageCorruptionError: If the data is corrupted or invalid
        """
        from reasoning_mcp.models.session import Session

        try:
            parsed = json.loads(data)
            # Remove marker if present
            parsed.pop("_graph_stored_separately", None)
            return Session.model_validate(parsed)
        except json.JSONDecodeError as e:
            raise StorageCorruptionError(
                f"Corrupted session data (invalid JSON): {e}",
                operation="deserialize_session",
                session_id=session_id,
            ) from e
        except Exception as e:
            raise StorageCorruptionError(
                f"Failed to deserialize session: {e}",
                operation="deserialize_session",
                session_id=session_id,
            ) from e

    def _serialize_graph(self, graph: ThoughtGraph) -> str:
        """Serialize a thought graph to JSON.

        Args:
            graph: The thought graph to serialize

        Returns:
            JSON string representation of the graph

        Raises:
            StorageIOError: If serialization fails
        """
        try:
            return json.dumps(graph.model_dump(mode="json"))
        except (TypeError, ValueError) as e:
            raise StorageIOError(
                f"Failed to serialize graph: {e}",
                operation="serialize_graph",
                original_error=e,
            ) from e

    def _deserialize_graph(self, data: str, session_id: str | None = None) -> ThoughtGraph:
        """Deserialize a thought graph from JSON.

        Args:
            data: JSON string to deserialize
            session_id: Optional session ID for error context

        Returns:
            Deserialized ThoughtGraph object

        Raises:
            StorageCorruptionError: If the data is corrupted or invalid
        """
        from reasoning_mcp.models.thought import ThoughtGraph

        try:
            return ThoughtGraph.model_validate(json.loads(data))
        except json.JSONDecodeError as e:
            raise StorageCorruptionError(
                f"Corrupted graph data (invalid JSON): {e}",
                operation="deserialize_graph",
                session_id=session_id,
            ) from e
        except Exception as e:
            raise StorageCorruptionError(
                f"Failed to deserialize graph: {e}",
                operation="deserialize_graph",
                session_id=session_id,
            ) from e

    def _extract_metadata(self, session: Session) -> dict[str, Any]:
        """Extract session metadata for quick access."""
        return {
            "id": session.id,
            "status": session.status.value
            if hasattr(session.status, "value")
            else str(session.status),
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
                    session_id = key[len(SESSION_PREFIX) :]
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

    def _get_session_metadata_sync(self, session_id: str) -> dict[str, Any] | None:
        """Synchronous metadata get (runs in executor)."""
        self._ensure_initialized()

        try:
            meta_key = f"{METADATA_PREFIX}{session_id}"
            meta_data = self._cache.get(meta_key)
            if meta_data is None:
                return None
            return cast("dict[str, Any]", json.loads(meta_data))
        except Exception as e:
            logger.error(f"Failed to get metadata for session {session_id}: {e}")
            return None

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
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
                total_size += len(
                    session_data.encode("utf-8") if isinstance(session_data, str) else session_data
                )

            graph_data = self._cache.get(graph_key)
            if graph_data:
                total_size += len(
                    graph_data.encode("utf-8") if isinstance(graph_data, str) else graph_data
                )

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

    def get_stats(self) -> dict[str, Any]:
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

        def _clear() -> None:
            self._cache.clear()

        try:
            await self._run_in_executor(_clear)
            logger.warning("All sessions cleared from disk storage")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
