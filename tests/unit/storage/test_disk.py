"""Unit tests for DiskSessionStorage.

Tests disk-based persistence using diskcache.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.storage.disk import (
    DiskSessionStorage,
    RetryConfig,
    StorageCorruptionError,
    StorageError,
    StorageIOError,
    StorageTimeoutError,
)


class TestStorageError:
    """Tests for StorageError base exception."""

    def test_storage_error_with_message(self) -> None:
        """Test StorageError with message only."""
        error = StorageError("Test error message")
        assert str(error) == "Test error message"
        assert error.operation is None

    def test_storage_error_with_operation(self) -> None:
        """Test StorageError with operation."""
        error = StorageError("Test error", operation="save_session")
        assert str(error) == "Test error"
        assert error.operation == "save_session"


class TestStorageTimeoutError:
    """Tests for StorageTimeoutError exception."""

    def test_timeout_error_with_all_args(self) -> None:
        """Test StorageTimeoutError with all arguments."""
        error = StorageTimeoutError(
            "Operation timed out",
            operation="load_session",
            timeout_seconds=30.0,
        )
        assert str(error) == "Operation timed out"
        assert error.operation == "load_session"
        assert error.timeout_seconds == 30.0

    def test_timeout_error_minimal(self) -> None:
        """Test StorageTimeoutError with minimal arguments."""
        error = StorageTimeoutError("Timeout")
        assert str(error) == "Timeout"
        assert error.operation is None
        assert error.timeout_seconds is None


class TestStorageIOError:
    """Tests for StorageIOError exception."""

    def test_io_error_with_original_error(self) -> None:
        """Test StorageIOError with original error."""
        original = IOError("Disk full")
        error = StorageIOError(
            "Failed to write",
            operation="save_session",
            original_error=original,
        )
        assert str(error) == "Failed to write"
        assert error.operation == "save_session"
        assert error.original_error is original

    def test_io_error_without_original(self) -> None:
        """Test StorageIOError without original error."""
        error = StorageIOError("I/O failure")
        assert error.original_error is None


class TestStorageCorruptionError:
    """Tests for StorageCorruptionError exception."""

    def test_corruption_error_with_session_id(self) -> None:
        """Test StorageCorruptionError with session ID."""
        error = StorageCorruptionError(
            "Data corruption detected",
            operation="load_graph",
            session_id="session-123",
        )
        assert str(error) == "Data corruption detected"
        assert error.operation == "load_graph"
        assert error.session_id == "session-123"

    def test_corruption_error_without_session_id(self) -> None:
        """Test StorageCorruptionError without session ID."""
        error = StorageCorruptionError("Corrupted data")
        assert error.session_id is None


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 0.1
        assert config.max_delay == 5.0
        assert config.exponential_base == 2.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=10.0,
            exponential_base=3.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0
        assert config.exponential_base == 3.0

    def test_get_delay_first_attempt(self) -> None:
        """Test delay calculation for first attempt."""
        config = RetryConfig(base_delay=0.1, exponential_base=2.0)
        delay = config.get_delay(0)
        assert delay == 0.1  # 0.1 * 2^0 = 0.1

    def test_get_delay_exponential_backoff(self) -> None:
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(base_delay=0.1, exponential_base=2.0, max_delay=10.0)
        assert config.get_delay(1) == 0.2  # 0.1 * 2^1 = 0.2
        assert config.get_delay(2) == 0.4  # 0.1 * 2^2 = 0.4
        assert config.get_delay(3) == 0.8  # 0.1 * 2^3 = 0.8

    def test_get_delay_capped_at_max(self) -> None:
        """Test delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=5.0)
        assert config.get_delay(5) == 5.0  # 1.0 * 2^5 = 32, capped at 5.0


class TestDiskSessionStorageInitialization:
    """Tests for DiskSessionStorage initialization."""

    def test_create_instance_default_path(self) -> None:
        """Test creating instance with default path."""
        storage = DiskSessionStorage()
        assert storage is not None
        assert storage._initialized is False

    def test_create_instance_custom_path(self) -> None:
        """Test creating instance with custom path."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        assert "/tmp/test_cache" in str(storage._cache_dir)

    def test_create_instance_custom_size_limit(self) -> None:
        """Test creating instance with custom size limit."""
        storage = DiskSessionStorage(size_limit_mb=100)
        assert storage._size_limit == 100 * 1024 * 1024

    def test_create_instance_custom_ttl(self) -> None:
        """Test creating instance with custom TTL."""
        storage = DiskSessionStorage(default_ttl=3600)
        assert storage._default_ttl == 3600

    def test_create_instance_custom_eviction_policy(self) -> None:
        """Test creating instance with custom eviction policy."""
        storage = DiskSessionStorage(eviction_policy="least-frequently-used")
        assert storage._eviction_policy == "least-frequently-used"


class TestDiskSessionStorageEnsureInitialized:
    """Tests for DiskSessionStorage ensure_initialized."""

    def test_ensure_initialized_sets_flag(self) -> None:
        """Test _ensure_initialized sets the initialized flag."""
        with patch("diskcache.Cache"):
            storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
            storage._ensure_initialized()
            assert storage._initialized is True

    def test_ensure_initialized_creates_cache(self) -> None:
        """Test _ensure_initialized creates the cache."""
        with patch("diskcache.Cache") as mock_cache:
            storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
            storage._ensure_initialized()
            assert mock_cache.called


class TestDiskSessionStorageSessionExists:
    """Tests for DiskSessionStorage session_exists."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a mock storage instance."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        return storage

    async def test_session_exists_returns_true(self, mock_storage: DiskSessionStorage) -> None:
        """Test session_exists returns True for existing session."""
        mock_storage._cache.__contains__ = MagicMock(return_value=True)
        result = await mock_storage.session_exists("test-123")
        assert result is True

    async def test_session_exists_returns_false(self, mock_storage: DiskSessionStorage) -> None:
        """Test session_exists returns False for non-existent session."""
        mock_storage._cache.__contains__ = MagicMock(return_value=False)
        result = await mock_storage.session_exists("nonexistent")
        assert result is False


class TestDiskSessionStorageListSessionIds:
    """Tests for DiskSessionStorage list_session_ids."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a mock storage instance."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        return storage

    async def test_list_empty(self, mock_storage: DiskSessionStorage) -> None:
        """Test listing when empty."""
        mock_storage._cache.__iter__ = MagicMock(return_value=iter([]))
        result = await mock_storage.list_session_ids()
        assert result == []

    async def test_list_multiple_sessions(self, mock_storage: DiskSessionStorage) -> None:
        """Test listing multiple sessions."""
        mock_storage._cache.__iter__ = MagicMock(
            return_value=iter(
                [
                    "session:test-1",
                    "session:test-2",
                    "graph:test-1",
                    "meta:test-1",
                ]
            )
        )
        result = await mock_storage.list_session_ids()
        assert "test-1" in result
        assert "test-2" in result
        assert len(result) == 2  # Only session: prefixed


class TestDiskSessionStorageGetStats:
    """Tests for DiskSessionStorage get_stats."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a mock storage instance."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        storage._cache.size = 1000
        storage._cache.volume = MagicMock(return_value=5000)
        storage._cache.__len__ = MagicMock(return_value=10)
        return storage

    def test_get_stats_returns_dict(self, mock_storage: DiskSessionStorage) -> None:
        """Test get_stats returns dictionary."""
        stats = mock_storage.get_stats()
        assert isinstance(stats, dict)
        assert "size" in stats
        assert "volume" in stats
        assert "count" in stats

    def test_get_stats_contains_expected_values(self, mock_storage: DiskSessionStorage) -> None:
        """Test get_stats contains expected values."""
        stats = mock_storage.get_stats()
        assert stats["size"] == 1000
        assert stats["volume"] == 5000
        assert stats["count"] == 10


class TestDiskSessionStorageClose:
    """Tests for DiskSessionStorage close method."""

    async def test_close_shuts_down_executor(self) -> None:
        """Test close shuts down the executor."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._cache = MagicMock()
        await storage.close()
        assert storage._executor._shutdown


class TestDiskSessionStorageClear:
    """Tests for DiskSessionStorage clear method."""

    async def test_clear_calls_cache_clear(self) -> None:
        """Test clear calls cache.clear()."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        await storage.clear()
        storage._cache.clear.assert_called_once()


class TestDiskSessionStorageRecordError:
    """Tests for DiskSessionStorage _record_error method."""

    def test_record_error_tracks_count(self) -> None:
        """Test _record_error increments error count."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        error = ValueError("test error")

        storage._record_error("save_session", error)
        assert storage._error_counts["save_session"] == 1

        storage._record_error("save_session", error)
        assert storage._error_counts["save_session"] == 2

    def test_record_error_updates_last_error_time(self) -> None:
        """Test _record_error updates last error time."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        assert storage._last_error_time is None

        storage._record_error("load_session", ValueError("test"))
        assert storage._last_error_time is not None
        assert isinstance(storage._last_error_time, float)


class TestDiskSessionStorageHandleError:
    """Tests for DiskSessionStorage _handle_error method."""

    def test_handle_error_returns_default_when_not_raising(self) -> None:
        """Test _handle_error returns default when raise_on_error=False."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache", raise_on_error=False)
        result = storage._handle_error("test_op", ValueError("test"), default_return=None)
        assert result is None

    def test_handle_error_raises_storage_error(self) -> None:
        """Test _handle_error re-raises StorageError when raise_on_error=True."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache", raise_on_error=True)
        original_error = StorageIOError("original error", operation="test")

        with pytest.raises(StorageIOError) as exc_info:
            storage._handle_error("test_op", original_error, default_return=None)
        assert exc_info.value is original_error

    def test_handle_error_raises_corruption_for_json_error(self) -> None:
        """Test _handle_error raises StorageCorruptionError for JSONDecodeError."""
        import json
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache", raise_on_error=True)
        json_error = json.JSONDecodeError("test", "doc", 0)

        with pytest.raises(StorageCorruptionError) as exc_info:
            storage._handle_error("parse_data", json_error, default_return=None, session_id="sess-1")
        assert exc_info.value.session_id == "sess-1"
        assert exc_info.value.operation == "parse_data"

    def test_handle_error_raises_io_error_for_other_exceptions(self) -> None:
        """Test _handle_error raises StorageIOError for other exceptions."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache", raise_on_error=True)
        generic_error = RuntimeError("something failed")

        with pytest.raises(StorageIOError) as exc_info:
            storage._handle_error("some_op", generic_error, default_return=None)
        assert exc_info.value.operation == "some_op"
        assert exc_info.value.original_error is generic_error


class TestDiskSessionStorageEnsureInitializedErrors:
    """Tests for DiskSessionStorage _ensure_initialized error paths."""

    def test_ensure_initialized_raises_on_permission_error(self) -> None:
        """Test _ensure_initialized raises StorageIOError on permission error."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Access denied")

            with pytest.raises(StorageIOError) as exc_info:
                storage._ensure_initialized()
            assert "Permission denied" in str(exc_info.value)
            assert exc_info.value.operation == "initialize"

    def test_ensure_initialized_raises_on_os_error(self) -> None:
        """Test _ensure_initialized raises StorageIOError on OS error."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = OSError("Disk error")

            with pytest.raises(StorageIOError) as exc_info:
                storage._ensure_initialized()
            assert exc_info.value.operation == "initialize"

    def test_ensure_initialized_raises_on_cache_creation_error(self) -> None:
        """Test _ensure_initialized raises StorageIOError on cache creation error."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")

        with patch("pathlib.Path.mkdir"):
            with patch("diskcache.Cache") as mock_cache:
                mock_cache.side_effect = Exception("Cache init failed")

                with pytest.raises(StorageIOError) as exc_info:
                    storage._ensure_initialized()
                assert "Failed to initialize diskcache" in str(exc_info.value)


class TestDiskSessionStorageRunInExecutor:
    """Tests for DiskSessionStorage _run_in_executor method."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a storage instance with mocked executor."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache", operation_timeout=1.0)
        return storage

    async def test_run_in_executor_success(self, mock_storage: DiskSessionStorage) -> None:
        """Test _run_in_executor returns result on success."""
        def simple_func(x: int) -> int:
            return x * 2

        result = await mock_storage._run_in_executor(simple_func, 5)
        assert result == 10

    async def test_run_in_executor_timeout(self, mock_storage: DiskSessionStorage) -> None:
        """Test _run_in_executor raises StorageTimeoutError on timeout."""
        import time as time_module

        def slow_func() -> None:
            time_module.sleep(5)

        with pytest.raises(StorageTimeoutError) as exc_info:
            await mock_storage._run_in_executor(slow_func)
        assert exc_info.value.timeout_seconds == 1.0


class TestDiskSessionStorageSerialization:
    """Tests for DiskSessionStorage serialization methods."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a storage instance."""
        return DiskSessionStorage(cache_dir="/tmp/test_cache")

    def test_serialize_graph_success(self, mock_storage: DiskSessionStorage) -> None:
        """Test _serialize_graph serializes ThoughtGraph to JSON."""
        from reasoning_mcp.models.thought import ThoughtGraph

        graph = ThoughtGraph()
        result = mock_storage._serialize_graph(graph)

        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_deserialize_graph_success(self, mock_storage: DiskSessionStorage) -> None:
        """Test _deserialize_graph deserializes JSON to ThoughtGraph."""
        from reasoning_mcp.models.thought import ThoughtGraph

        data = '{"nodes": {}, "edges": {}}'
        result = mock_storage._deserialize_graph(data)

        assert isinstance(result, ThoughtGraph)
        assert len(result.nodes) == 0

    def test_deserialize_graph_invalid_json(self, mock_storage: DiskSessionStorage) -> None:
        """Test _deserialize_graph raises StorageCorruptionError on invalid JSON."""
        with pytest.raises(StorageCorruptionError) as exc_info:
            mock_storage._deserialize_graph("not valid json", session_id="sess-1")
        assert exc_info.value.session_id == "sess-1"
        assert "invalid JSON" in str(exc_info.value)

    def test_deserialize_graph_with_invalid_node_type(self, mock_storage: DiskSessionStorage) -> None:
        """Test _deserialize_graph raises StorageCorruptionError on invalid node type."""
        # nodes must be a dict, not a list
        with pytest.raises(StorageCorruptionError) as exc_info:
            mock_storage._deserialize_graph('{"nodes": "not_a_dict", "edges": {}}')
        assert "Failed to deserialize graph" in str(exc_info.value)


class TestDiskSessionStorageSaveLoadSession:
    """Tests for DiskSessionStorage save/load session operations."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a mock storage instance."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        return storage

    def test_save_session_sync_success(self, mock_storage: DiskSessionStorage) -> None:
        """Test _save_session_sync saves session data correctly."""
        from reasoning_mcp.models.session import Session
        from reasoning_mcp.models.thought import ThoughtGraph

        session = Session(id="test-123", graph=ThoughtGraph())

        result = mock_storage._save_session_sync(session)

        assert result is True
        assert mock_storage._cache.set.call_count == 3  # session, graph, metadata

    def test_save_session_sync_handles_error(self, mock_storage: DiskSessionStorage) -> None:
        """Test _save_session_sync returns False on error."""
        from reasoning_mcp.models.session import Session
        from reasoning_mcp.models.thought import ThoughtGraph

        session = Session(id="test-123", graph=ThoughtGraph())
        mock_storage._cache.set.side_effect = Exception("Save failed")

        result = mock_storage._save_session_sync(session)

        assert result is False

    def test_load_session_sync_not_found(self, mock_storage: DiskSessionStorage) -> None:
        """Test _load_session_sync returns None for non-existent session."""
        mock_storage._cache.get.return_value = None

        result = mock_storage._load_session_sync("nonexistent")

        assert result is None

    async def test_save_session_async(self, mock_storage: DiskSessionStorage) -> None:
        """Test save_session async wrapper."""
        from reasoning_mcp.models.session import Session
        from reasoning_mcp.models.thought import ThoughtGraph

        session = Session(id="test-123", graph=ThoughtGraph())

        result = await mock_storage.save_session(session)

        assert result is True


class TestDiskSessionStorageDeleteSession:
    """Tests for DiskSessionStorage delete_session method."""

    @pytest.fixture
    def mock_storage(self) -> DiskSessionStorage:
        """Create a mock storage instance."""
        storage = DiskSessionStorage(cache_dir="/tmp/test_cache")
        storage._initialized = True
        storage._cache = MagicMock()
        return storage

    async def test_delete_session_success(self, mock_storage: DiskSessionStorage) -> None:
        """Test delete_session removes all session keys."""
        # Mock __contains__ to return True for session key (uses `in` operator)
        mock_storage._cache.__contains__ = MagicMock(return_value=True)
        # Mock __delitem__ (uses `del cache[key]` syntax)
        mock_storage._cache.__delitem__ = MagicMock()

        result = await mock_storage.delete_session("test-123")

        assert result is True
        # Should delete session, graph, and metadata keys via __delitem__
        assert mock_storage._cache.__delitem__.call_count == 3

    async def test_delete_session_not_found(self, mock_storage: DiskSessionStorage) -> None:
        """Test delete_session returns False for non-existent session."""
        mock_storage._cache.__contains__ = MagicMock(return_value=False)

        result = await mock_storage.delete_session("nonexistent")

        assert result is False
