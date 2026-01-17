"""Unit tests for HybridSessionManager.

Tests hybrid in-memory/disk storage with auto-persist.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.storage.hybrid import HybridSessionManager


class TestHybridSessionManagerInitialization:
    """Tests for HybridSessionManager initialization."""

    def test_create_instance_default(self) -> None:
        """Test creating instance with defaults."""
        manager = HybridSessionManager()
        assert manager is not None
        assert manager._storage is None

    def test_create_instance_with_storage(self) -> None:
        """Test creating instance with storage backend."""
        mock_storage = MagicMock()
        manager = HybridSessionManager(storage=mock_storage)
        assert manager._storage is mock_storage

    def test_create_instance_custom_max_sessions(self) -> None:
        """Test creating instance with custom max sessions."""
        manager = HybridSessionManager(max_sessions=200)
        assert manager._memory._max_sessions == 200

    def test_create_instance_custom_cleanup_interval(self) -> None:
        """Test creating instance with custom cleanup interval."""
        manager = HybridSessionManager(cleanup_interval=7200)
        assert manager._memory._cleanup_interval == 7200

    def test_create_instance_custom_lazy_threshold(self) -> None:
        """Test creating instance with custom lazy load threshold."""
        manager = HybridSessionManager(lazy_load_threshold_kb=100)
        assert manager._lazy_load_threshold == 100 * 1024

    def test_create_instance_custom_auto_persist_interval(self) -> None:
        """Test creating instance with custom auto persist interval."""
        manager = HybridSessionManager(auto_persist_interval=120)
        assert manager._auto_persist_interval == 120

    def test_create_instance_recovery_disabled(self) -> None:
        """Test creating instance with recovery disabled."""
        manager = HybridSessionManager(recovery_on_startup=False)
        assert manager._recovery_on_startup is False

    def test_create_instance_custom_max_recovery_age(self) -> None:
        """Test creating instance with custom max recovery age."""
        manager = HybridSessionManager(max_recovery_age_hours=48)
        assert manager._max_recovery_age_hours == 48


class TestHybridSessionManagerInitializeMethod:
    """Tests for HybridSessionManager initialize method."""

    async def test_initialize_no_storage_no_recovery(self) -> None:
        """Test initialize without storage doesn't recover."""
        manager = HybridSessionManager()
        await manager.initialize()
        # Should not raise and persist task should not be started
        assert manager._persist_task is None

    async def test_initialize_with_storage_starts_persist_task(self) -> None:
        """Test initialize with storage starts persist task."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=[])
        manager = HybridSessionManager(
            storage=mock_storage,
            recovery_on_startup=False,
            auto_persist_interval=60,
        )
        await manager.initialize()
        assert manager._persist_task is not None
        # Clean up
        manager._shutdown = True
        manager._persist_task.cancel()


class TestHybridSessionManagerDirtyTracking:
    """Tests for HybridSessionManager dirty session tracking."""

    def test_initial_dirty_sessions_empty(self) -> None:
        """Test initial dirty sessions set is empty."""
        manager = HybridSessionManager()
        assert len(manager._dirty_sessions) == 0


class TestHybridSessionManagerListSessions:
    """Tests for HybridSessionManager list_sessions."""

    async def test_list_sessions_delegates_to_memory(self) -> None:
        """Test list_sessions delegates to memory manager."""
        manager = HybridSessionManager()
        sessions = await manager.list_sessions()
        assert isinstance(sessions, list)


class TestHybridSessionManagerCount:
    """Tests for HybridSessionManager count."""

    async def test_count_returns_memory_count(self) -> None:
        """Test count returns memory session count."""
        manager = HybridSessionManager()
        count = await manager.count()
        assert count == 0


class TestHybridSessionManagerPersistDirty:
    """Tests for HybridSessionManager persist_dirty."""

    async def test_persist_dirty_without_storage(self) -> None:
        """Test persist_dirty without storage returns 0."""
        manager = HybridSessionManager()
        result = await manager.persist_dirty()
        assert result == 0


class TestHybridSessionManagerPersistAll:
    """Tests for HybridSessionManager persist_all."""

    async def test_persist_all_without_storage(self) -> None:
        """Test persist_all without storage returns 0."""
        manager = HybridSessionManager()
        result = await manager.persist_all()
        assert result == 0


class TestHybridSessionManagerRecoverSessions:
    """Tests for HybridSessionManager recover_sessions."""

    async def test_recover_without_storage(self) -> None:
        """Test recover_sessions without storage returns 0."""
        manager = HybridSessionManager()
        result = await manager.recover_sessions()
        assert result == 0


class TestHybridSessionManagerClear:
    """Tests for HybridSessionManager clear."""

    async def test_clear_clears_memory(self) -> None:
        """Test clear clears memory sessions."""
        manager = HybridSessionManager()
        manager._dirty_sessions.add("test-123")
        await manager.clear()
        assert len(manager._dirty_sessions) == 0


class TestHybridSessionManagerShutdown:
    """Tests for HybridSessionManager shutdown."""

    async def test_shutdown_sets_flag(self) -> None:
        """Test shutdown sets shutdown flag."""
        manager = HybridSessionManager()
        await manager.shutdown()
        assert manager._shutdown is True


class TestHybridSessionManagerSessionsProperty:
    """Tests for HybridSessionManager _sessions property."""

    def test_sessions_property_returns_dict(self) -> None:
        """Test _sessions property returns dict."""
        manager = HybridSessionManager()
        assert isinstance(manager._sessions, dict)


def create_mock_session(session_id: str = "test-session-123") -> MagicMock:
    """Helper to create a mock session."""
    session = MagicMock()
    session.id = session_id
    session.status = "active"
    session.created_at = datetime.now(UTC)
    return session


class TestHybridSessionManagerCreate:
    """Tests for HybridSessionManager create method."""

    @pytest.mark.asyncio
    async def test_create_returns_session(self) -> None:
        """Test create returns a session."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()

        with patch.object(manager._memory, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_session
            result = await manager.create()

            assert result is mock_session
            mock_create.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_create_with_config(self) -> None:
        """Test create passes config to memory manager."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()
        mock_config = MagicMock()

        with patch.object(manager._memory, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_session
            result = await manager.create(config=mock_config)

            assert result is mock_session
            mock_create.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_create_marks_session_dirty(self) -> None:
        """Test create marks the session as dirty."""
        manager = HybridSessionManager()
        mock_session = create_mock_session("dirty-session")

        with patch.object(manager._memory, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_session
            await manager.create()

            assert "dirty-session" in manager._dirty_sessions


class TestHybridSessionManagerGet:
    """Tests for HybridSessionManager get method."""

    @pytest.mark.asyncio
    async def test_get_returns_from_memory(self) -> None:
        """Test get returns session from memory."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_session
            result = await manager.get("test-session-123")

            assert result is mock_session
            mock_get.assert_called_once_with("test-session-123")

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_found(self) -> None:
        """Test get returns None when session not found."""
        manager = HybridSessionManager()

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            result = await manager.get("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_loads_from_storage_when_not_in_memory(self) -> None:
        """Test get loads from storage when not in memory."""
        mock_storage = AsyncMock()
        mock_session = create_mock_session("storage-session")
        mock_storage.get_session_size = AsyncMock(return_value=1024)
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            result = await manager.get("storage-session")

            assert result is mock_session
            mock_storage.load_session.assert_called_once_with("storage-session")

    @pytest.mark.asyncio
    async def test_get_adds_loaded_session_to_memory(self) -> None:
        """Test get adds loaded session to memory cache."""
        mock_storage = AsyncMock()
        mock_session = create_mock_session("loaded-session")
        mock_storage.get_session_size = AsyncMock(return_value=1024)
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            await manager.get("loaded-session")

            # Session should be added to memory sessions
            assert "loaded-session" in manager._memory._sessions

    @pytest.mark.asyncio
    async def test_get_handles_cache_add_failure(self) -> None:
        """Test get handles failure to add to cache gracefully."""
        mock_storage = AsyncMock()
        mock_session = create_mock_session("cache-fail-session")
        mock_storage.get_session_size = AsyncMock(return_value=1024)
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            # Mock _sessions to raise MemoryError
            with patch.object(
                manager._memory, "_sessions", new_callable=MagicMock
            ) as mock_sessions:
                mock_sessions.__setitem__ = MagicMock(side_effect=MemoryError("OOM"))
                result = await manager.get("cache-fail-session")

                # Should still return the session
                assert result is mock_session


class TestHybridSessionManagerLoadFromStorage:
    """Tests for HybridSessionManager _load_from_storage method."""

    @pytest.mark.asyncio
    async def test_load_returns_none_without_storage(self) -> None:
        """Test _load_from_storage returns None without storage."""
        manager = HybridSessionManager()
        result = await manager._load_from_storage("any-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_checks_session_size(self) -> None:
        """Test _load_from_storage checks session size."""
        mock_storage = AsyncMock()
        mock_session = create_mock_session()
        mock_storage.get_session_size = AsyncMock(return_value=2048)
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage)
        await manager._load_from_storage("test-id")

        mock_storage.get_session_size.assert_called_once_with("test-id")

    @pytest.mark.asyncio
    async def test_load_large_session_logs_debug(self) -> None:
        """Test loading large session logs debug message."""
        mock_storage = AsyncMock()
        mock_session = create_mock_session()
        # Large session - above threshold
        mock_storage.get_session_size = AsyncMock(return_value=100 * 1024)
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage, lazy_load_threshold_kb=50)

        with patch("reasoning_mcp.storage.hybrid.logger") as mock_logger:
            result = await manager._load_from_storage("large-session")

            assert result is mock_session
            mock_logger.debug.assert_called()


class TestHybridSessionManagerUpdate:
    """Tests for HybridSessionManager update method."""

    @pytest.mark.asyncio
    async def test_update_delegates_to_memory(self) -> None:
        """Test update delegates to memory manager."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()

        with patch.object(manager._memory, "update", new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            result = await manager.update("test-id", mock_session)

            assert result is True
            mock_update.assert_called_once_with("test-id", mock_session)

    @pytest.mark.asyncio
    async def test_update_marks_session_dirty_on_success(self) -> None:
        """Test update marks session as dirty on success."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()

        with patch.object(manager._memory, "update", new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            await manager.update("updated-session", mock_session)

            assert "updated-session" in manager._dirty_sessions

    @pytest.mark.asyncio
    async def test_update_does_not_mark_dirty_on_failure(self) -> None:
        """Test update does not mark session dirty on failure."""
        manager = HybridSessionManager()
        mock_session = create_mock_session()

        with patch.object(manager._memory, "update", new_callable=AsyncMock) as mock_update:
            mock_update.return_value = False
            await manager.update("nonexistent", mock_session)

            assert "nonexistent" not in manager._dirty_sessions


class TestHybridSessionManagerDelete:
    """Tests for HybridSessionManager delete method."""

    @pytest.mark.asyncio
    async def test_delete_from_memory_only(self) -> None:
        """Test delete from memory when no storage."""
        manager = HybridSessionManager()
        manager._dirty_sessions.add("to-delete")

        with patch.object(manager._memory, "delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            result = await manager.delete("to-delete")

            assert result is True
            assert "to-delete" not in manager._dirty_sessions

    @pytest.mark.asyncio
    async def test_delete_from_memory_and_storage(self) -> None:
        """Test delete from both memory and storage."""
        mock_storage = AsyncMock()
        mock_storage.delete_session = AsyncMock(return_value=True)

        manager = HybridSessionManager(storage=mock_storage)
        manager._dirty_sessions.add("stored-session")

        with patch.object(manager._memory, "delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            result = await manager.delete("stored-session")

            assert result is True
            mock_storage.delete_session.assert_called_once_with("stored-session")
            assert "stored-session" not in manager._dirty_sessions

    @pytest.mark.asyncio
    async def test_delete_returns_true_if_found_in_storage_only(self) -> None:
        """Test delete returns True if only found in storage."""
        mock_storage = AsyncMock()
        mock_storage.delete_session = AsyncMock(return_value=True)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = False  # Not in memory
            result = await manager.delete("storage-only-session")

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_if_not_found_anywhere(self) -> None:
        """Test delete returns False if session not found."""
        mock_storage = AsyncMock()
        mock_storage.delete_session = AsyncMock(return_value=False)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = False
            result = await manager.delete("nonexistent")

            assert result is False


class TestHybridSessionManagerPersist:
    """Tests for HybridSessionManager persist method."""

    @pytest.mark.asyncio
    async def test_persist_returns_false_without_storage(self) -> None:
        """Test persist returns False without storage."""
        manager = HybridSessionManager()
        result = await manager.persist("any-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_persist_returns_false_if_session_not_found(self) -> None:
        """Test persist returns False if session not in memory."""
        mock_storage = AsyncMock()
        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            result = await manager.persist("nonexistent")

            assert result is False

    @pytest.mark.asyncio
    async def test_persist_saves_session_to_storage(self) -> None:
        """Test persist saves session to storage."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=True)
        mock_session = create_mock_session("to-persist")

        manager = HybridSessionManager(storage=mock_storage)
        manager._dirty_sessions.add("to-persist")

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_session
            result = await manager.persist("to-persist")

            assert result is True
            mock_storage.save_session.assert_called_once_with(mock_session)
            # Should remove from dirty set
            assert "to-persist" not in manager._dirty_sessions

    @pytest.mark.asyncio
    async def test_persist_handles_save_failure(self) -> None:
        """Test persist handles save failure."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=False)
        mock_session = create_mock_session("fail-persist")

        manager = HybridSessionManager(storage=mock_storage)
        manager._dirty_sessions.add("fail-persist")

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_session
            result = await manager.persist("fail-persist")

            assert result is False
            # Should remain in dirty set
            assert "fail-persist" in manager._dirty_sessions


class TestHybridSessionManagerPersistDirtyWithSessions:
    """Tests for HybridSessionManager persist_dirty with actual sessions."""

    @pytest.mark.asyncio
    async def test_persist_dirty_persists_dirty_sessions(self) -> None:
        """Test persist_dirty persists all dirty sessions."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=True)

        manager = HybridSessionManager(storage=mock_storage)
        manager._dirty_sessions = {"session-1", "session-2", "session-3"}

        mock_sessions = {
            "session-1": create_mock_session("session-1"),
            "session-2": create_mock_session("session-2"),
            "session-3": create_mock_session("session-3"),
        }

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = lambda sid: mock_sessions.get(sid)
            result = await manager.persist_dirty()

            assert result == 3
            assert mock_storage.save_session.call_count == 3

    @pytest.mark.asyncio
    async def test_persist_dirty_handles_partial_failure(self) -> None:
        """Test persist_dirty continues on partial failures."""
        mock_storage = AsyncMock()

        manager = HybridSessionManager(storage=mock_storage)
        manager._dirty_sessions = {"session-1", "session-2"}

        # First succeeds, second fails
        mock_storage.save_session = AsyncMock(side_effect=[True, False])

        mock_sessions = {
            "session-1": create_mock_session("session-1"),
            "session-2": create_mock_session("session-2"),
        }

        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = lambda sid: mock_sessions.get(sid)
            result = await manager.persist_dirty()

            # Only 1 succeeded
            assert result == 1


class TestHybridSessionManagerPersistAllWithSessions:
    """Tests for HybridSessionManager persist_all with actual sessions."""

    @pytest.mark.asyncio
    async def test_persist_all_persists_all_sessions(self) -> None:
        """Test persist_all persists all sessions in memory."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=True)

        manager = HybridSessionManager(storage=mock_storage)

        mock_sessions = [
            create_mock_session("session-1"),
            create_mock_session("session-2"),
        ]

        with patch.object(manager._memory, "list_sessions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_sessions
            result = await manager.persist_all()

            assert result == 2
            assert mock_storage.save_session.call_count == 2


class TestHybridSessionManagerRecoverSessionsWithSessions:
    """Tests for HybridSessionManager recover_sessions with actual sessions."""

    @pytest.mark.asyncio
    async def test_recover_loads_sessions_from_storage(self) -> None:
        """Test recover_sessions loads sessions from storage."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["session-1", "session-2"])
        mock_storage.get_session_metadata = AsyncMock(
            return_value={"created_at": datetime.now(UTC).isoformat()}
        )
        mock_sessions = {
            "session-1": create_mock_session("session-1"),
            "session-2": create_mock_session("session-2"),
        }
        mock_storage.load_session = AsyncMock(side_effect=lambda sid: mock_sessions.get(sid))

        manager = HybridSessionManager(storage=mock_storage)

        result = await manager.recover_sessions(max_age_hours=24)

        assert result == 2
        assert "session-1" in manager._memory._sessions
        assert "session-2" in manager._memory._sessions

    @pytest.mark.asyncio
    async def test_recover_skips_old_sessions(self) -> None:
        """Test recover_sessions skips sessions older than max_age."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["old-session"])

        # Session created 48 hours ago
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        mock_storage.get_session_metadata = AsyncMock(return_value={"created_at": old_time})

        manager = HybridSessionManager(storage=mock_storage)

        result = await manager.recover_sessions(max_age_hours=24)

        assert result == 0
        # Should not have loaded the session
        mock_storage.load_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_skips_sessions_already_in_memory(self) -> None:
        """Test recover_sessions skips sessions already in memory."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["existing-session"])
        mock_storage.get_session_metadata = AsyncMock(
            return_value={"created_at": datetime.now(UTC).isoformat()}
        )

        manager = HybridSessionManager(storage=mock_storage)

        # Pre-add session to memory
        with patch.object(manager._memory, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = create_mock_session("existing-session")
            result = await manager.recover_sessions()

            assert result == 0
            mock_storage.load_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_respects_memory_limit(self) -> None:
        """Test recover_sessions respects memory limit."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(
            return_value=["session-1", "session-2", "session-3"]
        )
        mock_storage.get_session_metadata = AsyncMock(
            return_value={"created_at": datetime.now(UTC).isoformat()}
        )
        mock_sessions = {
            "session-1": create_mock_session("session-1"),
            "session-2": create_mock_session("session-2"),
            "session-3": create_mock_session("session-3"),
        }
        mock_storage.load_session = AsyncMock(side_effect=lambda sid: mock_sessions.get(sid))

        # Very small limit
        manager = HybridSessionManager(storage=mock_storage, max_sessions=2)

        result = await manager.recover_sessions()

        # Should only recover up to the limit
        assert result == 2

    @pytest.mark.asyncio
    async def test_recover_handles_session_without_created_at(self) -> None:
        """Test recover handles sessions without created_at metadata."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["no-created-at"])
        mock_storage.get_session_metadata = AsyncMock(return_value={})
        mock_storage.load_session = AsyncMock(return_value=create_mock_session("no-created-at"))

        manager = HybridSessionManager(storage=mock_storage)

        result = await manager.recover_sessions()

        # Should still load the session
        assert result == 1

    @pytest.mark.asyncio
    async def test_recover_handles_errors_gracefully(self) -> None:
        """Test recover_sessions handles errors gracefully."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["error-session", "good-session"])
        mock_storage.get_session_metadata = AsyncMock(side_effect=Exception("Metadata error"))

        manager = HybridSessionManager(storage=mock_storage)

        with patch("reasoning_mcp.storage.hybrid.logger") as mock_logger:
            result = await manager.recover_sessions()

            # Should continue despite errors
            assert result == 0
            mock_logger.error.assert_called()


class TestHybridSessionManagerListSessionsWithDisk:
    """Tests for HybridSessionManager list_sessions with disk option."""

    @pytest.mark.asyncio
    async def test_list_sessions_includes_disk_sessions(self) -> None:
        """Test list_sessions includes disk sessions when requested."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["disk-session-1", "disk-session-2"])
        mock_storage.load_session = AsyncMock(return_value=create_mock_session("disk-session-1"))

        manager = HybridSessionManager(storage=mock_storage)

        # No sessions in memory
        with patch.object(manager._memory, "list_sessions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            result = await manager.list_sessions(include_disk=True, limit=10)

            # Should have loaded from disk
            assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_list_sessions_respects_status_filter(self) -> None:
        """Test list_sessions respects status filter for disk sessions."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["filtered-session"])

        mock_session = create_mock_session("filtered-session")
        mock_session.status = "completed"
        mock_storage.load_session = AsyncMock(return_value=mock_session)

        manager = HybridSessionManager(storage=mock_storage)

        with patch.object(manager._memory, "list_sessions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            result = await manager.list_sessions(status="active", include_disk=True)

            # Should not include completed session when filtering for active
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_sessions_respects_limit(self) -> None:
        """Test list_sessions respects limit."""
        manager = HybridSessionManager()

        mock_sessions = [create_mock_session(f"session-{i}") for i in range(10)]

        with patch.object(manager._memory, "list_sessions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_sessions

            result = await manager.list_sessions(limit=5)

            assert len(result) == 5


class TestHybridSessionManagerCountWithDisk:
    """Tests for HybridSessionManager count with disk option."""

    @pytest.mark.asyncio
    async def test_count_includes_disk_sessions(self) -> None:
        """Test count includes disk sessions when requested."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(
            return_value=["disk-1", "disk-2", "memory-1"]
        )

        manager = HybridSessionManager(storage=mock_storage)

        # Add one session to memory
        manager._memory._sessions["memory-1"] = create_mock_session("memory-1")

        result = await manager.count(include_disk=True)

        # 1 in memory + 2 disk-only
        assert result == 3


class TestHybridSessionManagerCleanupExpired:
    """Tests for HybridSessionManager cleanup_expired method."""

    @pytest.mark.asyncio
    async def test_cleanup_delegates_to_memory(self) -> None:
        """Test cleanup_expired delegates to memory manager."""
        manager = HybridSessionManager()

        with patch.object(
            manager._memory, "cleanup_expired", new_callable=AsyncMock
        ) as mock_cleanup:
            mock_cleanup.return_value = 5
            result = await manager.cleanup_expired(max_age_seconds=3600)

            assert result == 5
            mock_cleanup.assert_called_once_with(3600)

    @pytest.mark.asyncio
    async def test_cleanup_removes_deleted_from_dirty_set(self) -> None:
        """Test cleanup removes deleted sessions from dirty set."""
        manager = HybridSessionManager()
        manager._dirty_sessions = {"session-1", "session-2", "session-3"}

        # Only session-1 remains in memory after cleanup
        manager._memory._sessions = {"session-1": create_mock_session("session-1")}

        with patch.object(manager._memory, "cleanup_expired", new_callable=AsyncMock) as mock_cleanup:
            mock_cleanup.return_value = 2
            await manager.cleanup_expired()

            # Dirty sessions should be cleaned up
            assert manager._dirty_sessions == {"session-1"}


class TestHybridSessionManagerAutoPersistLoop:
    """Tests for HybridSessionManager _auto_persist_loop method."""

    @pytest.mark.asyncio
    async def test_auto_persist_loop_persists_periodically(self) -> None:
        """Test auto persist loop persists dirty sessions."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=True)

        manager = HybridSessionManager(
            storage=mock_storage,
            auto_persist_interval=0,  # Very short for testing
        )

        mock_session = create_mock_session("auto-persist-session")
        manager._memory._sessions["auto-persist-session"] = mock_session
        manager._dirty_sessions.add("auto-persist-session")

        # Start the loop task
        task = asyncio.create_task(manager._auto_persist_loop())

        # Give it time to run once
        await asyncio.sleep(0.1)

        # Shutdown
        manager._shutdown = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have persisted
        assert mock_storage.save_session.call_count >= 1

    @pytest.mark.asyncio
    async def test_auto_persist_loop_handles_errors(self) -> None:
        """Test auto persist loop handles errors gracefully."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(side_effect=Exception("Persist error"))

        manager = HybridSessionManager(
            storage=mock_storage,
            auto_persist_interval=0,
        )
        # Add session to memory so persist actually tries to save
        mock_session = create_mock_session("error-session")
        manager._memory._sessions["error-session"] = mock_session
        manager._dirty_sessions.add("error-session")

        with patch("reasoning_mcp.storage.hybrid.logger") as mock_logger:
            task = asyncio.create_task(manager._auto_persist_loop())

            await asyncio.sleep(0.1)

            manager._shutdown = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have logged error
            mock_logger.error.assert_called()


class TestHybridSessionManagerShutdownWithStorage:
    """Tests for HybridSessionManager shutdown with storage."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_persist_task(self) -> None:
        """Test shutdown cancels auto-persist task."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=[])
        mock_storage.save_session = AsyncMock(return_value=True)
        mock_storage.close = AsyncMock()

        manager = HybridSessionManager(
            storage=mock_storage,
            recovery_on_startup=False,
            auto_persist_interval=60,
        )

        # Start the manager
        await manager.initialize()
        assert manager._persist_task is not None

        # Shutdown
        await manager.shutdown()

        assert manager._shutdown is True
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_persists_all_sessions(self) -> None:
        """Test shutdown persists all sessions."""
        mock_storage = AsyncMock()
        mock_storage.save_session = AsyncMock(return_value=True)
        mock_storage.close = AsyncMock()

        manager = HybridSessionManager(storage=mock_storage)

        mock_sessions = [create_mock_session("final-1"), create_mock_session("final-2")]

        with patch.object(manager._memory, "list_sessions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_sessions
            await manager.shutdown()

            # Should have persisted all sessions
            assert mock_storage.save_session.call_count == 2


class TestHybridSessionManagerInitializeWithRecovery:
    """Tests for HybridSessionManager initialize with recovery enabled."""

    @pytest.mark.asyncio
    async def test_initialize_recovers_sessions(self) -> None:
        """Test initialize recovers sessions when enabled."""
        mock_storage = AsyncMock()
        mock_storage.list_session_ids = AsyncMock(return_value=["recovered-session"])
        mock_storage.get_session_metadata = AsyncMock(
            return_value={"created_at": datetime.now(UTC).isoformat()}
        )
        mock_storage.load_session = AsyncMock(
            return_value=create_mock_session("recovered-session")
        )

        manager = HybridSessionManager(
            storage=mock_storage,
            recovery_on_startup=True,
            auto_persist_interval=0,  # Disable auto persist for test
        )

        await manager.initialize()

        # Should have recovered the session
        assert "recovered-session" in manager._memory._sessions
        manager._shutdown = True
