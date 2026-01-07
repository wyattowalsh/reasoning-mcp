"""Unit tests for SessionManager.

This module tests the SessionManager class which provides thread-safe async CRUD
operations and utilities for managing reasoning sessions.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import SessionConfig
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.sessions import SessionManager


class TestSessionManagerInit:
    """Test SessionManager initialization."""

    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        manager = SessionManager()

        assert manager._max_sessions == 100
        assert manager._cleanup_interval == 3600
        assert manager._sessions == {}
        assert isinstance(manager._lock, asyncio.Lock)
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        manager = SessionManager(max_sessions=50, cleanup_interval=1800)

        assert manager._max_sessions == 50
        assert manager._cleanup_interval == 1800
        assert manager._sessions == {}
        assert await manager.count() == 0


class TestSessionManagerCreate:
    """Test SessionManager.create() method."""

    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        """Test creating a session with default configuration."""
        manager = SessionManager()
        session = await manager.create()

        assert session is not None
        assert session.status == SessionStatus.CREATED
        assert session.config is not None
        assert session.config.max_depth == 10  # Default from SessionConfig
        assert session.id is not None
        assert await manager.count() == 1

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self):
        """Test creating a session with custom configuration."""
        manager = SessionManager()
        config = SessionConfig(
            max_depth=20,
            max_thoughts=500,
            timeout_seconds=600.0,
            enable_branching=False,
        )
        session = await manager.create(config=config)

        assert session is not None
        assert session.config.max_depth == 20
        assert session.config.max_thoughts == 500
        assert session.config.timeout_seconds == 600.0
        assert session.config.enable_branching is False
        assert await manager.count() == 1

    @pytest.mark.asyncio
    async def test_create_multiple_sessions(self):
        """Test creating multiple sessions."""
        manager = SessionManager()
        session1 = await manager.create()
        session2 = await manager.create()
        session3 = await manager.create()

        assert session1.id != session2.id != session3.id
        assert await manager.count() == 3

    @pytest.mark.asyncio
    async def test_create_respects_max_sessions_limit(self):
        """Test that create respects max_sessions limit."""
        manager = SessionManager(max_sessions=3)

        # Create up to the limit
        await manager.create()
        await manager.create()
        await manager.create()
        assert await manager.count() == 3

        # Try to exceed the limit
        with pytest.raises(RuntimeError, match="Maximum session limit reached"):
            await manager.create()

        assert await manager.count() == 3

    @pytest.mark.asyncio
    async def test_create_after_delete(self):
        """Test that create works after deleting sessions."""
        manager = SessionManager(max_sessions=2)

        session1 = await manager.create()
        await manager.create()
        assert await manager.count() == 2

        # Should fail
        with pytest.raises(RuntimeError):
            await manager.create()

        # Delete one session
        await manager.delete(session1.id)
        assert await manager.count() == 1

        # Should now succeed
        session3 = await manager.create()
        assert session3 is not None
        assert await manager.count() == 2


class TestSessionManagerGet:
    """Test SessionManager.get() method."""

    @pytest.mark.asyncio
    async def test_get_existing_session(self):
        """Test getting an existing session by ID."""
        manager = SessionManager()
        session = await manager.create()

        retrieved = await manager.get(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.status == session.status

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        """Test getting a non-existent session returns None."""
        manager = SessionManager()
        await manager.create()

        retrieved = await manager.get("non-existent-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_from_empty_manager(self):
        """Test getting from an empty manager returns None."""
        manager = SessionManager()

        retrieved = await manager.get("any-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_retrieves_correct_session(self):
        """Test that get retrieves the correct session among multiple."""
        manager = SessionManager()
        session1 = await manager.create()
        session2 = await manager.create()
        session3 = await manager.create()

        retrieved = await manager.get(session2.id)

        assert retrieved is not None
        assert retrieved.id == session2.id
        assert retrieved.id != session1.id
        assert retrieved.id != session3.id


class TestSessionManagerUpdate:
    """Test SessionManager.update() method."""

    @pytest.mark.asyncio
    async def test_update_existing_session(self):
        """Test updating an existing session."""
        manager = SessionManager()
        session = await manager.create()

        # Modify the session
        session.start()
        updated = await manager.update(session.id, session)

        assert updated is True

        # Verify the update
        retrieved = await manager.get(session.id)
        assert retrieved is not None
        assert retrieved.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_nonexistent_session(self):
        """Test updating a non-existent session returns False."""
        manager = SessionManager()
        session = await manager.create()

        updated = await manager.update("non-existent-id", session)

        assert updated is False

    @pytest.mark.asyncio
    async def test_update_multiple_times(self):
        """Test updating a session multiple times."""
        manager = SessionManager()
        session = await manager.create()

        # First update
        session.start()
        await manager.update(session.id, session)
        retrieved = await manager.get(session.id)
        assert retrieved.status == SessionStatus.ACTIVE

        # Second update
        session.pause()
        await manager.update(session.id, session)
        retrieved = await manager.get(session.id)
        assert retrieved.status == SessionStatus.PAUSED

        # Third update
        session.resume()
        await manager.update(session.id, session)
        retrieved = await manager.get(session.id)
        assert retrieved.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_with_thoughts(self):
        """Test updating a session after adding thoughts."""
        manager = SessionManager()
        session = await manager.create()
        session.start()

        # Add a thought
        thought = ThoughtNode(
            id="test-thought",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test thought",
            confidence=0.8,
        )
        session.add_thought(thought)

        # Update the session
        updated = await manager.update(session.id, session)
        assert updated is True

        # Verify
        retrieved = await manager.get(session.id)
        assert retrieved.thought_count == 1
        assert retrieved.metrics.total_thoughts == 1


class TestSessionManagerDelete:
    """Test SessionManager.delete() method."""

    @pytest.mark.asyncio
    async def test_delete_existing_session(self):
        """Test deleting an existing session."""
        manager = SessionManager()
        session = await manager.create()
        assert await manager.count() == 1

        deleted = await manager.delete(session.id)

        assert deleted is True
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self):
        """Test deleting a non-existent session returns False."""
        manager = SessionManager()
        await manager.create()

        deleted = await manager.delete("non-existent-id")

        assert deleted is False
        assert await manager.count() == 1

    @pytest.mark.asyncio
    async def test_delete_from_empty_manager(self):
        """Test deleting from an empty manager returns False."""
        manager = SessionManager()

        deleted = await manager.delete("any-id")

        assert deleted is False
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_delete_correct_session(self):
        """Test that delete removes the correct session among multiple."""
        manager = SessionManager()
        session1 = await manager.create()
        session2 = await manager.create()
        session3 = await manager.create()
        assert await manager.count() == 3

        deleted = await manager.delete(session2.id)

        assert deleted is True
        assert await manager.count() == 2
        assert await manager.get(session1.id) is not None
        assert await manager.get(session2.id) is None
        assert await manager.get(session3.id) is not None

    @pytest.mark.asyncio
    async def test_delete_same_session_twice(self):
        """Test that deleting the same session twice returns False the second time."""
        manager = SessionManager()
        session = await manager.create()

        deleted1 = await manager.delete(session.id)
        deleted2 = await manager.delete(session.id)

        assert deleted1 is True
        assert deleted2 is False
        assert await manager.count() == 0


class TestSessionManagerListSessions:
    """Test SessionManager.list_sessions() method."""

    @pytest.mark.asyncio
    async def test_list_all_sessions(self):
        """Test listing all sessions."""
        manager = SessionManager()
        session1 = await manager.create()
        session2 = await manager.create()
        session3 = await manager.create()

        sessions = await manager.list_sessions()

        assert len(sessions) == 3
        session_ids = {s.id for s in sessions}
        assert session1.id in session_ids
        assert session2.id in session_ids
        assert session3.id in session_ids

    @pytest.mark.asyncio
    async def test_list_empty_manager(self):
        """Test listing sessions from an empty manager."""
        manager = SessionManager()

        sessions = await manager.list_sessions()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self):
        """Test filtering sessions by status."""
        manager = SessionManager()

        # Create sessions with different statuses
        session1 = await manager.create()
        session2 = await manager.create()
        session2.start()
        await manager.update(session2.id, session2)

        session3 = await manager.create()
        session3.start()
        session3.complete()
        await manager.update(session3.id, session3)

        # Test filtering
        created = await manager.list_sessions(status=SessionStatus.CREATED)
        assert len(created) == 1
        assert created[0].id == session1.id

        active = await manager.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].id == session2.id

        completed = await manager.list_sessions(status=SessionStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == session3.id

    @pytest.mark.asyncio
    async def test_list_with_limit(self):
        """Test limiting the number of sessions returned."""
        manager = SessionManager()

        # Create 5 sessions
        for _ in range(5):
            await manager.create()

        # Get only 3
        sessions = await manager.list_sessions(limit=3)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sorted_by_creation_time(self):
        """Test that sessions are sorted by creation time descending."""
        manager = SessionManager()

        # Create sessions with small delays to ensure different timestamps
        session1 = await manager.create()
        await asyncio.sleep(0.01)
        session2 = await manager.create()
        await asyncio.sleep(0.01)
        session3 = await manager.create()

        sessions = await manager.list_sessions()

        # Most recent first
        assert sessions[0].id == session3.id
        assert sessions[1].id == session2.id
        assert sessions[2].id == session1.id

    @pytest.mark.asyncio
    async def test_list_status_filter_and_limit(self):
        """Test combining status filter and limit."""
        manager = SessionManager()

        # Create multiple active sessions
        for _ in range(5):
            session = await manager.create()
            session.start()
            await manager.update(session.id, session)

        # Create one created session
        await manager.create()

        # Get only 3 active sessions
        active = await manager.list_sessions(status=SessionStatus.ACTIVE, limit=3)

        assert len(active) == 3
        for session in active:
            assert session.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_list_with_zero_limit(self):
        """Test that limit=0 returns empty list."""
        manager = SessionManager()
        await manager.create()
        await manager.create()

        sessions = await manager.list_sessions(limit=0)

        assert sessions == []


class TestSessionManagerCleanupExpired:
    """Test SessionManager.cleanup_expired() method."""

    @pytest.mark.asyncio
    async def test_cleanup_no_expired_sessions(self):
        """Test cleanup when no sessions are expired."""
        manager = SessionManager()
        await manager.create()
        await manager.create()

        # No sessions older than 24 hours
        removed = await manager.cleanup_expired(max_age_seconds=86400)

        assert removed == 0
        assert await manager.count() == 2

    @pytest.mark.asyncio
    async def test_cleanup_all_expired_sessions(self):
        """Test cleanup removes all expired sessions."""
        manager = SessionManager()
        await manager.create()
        await manager.create()
        await manager.create()

        # Remove all sessions older than 0 seconds (all of them)
        removed = await manager.cleanup_expired(max_age_seconds=0)

        assert removed == 3
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_some_expired_sessions(self):
        """Test cleanup removes only expired sessions."""
        manager = SessionManager()

        # Create old sessions by manually setting created_at
        old_session1 = await manager.create()
        old_session1.created_at = datetime.now() - timedelta(hours=25)
        await manager.update(old_session1.id, old_session1)

        old_session2 = await manager.create()
        old_session2.created_at = datetime.now() - timedelta(hours=26)
        await manager.update(old_session2.id, old_session2)

        # Create recent session
        recent_session = await manager.create()

        # Cleanup sessions older than 24 hours
        removed = await manager.cleanup_expired(max_age_seconds=86400)

        assert removed == 2
        assert await manager.count() == 1

        # Verify the recent session remains
        remaining = await manager.get(recent_session.id)
        assert remaining is not None

    @pytest.mark.asyncio
    async def test_cleanup_empty_manager(self):
        """Test cleanup on empty manager."""
        manager = SessionManager()

        removed = await manager.cleanup_expired(max_age_seconds=3600)

        assert removed == 0
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_custom_age_threshold(self):
        """Test cleanup with custom age thresholds."""
        manager = SessionManager()

        # Create session
        session = await manager.create()
        session.created_at = datetime.now() - timedelta(minutes=31)
        await manager.update(session.id, session)

        # Should not be removed (30 minutes = 1800 seconds)
        removed = await manager.cleanup_expired(max_age_seconds=1800)
        assert removed == 1
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_returns_correct_count(self):
        """Test that cleanup returns the correct number of removed sessions."""
        manager = SessionManager()

        # Create sessions with different ages
        for i in range(5):
            session = await manager.create()
            session.created_at = datetime.now() - timedelta(hours=25 + i)
            await manager.update(session.id, session)

        # Create one recent session
        await manager.create()

        removed = await manager.cleanup_expired(max_age_seconds=86400)

        assert removed == 5
        assert await manager.count() == 1


class TestSessionManagerCount:
    """Test SessionManager.count() method."""

    @pytest.mark.asyncio
    async def test_count_empty_manager(self):
        """Test count on empty manager."""
        manager = SessionManager()

        count = await manager.count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_after_create(self):
        """Test count increases after creating sessions."""
        manager = SessionManager()

        assert await manager.count() == 0

        await manager.create()
        assert await manager.count() == 1

        await manager.create()
        assert await manager.count() == 2

        await manager.create()
        assert await manager.count() == 3

    @pytest.mark.asyncio
    async def test_count_after_delete(self):
        """Test count decreases after deleting sessions."""
        manager = SessionManager()

        session1 = await manager.create()
        session2 = await manager.create()
        session3 = await manager.create()
        assert await manager.count() == 3

        await manager.delete(session1.id)
        assert await manager.count() == 2

        await manager.delete(session2.id)
        assert await manager.count() == 1

        await manager.delete(session3.id)
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_count_after_cleanup(self):
        """Test count after cleanup expired sessions."""
        manager = SessionManager()

        # Create sessions
        for _ in range(5):
            await manager.create()

        assert await manager.count() == 5

        # Cleanup all
        await manager.cleanup_expired(max_age_seconds=0)

        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_count_after_clear(self):
        """Test count after clearing all sessions."""
        manager = SessionManager()

        for _ in range(10):
            await manager.create()

        assert await manager.count() == 10

        await manager.clear()

        assert await manager.count() == 0


class TestSessionManagerClear:
    """Test SessionManager.clear() method."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_sessions(self):
        """Test that clear removes all sessions."""
        manager = SessionManager()

        await manager.create()
        await manager.create()
        await manager.create()
        assert await manager.count() == 3

        await manager.clear()

        assert await manager.count() == 0
        sessions = await manager.list_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_clear_empty_manager(self):
        """Test clear on empty manager."""
        manager = SessionManager()

        await manager.clear()

        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_clear_then_create(self):
        """Test that sessions can be created after clear."""
        manager = SessionManager()

        # Create and clear
        await manager.create()
        await manager.create()
        await manager.clear()
        assert await manager.count() == 0

        # Create new sessions
        session1 = await manager.create()
        session2 = await manager.create()

        assert await manager.count() == 2
        assert await manager.get(session1.id) is not None
        assert await manager.get(session2.id) is not None

    @pytest.mark.asyncio
    async def test_clear_preserves_manager_state(self):
        """Test that clear only removes sessions, not manager configuration."""
        manager = SessionManager(max_sessions=50, cleanup_interval=1800)

        await manager.create()
        await manager.clear()

        # Configuration should remain
        assert manager._max_sessions == 50
        assert manager._cleanup_interval == 1800

        # Should still be able to create up to max_sessions
        for _ in range(50):
            await manager.create()
        assert await manager.count() == 50


class TestSessionManagerConcurrency:
    """Test SessionManager thread-safety with concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_create(self):
        """Test concurrent session creation."""
        manager = SessionManager()

        # Create 10 sessions concurrently
        tasks = [manager.create() for _ in range(10)]
        sessions = await asyncio.gather(*tasks)

        # All should succeed
        assert len(sessions) == 10
        assert await manager.count() == 10

        # All should have unique IDs
        session_ids = {s.id for s in sessions}
        assert len(session_ids) == 10

    @pytest.mark.asyncio
    async def test_concurrent_delete(self):
        """Test concurrent session deletion."""
        manager = SessionManager()

        # Create sessions
        sessions = [await manager.create() for _ in range(5)]
        assert await manager.count() == 5

        # Delete concurrently
        tasks = [manager.delete(s.id) for s in sessions]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations."""
        manager = SessionManager()

        # Create some initial sessions
        initial_sessions = [await manager.create() for _ in range(3)]

        # Mix of operations
        async def create_op():
            return await manager.create()

        async def get_op():
            return await manager.get(initial_sessions[0].id)

        async def delete_op():
            return await manager.delete(initial_sessions[1].id)

        async def list_op():
            return await manager.list_sessions()

        async def count_op():
            return await manager.count()

        # Run concurrently
        results = await asyncio.gather(
            create_op(),
            create_op(),
            get_op(),
            delete_op(),
            list_op(),
            count_op(),
        )

        # Verify results make sense
        assert results[0] is not None  # create
        assert results[1] is not None  # create
        assert results[2] is not None  # get
        assert results[3] is True  # delete
        assert isinstance(results[4], list)  # list
        assert isinstance(results[5], int)  # count


class TestSessionManagerEdgeCases:
    """Test SessionManager edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_get_with_empty_string_id(self):
        """Test getting with empty string ID."""
        manager = SessionManager()
        await manager.create()

        result = await manager.get("")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_with_empty_string_id(self):
        """Test deleting with empty string ID."""
        manager = SessionManager()
        await manager.create()

        result = await manager.delete("")

        assert result is False
        assert await manager.count() == 1

    @pytest.mark.asyncio
    async def test_update_with_different_session_id(self):
        """Test updating with mismatched session IDs."""
        manager = SessionManager()
        session1 = await manager.create()
        session2 = await manager.create()

        # Try to update session1's slot with session2's data
        updated = await manager.update(session1.id, session2)

        assert updated is True

        # session1's ID now points to session2's data
        retrieved = await manager.get(session1.id)
        assert retrieved.id == session2.id

    @pytest.mark.asyncio
    async def test_list_with_large_limit(self):
        """Test listing with limit larger than session count."""
        manager = SessionManager()
        await manager.create()
        await manager.create()

        sessions = await manager.list_sessions(limit=100)

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_cleanup_with_negative_age(self):
        """Test cleanup with negative max_age_seconds removes all sessions."""
        manager = SessionManager()
        await manager.create()
        await manager.create()

        # Negative max_age creates a future cutoff time
        # All sessions are "older" than a future time, so all get removed
        removed = await manager.cleanup_expired(max_age_seconds=-3600)

        assert removed == 2
        assert await manager.count() == 0

    @pytest.mark.asyncio
    async def test_max_sessions_boundary(self):
        """Test behavior at max_sessions boundary."""
        manager = SessionManager(max_sessions=1)

        # First should succeed
        session1 = await manager.create()
        assert session1 is not None

        # Second should fail
        with pytest.raises(RuntimeError):
            await manager.create()

        # After delete, should succeed again
        await manager.delete(session1.id)
        session2 = await manager.create()
        assert session2 is not None
