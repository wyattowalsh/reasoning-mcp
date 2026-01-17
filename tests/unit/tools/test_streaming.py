"""Tests for streaming tools module."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.streaming.backpressure import BackpressureStrategy
from reasoning_mcp.tools.streaming import (
    StreamingSessionResult,
    StreamingToolInput,
    get_streaming_context,
    list_active_sessions,
    start_streaming_session_impl,
    stop_streaming_session_impl,
)


class TestStreamingToolInput:
    """Tests for StreamingToolInput model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        input_model = StreamingToolInput()

        assert input_model.enable_streaming is True
        assert input_model.buffer_size == 100
        assert input_model.backpressure == BackpressureStrategy.BLOCK

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        input_model = StreamingToolInput(
            enable_streaming=False,
            buffer_size=50,
            backpressure=BackpressureStrategy.DROP_OLDEST,
        )

        assert input_model.enable_streaming is False
        assert input_model.buffer_size == 50
        assert input_model.backpressure == BackpressureStrategy.DROP_OLDEST

    def test_buffer_size_validation(self) -> None:
        """Test buffer size must be at least 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StreamingToolInput(buffer_size=0)

        with pytest.raises(ValidationError):
            StreamingToolInput(buffer_size=-1)

    def test_valid_buffer_size(self) -> None:
        """Test various valid buffer sizes."""
        for size in [1, 10, 100, 1000]:
            input_model = StreamingToolInput(buffer_size=size)
            assert input_model.buffer_size == size


class TestStreamingSessionResult:
    """Tests for StreamingSessionResult model."""

    def test_creation(self) -> None:
        """Test result model can be created."""
        result = StreamingSessionResult(
            session_id="test-123",
            status="active",
            message="Session started",
        )

        assert result.session_id == "test-123"
        assert result.status == "active"
        assert result.message == "Session started"

    def test_different_statuses(self) -> None:
        """Test various status values."""
        statuses = ["active", "closed", "not_found", "error"]

        for status in statuses:
            result = StreamingSessionResult(
                session_id="test",
                status=status,
                message="Test message",
            )
            assert result.status == status


class TestStartStreamingSessionImpl:
    """Tests for start_streaming_session_impl function."""

    @pytest.mark.asyncio
    async def test_creates_session(self) -> None:
        """Test creates a new streaming session."""
        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()
                    mock_context.return_value = MagicMock()

                    input_model = StreamingToolInput()
                    result = await start_streaming_session_impl(input_model)

                    assert result.status == "active"
                    assert result.message == "Streaming session started"
                    assert result.session_id is not None

    @pytest.mark.asyncio
    async def test_session_id_is_uuid(self) -> None:
        """Test session ID is a valid UUID format."""
        import uuid

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()
                    mock_context.return_value = MagicMock()

                    input_model = StreamingToolInput()
                    result = await start_streaming_session_impl(input_model)

                    # Should be a valid UUID
                    uuid.UUID(result.session_id)

    @pytest.mark.asyncio
    async def test_uses_buffer_size(self) -> None:
        """Test uses provided buffer size."""
        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()
                    mock_context.return_value = MagicMock()

                    input_model = StreamingToolInput(buffer_size=50)
                    await start_streaming_session_impl(input_model)

                    mock_emitter.assert_called_once_with(queue_size=50)

    @pytest.mark.asyncio
    async def test_stores_session_in_active_sessions(self) -> None:
        """Test session is stored in active sessions dict."""
        active_sessions: dict = {}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()
                    mock_ctx_instance = MagicMock()
                    mock_context.return_value = mock_ctx_instance

                    input_model = StreamingToolInput()
                    result = await start_streaming_session_impl(input_model)

                    assert result.session_id in active_sessions
                    assert active_sessions[result.session_id] is mock_ctx_instance


class TestStopStreamingSessionImpl:
    """Tests for stop_streaming_session_impl function."""

    @pytest.mark.asyncio
    async def test_closes_existing_session(self) -> None:
        """Test closes an existing session."""
        mock_ctx = MagicMock()
        mock_ctx.close = AsyncMock()
        active_sessions = {"test-session": mock_ctx}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            result = await stop_streaming_session_impl("test-session")

            assert result.status == "closed"
            assert result.message == "Session closed"
            assert result.session_id == "test-session"
            mock_ctx.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_removes_session_from_active_sessions(self) -> None:
        """Test session is removed from active sessions."""
        mock_ctx = MagicMock()
        mock_ctx.close = AsyncMock()
        active_sessions = {"test-session": mock_ctx}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            await stop_streaming_session_impl("test-session")

            assert "test-session" not in active_sessions

    @pytest.mark.asyncio
    async def test_handles_nonexistent_session(self) -> None:
        """Test handles attempt to stop nonexistent session."""
        active_sessions: dict = {}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            result = await stop_streaming_session_impl("nonexistent")

            assert result.status == "not_found"
            assert result.message == "Session not found"
            assert result.session_id == "nonexistent"


class TestGetStreamingContext:
    """Tests for get_streaming_context function."""

    def test_returns_context_for_existing_session(self) -> None:
        """Test returns context for existing session."""
        mock_ctx = MagicMock()
        active_sessions = {"test-session": mock_ctx}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            result = get_streaming_context("test-session")

            assert result is mock_ctx

    def test_returns_none_for_nonexistent_session(self) -> None:
        """Test returns None for nonexistent session."""
        active_sessions: dict = {}

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            result = get_streaming_context("nonexistent")

            assert result is None


class TestListActiveSessions:
    """Tests for list_active_sessions function."""

    def test_returns_empty_list_when_no_sessions(self) -> None:
        """Test returns empty list when no active sessions."""
        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            result = list_active_sessions()

            assert result == []

    def test_returns_session_ids(self) -> None:
        """Test returns list of session IDs."""
        active_sessions = {
            "session-1": MagicMock(),
            "session-2": MagicMock(),
            "session-3": MagicMock(),
        }

        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", active_sessions
        ):
            result = list_active_sessions()

            assert len(result) == 3
            assert "session-1" in result
            assert "session-2" in result
            assert "session-3" in result


class TestStreamingIntegration:
    """Integration tests for streaming tools."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self) -> None:
        """Test complete session lifecycle: start, get, stop."""
        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()
                    mock_ctx_instance = MagicMock()
                    mock_ctx_instance.close = AsyncMock()
                    mock_context.return_value = mock_ctx_instance

                    # Start session
                    input_model = StreamingToolInput()
                    start_result = await start_streaming_session_impl(input_model)

                    assert start_result.status == "active"
                    session_id = start_result.session_id

                    # Get context
                    ctx = get_streaming_context(session_id)
                    assert ctx is mock_ctx_instance

                    # Session should be in active list
                    active_list = list_active_sessions()
                    assert session_id in active_list

                    # Stop session
                    stop_result = await stop_streaming_session_impl(session_id)
                    assert stop_result.status == "closed"

                    # Session should no longer be in active list
                    active_list = list_active_sessions()
                    assert session_id not in active_list

                    # Getting context should return None
                    ctx = get_streaming_context(session_id)
                    assert ctx is None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self) -> None:
        """Test multiple concurrent sessions."""
        with patch(
            "reasoning_mcp.tools.streaming._active_sessions", {}
        ):
            with patch(
                "reasoning_mcp.tools.streaming.AsyncStreamEmitter"
            ) as mock_emitter:
                with patch(
                    "reasoning_mcp.tools.streaming.StreamingContext"
                ) as mock_context:
                    mock_emitter.return_value = MagicMock()

                    def create_mock_ctx(**kwargs):
                        ctx = MagicMock()
                        ctx.close = AsyncMock()
                        return ctx

                    mock_context.side_effect = create_mock_ctx

                    # Start multiple sessions
                    input_model = StreamingToolInput()
                    result1 = await start_streaming_session_impl(input_model)
                    result2 = await start_streaming_session_impl(input_model)
                    result3 = await start_streaming_session_impl(input_model)

                    # All should be active
                    active_list = list_active_sessions()
                    assert len(active_list) == 3
                    assert result1.session_id in active_list
                    assert result2.session_id in active_list
                    assert result3.session_id in active_list

                    # All session IDs should be unique
                    session_ids = [
                        result1.session_id,
                        result2.session_id,
                        result3.session_id,
                    ]
                    assert len(set(session_ids)) == 3
