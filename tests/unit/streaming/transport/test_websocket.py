"""Tests for WebSocket transport module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import ThoughtEvent
from reasoning_mcp.streaming.transport.websocket import WebSocketTransport


def create_event() -> ThoughtEvent:
    """Create a test event."""
    return ThoughtEvent(
        session_id="test",
        thought_number=1,
        content="Test thought",
        method_name="test",
    )


class TestWebSocketTransport:
    """Tests for WebSocketTransport."""

    @pytest.mark.asyncio
    async def test_websocket_send(self):
        """Test sending an event over WebSocket."""
        transport = WebSocketTransport()
        ws = AsyncMock()
        event = create_event()

        await transport.send(ws, event)

        ws.send_text.assert_called_once()
        call_arg = ws.send_text.call_args[0][0]
        assert "Test thought" in call_arg

    @pytest.mark.asyncio
    async def test_send_batch(self):
        """Test sending multiple events."""
        transport = WebSocketTransport()
        ws = AsyncMock()
        events = [create_event() for _ in range(3)]

        await transport.send_batch(ws, events)

        assert ws.send_text.call_count == 3

    def test_create_handler(self):
        """Test creating a WebSocket handler."""
        transport = WebSocketTransport()
        emitter = AsyncStreamEmitter()

        handler = transport.create_handler(emitter)

        assert callable(handler)

    @pytest.mark.asyncio
    async def test_stream_to_websocket(self):
        """Test streaming to WebSocket."""
        transport = WebSocketTransport()
        ws = AsyncMock()
        emitter = AsyncStreamEmitter()

        # Start streaming task
        stream_task = asyncio.create_task(transport.stream_to_websocket(ws, emitter))

        # Give it time to start
        await asyncio.sleep(0.01)

        # Emit and close
        await emitter.emit(create_event())
        await emitter.close()

        # Wait for stream to finish
        await asyncio.wait_for(stream_task, timeout=1.0)

        # Should have sent at least one event
        assert ws.send_text.call_count >= 1
