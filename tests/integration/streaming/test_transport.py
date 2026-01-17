"""Integration tests for transport layers."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from reasoning_mcp.streaming import (
    AsyncStreamEmitter,
    StreamingContext,
)
from reasoning_mcp.streaming.transport import SSETransport, WebSocketTransport


class TestSSETransportIntegration:
    """Integration tests for SSE transport."""

    @pytest.mark.asyncio
    async def test_sse_transport_streams_events(self):
        """Test SSE transport correctly streams events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        transport = SSETransport()

        sse_output = []

        async def collect_sse():
            async for sse_data in transport._event_generator(emitter):
                sse_output.append(sse_data)
                if len(sse_output) >= 3:
                    break

        task = asyncio.create_task(collect_sse())
        await asyncio.sleep(0.01)

        # Emit events
        await ctx.emit_thought("Thought 1", "test")
        await ctx.emit_thought("Thought 2", "test")
        await ctx.emit_thought("Thought 3", "test")

        await asyncio.wait_for(task, timeout=1.0)

        # Verify SSE format
        assert len(sse_output) == 3
        for sse_data in sse_output:
            assert "event: thought" in sse_data
            assert "data: " in sse_data
            assert sse_data.endswith("\n\n")

        await ctx.close()

    def test_sse_response_headers(self):
        """Test SSE response has correct headers."""
        emitter = AsyncStreamEmitter()
        transport = SSETransport()

        response = transport.create_response(emitter)

        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("Connection") == "keep-alive"


class TestWebSocketTransportIntegration:
    """Integration tests for WebSocket transport."""

    @pytest.mark.asyncio
    async def test_websocket_transport_sends_events(self):
        """Test WebSocket transport sends events correctly."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        transport = WebSocketTransport()

        # Mock WebSocket
        mock_ws = AsyncMock()
        sent_messages = []
        mock_ws.send_text = AsyncMock(side_effect=lambda m: sent_messages.append(m))

        async def stream():
            try:
                await transport.stream_to_websocket(mock_ws, emitter)
            except asyncio.CancelledError:
                pass

        stream_task = asyncio.create_task(stream())
        await asyncio.sleep(0.01)

        # Emit events
        await ctx.emit_thought("WS Thought 1", "test")
        await ctx.emit_thought("WS Thought 2", "test")

        # Give time for events to be sent
        await asyncio.sleep(0.05)

        await ctx.close()
        await asyncio.sleep(0.01)
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

        # Verify messages were sent
        assert len(sent_messages) >= 2
        assert "WS Thought 1" in sent_messages[0]

    @pytest.mark.asyncio
    async def test_websocket_batch_send(self):
        """Test batch sending over WebSocket."""
        transport = WebSocketTransport()

        mock_ws = AsyncMock()
        sent_count = 0

        async def track_sends(msg):
            nonlocal sent_count
            sent_count += 1

        mock_ws.send_text = track_sends

        from reasoning_mcp.streaming.events import ThoughtEvent

        events = [
            ThoughtEvent(
                session_id="test",
                thought_number=i,
                content=f"Batch {i}",
                method_name="test",
            )
            for i in range(5)
        ]

        await transport.send_batch(mock_ws, events)

        assert sent_count == 5
