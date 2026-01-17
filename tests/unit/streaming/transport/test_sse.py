"""Tests for SSE transport module."""

import pytest

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import ThoughtEvent
from reasoning_mcp.streaming.transport.sse import SSETransport


def create_event() -> ThoughtEvent:
    """Create a test event."""
    return ThoughtEvent(
        session_id="test",
        thought_number=1,
        content="Test thought",
        method_name="test",
    )


class TestSSETransport:
    """Tests for SSETransport."""

    def test_format_event(self):
        """Test formatting an event for SSE."""
        transport = SSETransport()
        event = create_event()
        result = transport.format_event(event)

        assert result.startswith("event: thought\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_create_response(self):
        """Test creating a streaming response."""
        transport = SSETransport()
        emitter = AsyncStreamEmitter()

        response = transport.create_response(emitter)

        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("Connection") == "keep-alive"

    @pytest.mark.asyncio
    async def test_event_generator(self):
        """Test the event generator."""
        transport = SSETransport()
        emitter = AsyncStreamEmitter()

        # Emit some events and close
        await emitter.emit(create_event())
        await emitter.close()

        # Collect from generator
        results = []
        async for sse_data in transport._event_generator(emitter):
            results.append(sse_data)

        assert len(results) == 1
        assert "event: thought" in results[0]
