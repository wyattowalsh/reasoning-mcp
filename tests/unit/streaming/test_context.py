"""Tests for streaming context module."""

import asyncio

import pytest

from reasoning_mcp.streaming.context import StreamingContext
from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import (
    CompleteEvent,
    ErrorEvent,
    ProgressEvent,
    ThoughtEvent,
)


class TestStreamingContext:
    """Tests for StreamingContext."""

    def test_streaming_context(self):
        """Test context initialization."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter, session_id="test-123")
        assert ctx.session_id == "test-123"
        assert ctx.enabled is True

    def test_context_auto_session_id(self):
        """Test auto-generated session ID."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        assert ctx.session_id is not None
        assert len(ctx.session_id) > 0

    @pytest.mark.asyncio
    async def test_context_helpers_disabled(self):
        """Test that disabled context is a no-op."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter, enabled=False)

        # These should not raise
        await ctx.emit_thought("content", "method")
        await ctx.emit_progress(1, 10)
        await ctx.emit_stage_start("stage")
        await ctx.emit_stage_end("stage")
        await ctx.emit_token("token")
        await ctx.emit_error("E001", "Error")
        await ctx.emit_complete({}, 1000)

    @pytest.mark.asyncio
    async def test_emit_thought(self):
        """Test emitting thought events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_thought("Test thought", "chain_of_thought", confidence=0.9)

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 1
        assert isinstance(received[0], ThoughtEvent)
        assert received[0].content == "Test thought"
        assert received[0].confidence == 0.9

        await ctx.close()

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emitting progress events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_progress(5, 10, message="Halfway")

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 1
        assert isinstance(received[0], ProgressEvent)
        assert received[0].percentage == 50.0

        await ctx.close()

    @pytest.mark.asyncio
    async def test_emit_stages(self):
        """Test emitting stage events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_stage_start("analysis", {"key": "value"})
        await ctx.emit_stage_end("analysis")

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 2
        assert received[0].stage_type == "start"
        assert received[1].stage_type == "end"

        await ctx.close()

    @pytest.mark.asyncio
    async def test_emit_tokens(self):
        """Test emitting token events with cumulative text."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                if len(received) >= 3:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_token("Hello")
        await ctx.emit_token(" ")
        await ctx.emit_token("World")

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 3
        assert received[0].cumulative_text == "Hello"
        assert received[1].cumulative_text == "Hello "
        assert received[2].cumulative_text == "Hello World"

        await ctx.close()

    @pytest.mark.asyncio
    async def test_emit_error(self):
        """Test emitting error events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_error("E001", "Test error", recoverable=True)

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 1
        assert isinstance(received[0], ErrorEvent)
        assert received[0].error_code == "E001"

        await ctx.close()

    @pytest.mark.asyncio
    async def test_emit_complete(self):
        """Test emitting complete events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await ctx.emit_complete({"answer": "42"}, total_duration_ms=5000)

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 1
        assert isinstance(received[0], CompleteEvent)
        assert received[0].final_result == {"answer": "42"}

        await ctx.close()
