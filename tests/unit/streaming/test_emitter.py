"""Tests for streaming emitter module."""

import asyncio

import pytest

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter, StreamEmitterProtocol
from reasoning_mcp.streaming.events import ThoughtEvent


def create_event(n: int = 1) -> ThoughtEvent:
    """Create a test event."""
    return ThoughtEvent(
        session_id="test",
        thought_number=n,
        content=f"Thought {n}",
        method_name="test",
    )


class TestStreamEmitterProtocol:
    """Tests for StreamEmitterProtocol."""

    def test_emitter_protocol(self):
        """Test that AsyncStreamEmitter implements protocol."""
        emitter = AsyncStreamEmitter()
        assert isinstance(emitter, StreamEmitterProtocol)


class TestAsyncStreamEmitter:
    """Tests for AsyncStreamEmitter."""

    def test_async_emitter_init(self):
        """Test emitter initialization."""
        emitter = AsyncStreamEmitter(queue_size=500)
        assert not emitter.is_closed

    @pytest.mark.asyncio
    async def test_async_emitter_emit(self):
        """Test emitting events."""
        emitter = AsyncStreamEmitter()
        event = create_event()
        await emitter.emit(event)
        # Should not raise

    @pytest.mark.asyncio
    async def test_async_emitter_close(self):
        """Test closing emitter."""
        emitter = AsyncStreamEmitter()
        assert not emitter.is_closed
        await emitter.close()
        assert emitter.is_closed

    @pytest.mark.asyncio
    async def test_emit_after_close_raises(self):
        """Test that emit after close raises error."""
        emitter = AsyncStreamEmitter()
        await emitter.close()
        with pytest.raises(RuntimeError):
            await emitter.emit(create_event())

    @pytest.mark.asyncio
    async def test_async_emitter_subscribe(self):
        """Test subscribing to events."""
        emitter = AsyncStreamEmitter()
        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                if len(received) >= 3:
                    break

        # Start consumer task
        consumer_task = asyncio.create_task(consumer())

        # Give consumer time to start
        await asyncio.sleep(0.01)

        # Emit events
        for i in range(3):
            await emitter.emit(create_event(i))

        # Wait for consumer
        await asyncio.wait_for(consumer_task, timeout=1.0)

        assert len(received) == 3
        await emitter.close()

    @pytest.mark.asyncio
    async def test_emitter_metrics(self):
        """Test that emitter tracks metrics."""
        emitter = AsyncStreamEmitter()

        for i in range(5):
            await emitter.emit(create_event(i))

        metrics = emitter.metrics.get_metrics()
        assert metrics.events_emitted == 5
        assert metrics.avg_latency_ms >= 0

        await emitter.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers receive events."""
        emitter = AsyncStreamEmitter()
        received_1 = []
        received_2 = []

        async def consumer1():
            async for event in emitter.subscribe():
                received_1.append(event)
                if len(received_1) >= 2:
                    break

        async def consumer2():
            async for event in emitter.subscribe():
                received_2.append(event)
                if len(received_2) >= 2:
                    break

        task1 = asyncio.create_task(consumer1())
        task2 = asyncio.create_task(consumer2())

        await asyncio.sleep(0.01)

        for i in range(2):
            await emitter.emit(create_event(i))

        await asyncio.wait_for(asyncio.gather(task1, task2), timeout=1.0)

        assert len(received_1) == 2
        assert len(received_2) == 2

        await emitter.close()
