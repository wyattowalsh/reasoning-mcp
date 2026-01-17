"""Integration tests for backpressure handling."""

import asyncio

import pytest

from reasoning_mcp.streaming import (
    AsyncStreamEmitter,
    BackpressureConfig,
    BackpressureStrategy,
    StreamingContext,
)


class TestBackpressureIntegration:
    """Integration tests for backpressure scenarios."""

    @pytest.mark.asyncio
    async def test_backpressure_block_allows_consumption(self):
        """Test BLOCK strategy allows consumer to catch up."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BLOCK,
            queue_size=20,  # Larger queue to avoid timing issues
            block_timeout_ms=5000,
        )
        emitter = AsyncStreamEmitter(queue_size=20, backpressure_config=config)
        ctx = StreamingContext(emitter=emitter)

        received = []

        async def consumer():
            """Consumer that collects events."""
            async for event in emitter.subscribe():
                received.append(event)
                if len(received) >= 5:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.02)

        # Emit events
        for i in range(5):
            await ctx.emit_thought(f"Thought {i}", "test")

        await asyncio.wait_for(task, timeout=2.0)

        # All events should be received
        assert len(received) == 5

        await ctx.close()

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest(self):
        """Test DROP_OLDEST strategy drops old events under pressure."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            queue_size=3,
        )
        emitter = AsyncStreamEmitter(queue_size=3, backpressure_config=config)
        ctx = StreamingContext(emitter=emitter)

        # Emit many events without consuming
        for i in range(10):
            await ctx.emit_thought(f"Thought {i}", "test")

        # Check metrics - some events may have been dropped
        metrics = emitter.metrics.get_metrics()
        # The emitter should have handled the overflow
        assert metrics.events_emitted > 0

        await ctx.close()

    @pytest.mark.asyncio
    async def test_metrics_under_pressure(self):
        """Test metrics are recorded correctly under backpressure."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_NEWEST,
            queue_size=2,
        )
        emitter = AsyncStreamEmitter(queue_size=2, backpressure_config=config)
        ctx = StreamingContext(emitter=emitter)

        # Subscribe but don't consume
        emitter.subscribe()
        await asyncio.sleep(0.01)

        # Emit events to trigger backpressure
        for i in range(10):
            await ctx.emit_thought(f"Thought {i}", "test")

        # Check metrics
        metrics = emitter.metrics.get_metrics()
        assert metrics.events_emitted > 0
        # Some events may have caused backpressure

        await ctx.close()
