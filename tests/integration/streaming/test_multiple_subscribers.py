"""Integration tests for multiple subscriber scenarios."""

import asyncio

import pytest

from reasoning_mcp.streaming import (
    AsyncStreamEmitter,
    StreamingContext,
)


class TestMultipleSubscribers:
    """Tests for fan-out to multiple subscribers."""

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_all_events(self):
        """Test that all subscribers receive all events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)

        received_1 = []
        received_2 = []
        received_3 = []

        async def subscriber(target_list: list):
            async for event in emitter.subscribe():
                target_list.append(event)
                if len(target_list) >= 5:
                    break

        # Start multiple subscribers
        task1 = asyncio.create_task(subscriber(received_1))
        task2 = asyncio.create_task(subscriber(received_2))
        task3 = asyncio.create_task(subscriber(received_3))

        await asyncio.sleep(0.02)  # Give subscribers time to start

        # Emit events
        for i in range(5):
            await ctx.emit_thought(f"Thought {i}", "test")

        # Wait for all subscribers
        await asyncio.wait_for(asyncio.gather(task1, task2, task3), timeout=2.0)

        # All subscribers should have received all events
        assert len(received_1) == 5
        assert len(received_2) == 5
        assert len(received_3) == 5

        # Events should be identical
        for i in range(5):
            assert received_1[i].content == received_2[i].content
            assert received_2[i].content == received_3[i].content

        await ctx.close()

    @pytest.mark.asyncio
    async def test_subscriber_joins_mid_stream(self):
        """Test that late subscribers receive new events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)

        early_received = []
        late_received = []

        async def early_subscriber():
            async for event in emitter.subscribe():
                early_received.append(event)
                if len(early_received) >= 10:
                    break

        async def late_subscriber():
            async for event in emitter.subscribe():
                late_received.append(event)
                if len(late_received) >= 5:
                    break

        # Start early subscriber
        early_task = asyncio.create_task(early_subscriber())
        await asyncio.sleep(0.01)

        # Emit first batch
        for i in range(5):
            await ctx.emit_thought(f"Early {i}", "test")

        # Start late subscriber
        late_task = asyncio.create_task(late_subscriber())
        await asyncio.sleep(0.01)

        # Emit second batch
        for i in range(5):
            await ctx.emit_thought(f"Late {i}", "test")

        await asyncio.wait_for(asyncio.gather(early_task, late_task), timeout=2.0)

        # Early subscriber got all 10
        assert len(early_received) == 10

        # Late subscriber got only the second batch
        assert len(late_received) == 5
        assert all("Late" in e.content for e in late_received)

        await ctx.close()

    @pytest.mark.asyncio
    async def test_subscriber_unsubscribes_cleanly(self):
        """Test that subscribers can unsubscribe without affecting others."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)

        active_received = []

        async def active_subscriber():
            async for event in emitter.subscribe():
                active_received.append(event)
                if len(active_received) >= 10:
                    break

        async def short_subscriber():
            count = 0
            async for _event in emitter.subscribe():
                count += 1
                if count >= 3:
                    break  # Unsubscribe early

        active_task = asyncio.create_task(active_subscriber())
        short_task = asyncio.create_task(short_subscriber())

        await asyncio.sleep(0.01)

        # Emit events
        for i in range(10):
            await ctx.emit_thought(f"Thought {i}", "test")

        await asyncio.wait_for(asyncio.gather(active_task, short_task), timeout=2.0)

        # Active subscriber got all events despite short one leaving
        assert len(active_received) == 10

        await ctx.close()
