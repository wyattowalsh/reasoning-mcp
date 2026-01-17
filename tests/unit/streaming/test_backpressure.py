"""Tests for streaming backpressure module."""

import asyncio

import pytest

from reasoning_mcp.streaming.backpressure import (
    BackpressureConfig,
    BackpressureError,
    BackpressureHandler,
    BackpressureStrategy,
)
from reasoning_mcp.streaming.events import ThoughtEvent


def create_event() -> ThoughtEvent:
    """Create a test event."""
    return ThoughtEvent(
        session_id="test",
        thought_number=1,
        content="Test",
        method_name="test",
    )


class TestBackpressureStrategy:
    """Tests for BackpressureStrategy enum."""

    def test_strategy_enum(self):
        """Test enum values."""
        assert BackpressureStrategy.BLOCK.value == "block"
        assert BackpressureStrategy.DROP_OLDEST.value == "drop_oldest"
        assert BackpressureStrategy.DROP_NEWEST.value == "drop_newest"
        assert BackpressureStrategy.ERROR.value == "error"


class TestBackpressureConfig:
    """Tests for BackpressureConfig model."""

    def test_config_model(self):
        """Test config model defaults."""
        config = BackpressureConfig()
        assert config.strategy == BackpressureStrategy.BLOCK
        assert config.queue_size == 1000
        assert config.block_timeout_ms == 5000

    def test_config_custom(self):
        """Test config with custom values."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            queue_size=500,
            block_timeout_ms=10000,
        )
        assert config.strategy == BackpressureStrategy.DROP_OLDEST
        assert config.queue_size == 500


class TestBackpressureHandler:
    """Tests for BackpressureHandler."""

    @pytest.mark.asyncio
    async def test_handle_not_full(self):
        """Test handling when queue is not full."""
        handler = BackpressureHandler()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        config = BackpressureConfig()
        event = create_event()

        result = await handler.handle(queue, event, config)
        assert result is True
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_handle_drop_newest(self):
        """Test DROP_NEWEST strategy."""
        handler = BackpressureHandler()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        config = BackpressureConfig(strategy=BackpressureStrategy.DROP_NEWEST)

        # Fill the queue
        await queue.put(create_event())
        assert queue.full()

        # Try to add another
        result = await handler.handle(queue, create_event(), config)
        assert result is False  # Dropped
        assert queue.qsize() == 1  # Still just one

    @pytest.mark.asyncio
    async def test_handle_drop_oldest(self):
        """Test DROP_OLDEST strategy."""
        handler = BackpressureHandler()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        config = BackpressureConfig(strategy=BackpressureStrategy.DROP_OLDEST)

        # Fill the queue
        old_event = create_event()
        await queue.put(old_event)

        # Add new event (should drop old)
        new_event = ThoughtEvent(
            session_id="test",
            thought_number=2,
            content="New",
            method_name="test",
        )
        result = await handler.handle(queue, new_event, config)
        assert result is True
        assert queue.qsize() == 1
        # Queue should have new event
        retrieved = await queue.get()
        assert retrieved.thought_number == 2

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Test ERROR strategy."""
        handler = BackpressureHandler()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        config = BackpressureConfig(strategy=BackpressureStrategy.ERROR)

        await queue.put(create_event())

        with pytest.raises(BackpressureError):
            await handler.handle(queue, create_event(), config)

    @pytest.mark.asyncio
    async def test_handle_block_timeout(self):
        """Test BLOCK strategy with timeout."""
        handler = BackpressureHandler()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BLOCK,
            block_timeout_ms=100,  # 100ms timeout
        )

        await queue.put(create_event())

        with pytest.raises(asyncio.TimeoutError):
            await handler.handle(queue, create_event(), config)
