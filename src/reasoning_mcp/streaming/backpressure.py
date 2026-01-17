"""Backpressure handling for streaming operations."""

import asyncio
from enum import Enum

from pydantic import BaseModel, Field

from reasoning_mcp.streaming.events import BaseStreamEvent


class BackpressureStrategy(Enum):
    """Strategy for handling backpressure when queue is full."""

    BLOCK = "block"  # Wait until space is available
    DROP_OLDEST = "drop_oldest"  # Remove oldest event to make room
    DROP_NEWEST = "drop_newest"  # Discard the new event
    ERROR = "error"  # Raise an exception


class BackpressureConfig(BaseModel):
    """Configuration for backpressure handling."""

    strategy: BackpressureStrategy = BackpressureStrategy.BLOCK
    queue_size: int = Field(default=1000, ge=1)
    block_timeout_ms: int = Field(default=5000, ge=0)


class BackpressureError(Exception):
    """Raised when backpressure handling fails."""

    pass


class BackpressureHandler:
    """Handles backpressure for streaming event queues."""

    async def handle(
        self,
        queue: asyncio.Queue[BaseStreamEvent],
        event: BaseStreamEvent,
        config: BackpressureConfig,
    ) -> bool:
        """Handle adding an event to a potentially full queue.

        Args:
            queue: The asyncio queue to add the event to.
            event: The streaming event to add.
            config: Backpressure configuration.

        Returns:
            True if the event was added successfully, False if dropped.

        Raises:
            BackpressureError: If strategy is ERROR and queue is full.
            asyncio.TimeoutError: If strategy is BLOCK and timeout expires.
        """
        if not queue.full():
            await queue.put(event)
            return True

        match config.strategy:
            case BackpressureStrategy.BLOCK:
                timeout = config.block_timeout_ms / 1000.0
                try:
                    await asyncio.wait_for(queue.put(event), timeout=timeout)
                    return True
                except TimeoutError:
                    raise TimeoutError(f"Backpressure timeout after {config.block_timeout_ms}ms")

            case BackpressureStrategy.DROP_OLDEST:
                try:
                    queue.get_nowait()  # Remove oldest
                except asyncio.QueueEmpty:
                    pass
                await queue.put(event)
                return True

            case BackpressureStrategy.DROP_NEWEST:
                return False  # Discard the new event

            case BackpressureStrategy.ERROR:
                raise BackpressureError(f"Queue full (size={queue.qsize()}) and strategy is ERROR")

        return False
