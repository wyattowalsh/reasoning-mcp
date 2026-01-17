"""Stream emitter for publishing and subscribing to events."""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from reasoning_mcp.streaming.backpressure import (
    BackpressureConfig,
    BackpressureHandler,
    BackpressureStrategy,
)
from reasoning_mcp.streaming.events import BaseStreamEvent
from reasoning_mcp.streaming.metrics import MetricsCollector


@runtime_checkable
class StreamEmitterProtocol(Protocol):
    """Protocol defining the interface for stream emitters."""

    async def emit(self, event: BaseStreamEvent) -> None:
        """Emit an event to all subscribers."""
        ...

    async def close(self) -> None:
        """Close the emitter and clean up resources."""
        ...

    @property
    def is_closed(self) -> bool:
        """Check if the emitter has been closed."""
        ...


class AsyncStreamEmitter:
    """Async stream emitter with backpressure support.

    Allows multiple subscribers to receive events through async iteration.
    Implements fan-out to all active subscribers.
    """

    def __init__(
        self,
        queue_size: int = 1000,
        backpressure_config: BackpressureConfig | None = None,
    ) -> None:
        """Initialize the async stream emitter.

        Args:
            queue_size: Maximum size of the internal event queue.
            backpressure_config: Configuration for backpressure handling.
        """
        self._queue: asyncio.Queue[BaseStreamEvent | None] = asyncio.Queue(maxsize=queue_size)
        self._closed = False
        self._backpressure_config = backpressure_config or BackpressureConfig(
            strategy=BackpressureStrategy.BLOCK,
            queue_size=queue_size,
        )
        self._backpressure_handler = BackpressureHandler()
        self._subscribers: list[asyncio.Queue[BaseStreamEvent | None]] = []
        self._metrics = MetricsCollector()

    async def emit(self, event: BaseStreamEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event: The streaming event to emit.

        Raises:
            RuntimeError: If emitter is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot emit to a closed emitter")

        import time

        start = time.monotonic()

        # Fan out to all subscriber queues
        for sub_queue in self._subscribers:
            try:
                sub_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Handle backpressure per subscriber
                self._metrics.record_backpressure()
                if self._backpressure_config.strategy == BackpressureStrategy.DROP_NEWEST:
                    self._metrics.record_drop()
                elif self._backpressure_config.strategy == BackpressureStrategy.DROP_OLDEST:
                    try:
                        sub_queue.get_nowait()
                        sub_queue.put_nowait(event)
                    except asyncio.QueueEmpty:
                        sub_queue.put_nowait(event)

        # Also put in main queue for single-consumer pattern
        if not self._queue.full():
            await self._queue.put(event)

        latency_ms = (time.monotonic() - start) * 1000
        self._metrics.record_emit(latency_ms)

    async def close(self) -> None:
        """Close the emitter and signal subscribers to stop."""
        self._closed = True
        # Send sentinel to all subscriber queues
        for sub_queue in self._subscribers:
            try:
                sub_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        # Send sentinel to main queue
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    @property
    def is_closed(self) -> bool:
        """Check if the emitter has been closed."""
        return self._closed

    async def subscribe(self) -> AsyncIterator[BaseStreamEvent]:
        """Subscribe to receive events.

        Yields:
            Streaming events as they are emitted.

        Note:
            Iteration stops when the emitter is closed.
        """
        sub_queue: asyncio.Queue[BaseStreamEvent | None] = asyncio.Queue(
            maxsize=self._backpressure_config.queue_size
        )
        self._subscribers.append(sub_queue)
        try:
            while True:
                event = await sub_queue.get()
                if event is None:  # Sentinel indicating close
                    break
                yield event
        finally:
            self._subscribers.remove(sub_queue)

    async def __aiter__(self) -> AsyncIterator[BaseStreamEvent]:
        """Iterate over events from the main queue."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics
