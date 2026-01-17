"""Event buffering for streaming operations."""

from collections import deque
from collections.abc import Iterator

from reasoning_mcp.streaming.events import BaseStreamEvent


class EventBuffer:
    """Buffer for collecting and managing streaming events.

    Provides a bounded buffer with automatic overflow handling
    when the maximum size is reached.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the event buffer.

        Args:
            max_size: Maximum number of events to store. When exceeded,
                     oldest events are automatically dropped.
        """
        self.max_size = max_size
        self._events: deque[BaseStreamEvent] = deque(maxlen=max_size)

    def append(self, event: BaseStreamEvent) -> None:
        """Add an event to the buffer.

        If the buffer is at capacity, the oldest event will be
        automatically removed.

        Args:
            event: The streaming event to add.
        """
        self._events.append(event)

    def flush(self) -> list[BaseStreamEvent]:
        """Remove and return all events from the buffer.

        Returns:
            List of all buffered events in order of arrival.
        """
        events = list(self._events)
        self._events.clear()
        return events

    def peek(self, n: int | None = None) -> list[BaseStreamEvent]:
        """View events without removing them.

        Args:
            n: Number of events to peek. If None, returns all.

        Returns:
            List of events (most recent last).
        """
        if n is None:
            return list(self._events)
        return list(self._events)[-n:]

    def __len__(self) -> int:
        """Return the number of buffered events."""
        return len(self._events)

    def __iter__(self) -> Iterator[BaseStreamEvent]:
        """Iterate over buffered events."""
        return iter(self._events)

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._events) >= self.max_size

    @property
    def is_empty(self) -> bool:
        """Check if buffer has no events."""
        return len(self._events) == 0
