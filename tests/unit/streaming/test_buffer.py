"""Tests for streaming buffer module."""

from reasoning_mcp.streaming.buffer import EventBuffer
from reasoning_mcp.streaming.events import ThoughtEvent


def create_thought_event(n: int) -> ThoughtEvent:
    """Helper to create test thought events."""
    return ThoughtEvent(
        session_id="test",
        thought_number=n,
        content=f"Thought {n}",
        method_name="test",
    )


class TestEventBuffer:
    """Tests for EventBuffer."""

    def test_init_default_size(self):
        """Test default buffer initialization."""
        buffer = EventBuffer()
        assert buffer.max_size == 1000
        assert len(buffer) == 0

    def test_init_custom_size(self):
        """Test buffer with custom size."""
        buffer = EventBuffer(max_size=50)
        assert buffer.max_size == 50

    def test_append(self):
        """Test appending events to buffer."""
        buffer = EventBuffer(max_size=10)
        event = create_thought_event(1)
        buffer.append(event)
        assert len(buffer) == 1

    def test_flush(self):
        """Test flushing events from buffer."""
        buffer = EventBuffer(max_size=10)
        for i in range(5):
            buffer.append(create_thought_event(i))

        events = buffer.flush()
        assert len(events) == 5
        assert len(buffer) == 0

    def test_overflow_drops_oldest(self):
        """Test that overflow drops oldest events."""
        buffer = EventBuffer(max_size=3)
        for i in range(5):
            buffer.append(create_thought_event(i))

        assert len(buffer) == 3
        events = buffer.flush()
        # Should have events 2, 3, 4 (oldest 0, 1 dropped)
        assert events[0].thought_number == 2
        assert events[2].thought_number == 4

    def test_peek(self):
        """Test peeking at events without removing."""
        buffer = EventBuffer(max_size=10)
        for i in range(5):
            buffer.append(create_thought_event(i))

        peeked = buffer.peek(3)
        assert len(peeked) == 3
        assert len(buffer) == 5  # Not removed

    def test_peek_all(self):
        """Test peeking all events."""
        buffer = EventBuffer(max_size=10)
        for i in range(5):
            buffer.append(create_thought_event(i))

        peeked = buffer.peek()
        assert len(peeked) == 5

    def test_is_full(self):
        """Test is_full property."""
        buffer = EventBuffer(max_size=3)
        assert not buffer.is_full
        for i in range(3):
            buffer.append(create_thought_event(i))
        assert buffer.is_full

    def test_is_empty(self):
        """Test is_empty property."""
        buffer = EventBuffer(max_size=3)
        assert buffer.is_empty
        buffer.append(create_thought_event(1))
        assert not buffer.is_empty

    def test_iteration(self):
        """Test iterating over buffer."""
        buffer = EventBuffer(max_size=10)
        for i in range(5):
            buffer.append(create_thought_event(i))

        collected = list(buffer)
        assert len(collected) == 5
        assert len(buffer) == 5  # Not consumed
