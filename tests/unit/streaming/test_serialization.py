"""Tests for streaming serialization module."""

import json

from reasoning_mcp.streaming.events import (
    ProgressEvent,
    ThoughtEvent,
)
from reasoning_mcp.streaming.serialization import (
    JSONLSerializer,
    SSESerializer,
)


class TestSSESerializer:
    """Tests for SSESerializer."""

    def test_serialize_single_event(self):
        """Test serializing a single event to SSE format."""
        event = ThoughtEvent(
            session_id="test-session",
            thought_number=1,
            content="Test thought",
            method_name="test_method",
        )
        result = SSESerializer.serialize(event)
        assert result.startswith("event: thought\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_serialize_batch(self):
        """Test serializing multiple events."""
        events = [
            ThoughtEvent(
                session_id="test-session",
                thought_number=i,
                content=f"Thought {i}",
                method_name="test",
            )
            for i in range(3)
        ]
        result = SSESerializer.serialize_batch(events)
        assert result.count("event: thought") == 3
        assert result.count("data: ") == 3


class TestJSONLSerializer:
    """Tests for JSONLSerializer."""

    def test_serialize(self):
        """Test serializing to JSON Lines format."""
        event = ProgressEvent(
            session_id="test-session",
            current_step=1,
            total_steps=10,
            percentage=10.0,
            message="Progress",
        )
        result = JSONLSerializer.serialize(event)
        assert result.endswith("\n")
        data = json.loads(result)
        assert data["current_step"] == 1
        assert data["total_steps"] == 10

    def test_deserialize(self):
        """Test deserializing from JSON Lines format."""
        event = ThoughtEvent(
            session_id="test-session",
            thought_number=1,
            content="Test",
            method_name="test",
        )
        serialized = JSONLSerializer.serialize(event)
        deserialized = JSONLSerializer.deserialize(serialized)
        assert isinstance(deserialized, ThoughtEvent)
        assert deserialized.thought_number == 1
        assert deserialized.content == "Test"

    def test_roundtrip(self):
        """Test serialize/deserialize roundtrip."""
        original = ProgressEvent(
            session_id="test-session",
            current_step=5,
            total_steps=10,
            percentage=50.0,
            message="Halfway",
        )
        serialized = JSONLSerializer.serialize(original)
        deserialized = JSONLSerializer.deserialize(serialized)
        assert deserialized.current_step == original.current_step
        assert deserialized.percentage == original.percentage
