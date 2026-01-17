"""Serialization utilities for streaming events."""

import json

from reasoning_mcp.streaming.events import (
    BaseStreamEvent,
    CompleteEvent,
    ErrorEvent,
    ProgressEvent,
    StageEvent,
    StreamingEventType,
    ThoughtEvent,
    TokenEvent,
)


class SSESerializer:
    """Serializer for Server-Sent Events format."""

    @staticmethod
    def serialize(event: BaseStreamEvent) -> str:
        """Serialize an event to SSE format.

        Args:
            event: The streaming event to serialize.

        Returns:
            SSE-formatted string with event type and data.
        """
        event_type = event.event_type.value
        data = event.model_dump_json()
        return f"event: {event_type}\ndata: {data}\n\n"

    @staticmethod
    def serialize_batch(events: list[BaseStreamEvent]) -> str:
        """Serialize multiple events to SSE format.

        Args:
            events: List of streaming events to serialize.

        Returns:
            Concatenated SSE-formatted string for all events.
        """
        return "".join(SSESerializer.serialize(event) for event in events)


EVENT_TYPE_MAP: dict[StreamingEventType, type[BaseStreamEvent]] = {
    StreamingEventType.THOUGHT: ThoughtEvent,
    StreamingEventType.PROGRESS: ProgressEvent,
    StreamingEventType.STAGE_START: StageEvent,
    StreamingEventType.STAGE_END: StageEvent,
    StreamingEventType.TOKEN: TokenEvent,
    StreamingEventType.ERROR: ErrorEvent,
    StreamingEventType.COMPLETE: CompleteEvent,
}


class JSONLSerializer:
    """Serializer for JSON Lines format."""

    @staticmethod
    def serialize(event: BaseStreamEvent) -> str:
        """Serialize an event to JSON Lines format.

        Args:
            event: The streaming event to serialize.

        Returns:
            JSON string followed by newline.
        """
        return event.model_dump_json() + "\n"

    @staticmethod
    def deserialize(line: str) -> BaseStreamEvent:
        """Deserialize a JSON line to an event.

        Args:
            line: JSON string representing an event.

        Returns:
            The deserialized streaming event.

        Raises:
            ValueError: If the event type is unknown or data is invalid.
        """
        data = json.loads(line.strip())
        event_type = StreamingEventType(data.get("event_type"))
        event_class = EVENT_TYPE_MAP.get(event_type, BaseStreamEvent)
        return event_class.model_validate(data)
