"""Streaming event models and types for real-time reasoning updates."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class StreamingEventType(Enum):
    """Types of streaming events emitted during reasoning."""

    THOUGHT = "thought"
    PROGRESS = "progress"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    TOKEN = "token"
    ERROR = "error"
    COMPLETE = "complete"


class BaseStreamEvent(BaseModel):
    """Base class for all streaming events."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: StreamingEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str


# Batch 2: Specialized Event Models


class ThoughtEvent(BaseStreamEvent):
    """Event emitted for each reasoning thought."""

    event_type: StreamingEventType = StreamingEventType.THOUGHT
    thought_number: int
    content: str
    method_name: str
    confidence: float | None = None


class ProgressEvent(BaseStreamEvent):
    """Event emitted to indicate progress through reasoning steps."""

    event_type: StreamingEventType = StreamingEventType.PROGRESS
    current_step: int
    total_steps: int
    percentage: float
    message: str

    @field_validator("percentage")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError("Percentage must be between 0 and 100")
        return v


class StageEvent(BaseStreamEvent):
    """Event emitted at the start or end of a reasoning stage."""

    event_type: StreamingEventType = StreamingEventType.STAGE_START
    stage_name: str
    stage_type: Literal["start", "end"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenEvent(BaseStreamEvent):
    """Event emitted for individual tokens during streaming output."""

    event_type: StreamingEventType = StreamingEventType.TOKEN
    token: str
    token_index: int
    cumulative_text: str


class ErrorEvent(BaseStreamEvent):
    """Event emitted when an error occurs during reasoning."""

    event_type: StreamingEventType = StreamingEventType.ERROR
    error_code: str
    error_message: str
    recoverable: bool
    stack_trace: str | None = None


class CompleteEvent(BaseStreamEvent):
    """Event emitted when reasoning completes."""

    event_type: StreamingEventType = StreamingEventType.COMPLETE
    final_result: dict[str, Any]
    total_duration_ms: int
    token_count: int


# SSE Resumability Support (FastMCP SEP-1699)


@dataclass
class StoredEvent:
    """Event stored for SSE resumability (SEP-1699)."""

    event_id: str
    event_type: str
    data: dict[str, Any]
    timestamp: datetime
    sequence: int


@dataclass
class EventStore:
    """Event store for SSE polling and resumability (SEP-1699).

    FastMCP v2.14+ feature that enables clients to resume event streams
    from the last received event ID.

    Attributes:
        store_id: Unique identifier for this event store
        events: List of stored events
        max_events: Maximum events to retain (default 1000)
    """

    store_id: str
    events: list[StoredEvent] = field(default_factory=list)
    max_events: int = 1000
    _sequence: int = 0

    def generate_event_id(self, event_type: str, data: dict[str, Any]) -> str:
        """Generate a unique event ID."""
        self._sequence += 1
        content = f"{self.store_id}:{self._sequence}:{event_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def store_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> StoredEvent:
        """Store an event and return it with generated ID."""
        event_id = self.generate_event_id(event_type, data)

        event = StoredEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            sequence=self._sequence,
        )

        self.events.append(event)

        # Prune old events if over limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

        return event

    def get_events_since(
        self,
        last_event_id: str | None = None,
    ) -> list[StoredEvent]:
        """Get events since a given event ID for resumability.

        Args:
            last_event_id: Last event ID the client received.
                          If None, returns all events.

        Returns:
            List of events after the specified event ID.
        """
        if last_event_id is None:
            return list(self.events)

        # Find the index of the last event
        for i, event in enumerate(self.events):
            if event.event_id == last_event_id:
                return self.events[i + 1 :]

        # Event not found, return all events
        return list(self.events)

    def clear(self) -> int:
        """Clear all stored events."""
        count = len(self.events)
        self.events.clear()
        self._sequence = 0
        return count


def create_event_store(store_id: str | None = None, max_events: int = 1000) -> EventStore:
    """Create a new event store for SSE resumability.

    Args:
        store_id: Optional store ID (generated if not provided)
        max_events: Maximum events to retain

    Returns:
        EventStore instance
    """
    if store_id is None:
        import uuid

        store_id = str(uuid.uuid4())[:8]

    return EventStore(store_id=store_id, max_events=max_events)


__all__ = [
    # Event Types
    "StreamingEventType",
    # Event Models
    "BaseStreamEvent",
    "ThoughtEvent",
    "ProgressEvent",
    "StageEvent",
    "TokenEvent",
    "ErrorEvent",
    "CompleteEvent",
    # SSE Resumability (SEP-1699)
    "StoredEvent",
    "EventStore",
    "create_event_store",
]
