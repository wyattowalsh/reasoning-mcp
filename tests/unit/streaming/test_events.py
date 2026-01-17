"""Tests for streaming events module."""

from datetime import datetime

import pytest

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


def test_event_type_enum():
    """Test StreamingEventType enum values."""
    assert StreamingEventType.THOUGHT.value == "thought"
    assert StreamingEventType.PROGRESS.value == "progress"
    assert StreamingEventType.STAGE_START.value == "stage_start"
    assert StreamingEventType.STAGE_END.value == "stage_end"
    assert StreamingEventType.TOKEN.value == "token"
    assert StreamingEventType.ERROR.value == "error"
    assert StreamingEventType.COMPLETE.value == "complete"
    assert len(StreamingEventType) == 7


def test_base_event_model():
    """Test BaseStreamEvent model."""
    event = BaseStreamEvent(
        event_type=StreamingEventType.THOUGHT,
        session_id="test-session",
    )
    assert event.event_id is not None
    assert event.event_type == StreamingEventType.THOUGHT
    assert event.timestamp is not None
    assert isinstance(event.timestamp, datetime)
    assert event.session_id == "test-session"


def test_thought_event():
    """Test ThoughtEvent model."""
    event = ThoughtEvent(
        session_id="test-session",
        thought_number=1,
        content="This is a thought",
        method_name="chain_of_thought",
        confidence=0.85,
    )
    assert event.event_type == StreamingEventType.THOUGHT
    assert event.thought_number == 1
    assert event.content == "This is a thought"
    assert event.method_name == "chain_of_thought"
    assert event.confidence == 0.85


def test_progress_event():
    """Test ProgressEvent model."""
    event = ProgressEvent(
        session_id="test-session",
        current_step=5,
        total_steps=10,
        percentage=50.0,
        message="Halfway there",
    )
    assert event.event_type == StreamingEventType.PROGRESS
    assert event.current_step == 5
    assert event.total_steps == 10
    assert event.percentage == 50.0
    assert event.message == "Halfway there"


def test_progress_event_percentage_validation():
    """Test ProgressEvent percentage validation."""
    with pytest.raises(ValueError):
        ProgressEvent(
            session_id="test-session",
            current_step=1,
            total_steps=10,
            percentage=150.0,  # Invalid
            message="Invalid",
        )


def test_stage_event():
    """Test StageEvent model."""
    event = StageEvent(
        session_id="test-session",
        stage_name="analysis",
        stage_type="start",
        metadata={"key": "value"},
    )
    assert event.stage_name == "analysis"
    assert event.stage_type == "start"
    assert event.metadata == {"key": "value"}


def test_token_event():
    """Test TokenEvent model."""
    event = TokenEvent(
        session_id="test-session",
        token="hello",
        token_index=1,
        cumulative_text="hello",
    )
    assert event.event_type == StreamingEventType.TOKEN
    assert event.token == "hello"
    assert event.token_index == 1
    assert event.cumulative_text == "hello"


def test_error_event():
    """Test ErrorEvent model."""
    event = ErrorEvent(
        session_id="test-session",
        error_code="E001",
        error_message="Something went wrong",
        recoverable=True,
        stack_trace="Traceback...",
    )
    assert event.event_type == StreamingEventType.ERROR
    assert event.error_code == "E001"
    assert event.error_message == "Something went wrong"
    assert event.recoverable is True
    assert event.stack_trace == "Traceback..."


def test_complete_event():
    """Test CompleteEvent model."""
    event = CompleteEvent(
        session_id="test-session",
        final_result={"answer": "42"},
        total_duration_ms=1500,
        token_count=100,
    )
    assert event.event_type == StreamingEventType.COMPLETE
    assert event.final_result == {"answer": "42"}
    assert event.total_duration_ms == 1500
    assert event.token_count == 100
