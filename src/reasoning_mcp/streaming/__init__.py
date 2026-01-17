"""
Streaming functionality for reasoning-mcp.

This module provides real-time streaming capabilities for reasoning operations,
enabling progressive output and event-driven updates.
"""

from reasoning_mcp.streaming.backpressure import (
    BackpressureConfig,
    BackpressureError,
    BackpressureHandler,
    BackpressureStrategy,
)
from reasoning_mcp.streaming.buffer import EventBuffer
from reasoning_mcp.streaming.context import StreamingContext
from reasoning_mcp.streaming.emitter import AsyncStreamEmitter, StreamEmitterProtocol
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
from reasoning_mcp.streaming.metrics import MetricsCollector, StreamingMetrics
from reasoning_mcp.streaming.serialization import JSONLSerializer, SSESerializer

__all__ = [
    # Event types
    "StreamingEventType",
    "BaseStreamEvent",
    "ThoughtEvent",
    "ProgressEvent",
    "StageEvent",
    "TokenEvent",
    "ErrorEvent",
    "CompleteEvent",
    # Emitter
    "StreamEmitterProtocol",
    "AsyncStreamEmitter",
    # Context
    "StreamingContext",
    # Backpressure
    "BackpressureStrategy",
    "BackpressureConfig",
    "BackpressureHandler",
    "BackpressureError",
    # Buffer
    "EventBuffer",
    # Serialization
    "SSESerializer",
    "JSONLSerializer",
    # Metrics
    "StreamingMetrics",
    "MetricsCollector",
]
