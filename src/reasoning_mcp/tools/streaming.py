"""Streaming tools for MCP."""

from uuid import uuid4

from pydantic import BaseModel, Field

from reasoning_mcp.streaming.backpressure import BackpressureStrategy
from reasoning_mcp.streaming.context import StreamingContext
from reasoning_mcp.streaming.emitter import AsyncStreamEmitter

# Store active streaming sessions
_active_sessions: dict[str, StreamingContext] = {}


class StreamingToolInput(BaseModel):
    """Input for starting a streaming session."""

    enable_streaming: bool = True
    buffer_size: int = Field(default=100, ge=1)
    backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK


class StreamingSessionResult(BaseModel):
    """Result of streaming session operations."""

    session_id: str
    status: str
    message: str


async def start_streaming_session_impl(
    input: StreamingToolInput,
) -> StreamingSessionResult:
    """Initialize a streaming session."""
    session_id = str(uuid4())
    emitter = AsyncStreamEmitter(queue_size=input.buffer_size)
    ctx = StreamingContext(emitter=emitter, session_id=session_id, enabled=input.enable_streaming)
    _active_sessions[session_id] = ctx
    return StreamingSessionResult(
        session_id=session_id, status="active", message="Streaming session started"
    )


async def stop_streaming_session_impl(session_id: str) -> StreamingSessionResult:
    """Stop a streaming session."""
    ctx = _active_sessions.pop(session_id, None)
    if ctx:
        await ctx.close()
        return StreamingSessionResult(
            session_id=session_id, status="closed", message="Session closed"
        )
    return StreamingSessionResult(
        session_id=session_id, status="not_found", message="Session not found"
    )


def get_streaming_context(session_id: str) -> StreamingContext | None:
    """Get an active streaming context."""
    return _active_sessions.get(session_id)


def list_active_sessions() -> list[str]:
    """List all active streaming session IDs."""
    return list(_active_sessions.keys())
