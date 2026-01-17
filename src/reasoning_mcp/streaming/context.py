"""Streaming context for managing event emission during reasoning."""

from typing import Any
from uuid import uuid4

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import (
    CompleteEvent,
    ErrorEvent,
    ProgressEvent,
    StageEvent,
    StreamingEventType,
    ThoughtEvent,
    TokenEvent,
)


class StreamingContext:
    """Context manager for streaming events during reasoning operations.

    Provides convenient methods for emitting various event types
    and manages the lifecycle of a streaming session.
    """

    def __init__(
        self,
        emitter: AsyncStreamEmitter,
        session_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the streaming context.

        Args:
            emitter: The stream emitter to use for publishing events.
            session_id: Optional session identifier. Auto-generated if not provided.
            enabled: Whether streaming is enabled. If False, all emit calls are no-ops.
        """
        self.emitter = emitter
        self.session_id = session_id or str(uuid4())
        self.enabled = enabled
        self._thought_counter = 0
        self._token_counter = 0
        self._cumulative_text = ""

    async def emit_thought(
        self,
        content: str,
        method_name: str,
        confidence: float | None = None,
    ) -> None:
        """Emit a thought event.

        Args:
            content: The thought content.
            method_name: Name of the reasoning method producing this thought.
            confidence: Optional confidence score (0.0 to 1.0).
        """
        if not self.enabled:
            return
        self._thought_counter += 1
        event = ThoughtEvent(
            session_id=self.session_id,
            thought_number=self._thought_counter,
            content=content,
            method_name=method_name,
            confidence=confidence,
        )
        await self.emitter.emit(event)

    async def emit_progress(
        self,
        current_step: int,
        total_steps: int,
        message: str = "",
    ) -> None:
        """Emit a progress event.

        Args:
            current_step: Current step number.
            total_steps: Total number of steps.
            message: Optional progress message.
        """
        if not self.enabled:
            return
        percentage = (current_step / total_steps * 100) if total_steps > 0 else 0.0
        event = ProgressEvent(
            session_id=self.session_id,
            current_step=current_step,
            total_steps=total_steps,
            percentage=percentage,
            message=message,
        )
        await self.emitter.emit(event)

    async def emit_stage_start(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage start event.

        Args:
            stage_name: Name of the stage being started.
            metadata: Optional metadata about the stage.
        """
        if not self.enabled:
            return
        event = StageEvent(
            session_id=self.session_id,
            event_type=StreamingEventType.STAGE_START,
            stage_name=stage_name,
            stage_type="start",
            metadata=metadata or {},
        )
        await self.emitter.emit(event)

    async def emit_stage_end(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage end event.

        Args:
            stage_name: Name of the stage being ended.
            metadata: Optional metadata about the stage completion.
        """
        if not self.enabled:
            return
        event = StageEvent(
            session_id=self.session_id,
            event_type=StreamingEventType.STAGE_END,
            stage_name=stage_name,
            stage_type="end",
            metadata=metadata or {},
        )
        await self.emitter.emit(event)

    async def emit_token(self, token: str) -> None:
        """Emit a token event for streaming output.

        Args:
            token: The token string.
        """
        if not self.enabled:
            return
        self._token_counter += 1
        self._cumulative_text += token
        event = TokenEvent(
            session_id=self.session_id,
            token=token,
            token_index=self._token_counter,
            cumulative_text=self._cumulative_text,
        )
        await self.emitter.emit(event)

    async def emit_error(
        self,
        error_code: str,
        error_message: str,
        recoverable: bool = False,
        stack_trace: str | None = None,
    ) -> None:
        """Emit an error event.

        Args:
            error_code: Error code identifier.
            error_message: Human-readable error message.
            recoverable: Whether the error can be recovered from.
            stack_trace: Optional stack trace string.
        """
        if not self.enabled:
            return
        event = ErrorEvent(
            session_id=self.session_id,
            error_code=error_code,
            error_message=error_message,
            recoverable=recoverable,
            stack_trace=stack_trace,
        )
        await self.emitter.emit(event)

    async def emit_complete(
        self,
        final_result: dict[str, Any],
        total_duration_ms: int,
    ) -> None:
        """Emit a completion event.

        Args:
            final_result: The final reasoning result.
            total_duration_ms: Total duration in milliseconds.
        """
        if not self.enabled:
            return
        event = CompleteEvent(
            session_id=self.session_id,
            final_result=final_result,
            total_duration_ms=total_duration_ms,
            token_count=self._token_counter,
        )
        await self.emitter.emit(event)

    async def close(self) -> None:
        """Close the streaming context and underlying emitter."""
        await self.emitter.close()
