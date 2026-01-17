"""Server-Sent Events transport for streaming."""

from collections.abc import AsyncIterator

from starlette.responses import StreamingResponse

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import BaseStreamEvent
from reasoning_mcp.streaming.serialization import SSESerializer


class SSETransport:
    """Transport for Server-Sent Events (SSE) streaming.

    Provides utilities for formatting events as SSE and creating
    streaming HTTP responses.
    """

    def __init__(self, serializer: SSESerializer | None = None) -> None:
        """Initialize the SSE transport.

        Args:
            serializer: Optional custom serializer. Uses default if not provided.
        """
        self._serializer = serializer or SSESerializer()

    def format_event(self, event: BaseStreamEvent) -> str:
        """Format an event as SSE data.

        Args:
            event: The streaming event to format.

        Returns:
            SSE-formatted string.
        """
        return SSESerializer.serialize(event)

    async def _event_generator(self, emitter: AsyncStreamEmitter) -> AsyncIterator[str]:
        """Generate SSE-formatted events from emitter.

        Args:
            emitter: The stream emitter to consume events from.

        Yields:
            SSE-formatted event strings.
        """
        async for event in emitter:
            yield self.format_event(event)

    def create_response(
        self,
        emitter: AsyncStreamEmitter,
        media_type: str = "text/event-stream",
    ) -> StreamingResponse:
        """Create a Starlette StreamingResponse for SSE.

        Args:
            emitter: The stream emitter providing events.
            media_type: MIME type for the response.

        Returns:
            StreamingResponse configured for SSE.
        """
        return StreamingResponse(
            self._event_generator(emitter),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
