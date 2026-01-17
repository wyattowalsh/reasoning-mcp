"""WebSocket transport for streaming."""

from collections.abc import Callable, Coroutine
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from reasoning_mcp.streaming.emitter import AsyncStreamEmitter
from reasoning_mcp.streaming.events import BaseStreamEvent


class WebSocketTransport:
    """Transport for WebSocket streaming.

    Provides utilities for sending events over WebSocket connections
    and creating WebSocket handlers.
    """

    async def send(self, ws: WebSocket, event: BaseStreamEvent) -> None:
        """Send an event over WebSocket.

        Args:
            ws: The WebSocket connection.
            event: The streaming event to send.
        """
        data = event.model_dump_json()
        await ws.send_text(data)

    async def send_batch(self, ws: WebSocket, events: list[BaseStreamEvent]) -> None:
        """Send multiple events over WebSocket.

        Args:
            ws: The WebSocket connection.
            events: List of events to send.
        """
        for event in events:
            await self.send(ws, event)

    def create_handler(
        self,
        emitter: AsyncStreamEmitter,
    ) -> Callable[[WebSocket], Coroutine[Any, Any, None]]:
        """Create a WebSocket handler that streams events.

        Args:
            emitter: The stream emitter providing events.

        Returns:
            Async function that can be used as a WebSocket endpoint.
        """

        async def handler(websocket: WebSocket) -> None:
            await websocket.accept()
            try:
                async for event in emitter.subscribe():
                    await self.send(websocket, event)
            except WebSocketDisconnect:
                pass
            finally:
                await websocket.close()

        return handler

    async def stream_to_websocket(
        self,
        websocket: WebSocket,
        emitter: AsyncStreamEmitter,
    ) -> None:
        """Stream events to an already-connected WebSocket.

        Args:
            websocket: The WebSocket connection (must be accepted).
            emitter: The stream emitter providing events.
        """
        try:
            async for event in emitter.subscribe():
                await self.send(websocket, event)
        except WebSocketDisconnect:
            pass
