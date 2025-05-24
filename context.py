import asyncio
import json
import uuid
from typing import Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol


class Context:
    def __init__(self):
        self._ws: Optional[WebSocketServerProtocol] = None
        self._pending_requests: dict[str, asyncio.Future] = {}

    @property
    def ws(self) -> WebSocketServerProtocol:
        if not self._ws:
            raise Exception("No connection to browser extension. Please connect your browser extension first.")
        return self._ws

    def set_ws(self, ws: WebSocketServerProtocol):
        """Set the active WebSocket connection."""
        # Replace the active WebSocket without explicitly closing the old one.
        # Closing it here causes the browser extension to immediately reconnect,
        # resulting in a connection-churn loop. Let the client decide when to
        # disconnect instead.
        self._ws = ws

    def has_ws(self) -> bool:
        """Check if we have an active WebSocket connection."""
        return self._ws is not None

    async def send_socket_message(self, message_type: str, payload: dict = None, timeout: float = 30.0) -> Any:
        """Send a message to the browser extension and wait for response."""
        if not self.has_ws():
            raise Exception("No connection to browser extension. Please connect your browser extension first.")
        
        message_id = str(uuid.uuid4())
        message = {
            "id": message_id,
            "type": message_type,
            "payload": payload or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[message_id] = future
        
        try:
            # Send message
            await self.ws.send(json.dumps(message))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)
            
            if "error" in response:
                raise Exception(f"Browser extension error: {response['error']}")
            
            return response.get("result")
            
        except asyncio.TimeoutError:
            raise Exception(f"Timeout waiting for response from browser extension")
        finally:
            # Clean up pending request
            self._pending_requests.pop(message_id, None)

    def handle_response(self, message: dict):
        """Handle incoming response from browser extension."""
        message_id = message.get("id")
        if message_id in self._pending_requests:
            future = self._pending_requests[message_id]
            if not future.done():
                future.set_result(message)

    async def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None 