import asyncio
import json
import logging
import websockets
from websockets.server import WebSocketServerProtocol
from context import Context

logger = logging.getLogger(__name__)


async def handle_websocket_connection(websocket: WebSocketServerProtocol, context: Context):
    """Handle incoming WebSocket connection from browser extension."""
    logger.info("Browser extension connected")
    context.set_ws(websocket)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                # Handle debug log messages separately
                if data.get("type") == "debug_log":
                    payload = data.get("payload", {})
                    logger.info("[DEBUG_LOG] %s", payload.get("message"))
                    continue  # Skip normal handling

                context.handle_response(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Browser extension disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if context.has_ws() and context._ws == websocket:
            context._ws = None


async def start_websocket_server(context: Context, host: str = "127.0.0.1", port: int = 8765):
    """Start the WebSocket server for browser extension connections."""
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    server = await websockets.serve(
        lambda ws: handle_websocket_connection(ws, context),
        host,
        port
    )
    
    logger.info(f"WebSocket server listening on ws://{host}:{port}")
    return server 