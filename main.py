#!/usr/bin/env python3
"""
Dex MCP Server - Model Context Protocol server for browser automation.

This server provides tools to interact with a browser extension via WebSocket,
allowing AI models to perform browser actions like getting tabs, taking screenshots,
and navigating to URLs.
"""

import asyncio
import logging
import signal
import sys
import json
from typing import Any, Dict, Optional, List

from mcp.server.fastmcp import FastMCP
from memory.memory_db import MemoryDBHandler

from datetime import datetime, timedelta, timezone

# Import planning tool helpers
from tools.planning import generate_plan_from_llm, get_formatted_tool_schemas

db = MemoryDBHandler()

from context import Context
from ws_server import start_websocket_server
from tools.browser import (
    get_selected_text_tool,
    get_browser_state_tool,
    store_in_memory_tool,
    retrieve_from_memory_tool,
    secret_word_tool,
    get_tabs_tool, 
    screenshot_tool, 
    navigate_tool,
    select_tab_tool,
    new_tab_tool,
    close_tab_tool,
    search_google_tool,
    click_element_tool,
    input_text_tool,
    send_keys_tool,
    grab_dom_tool,
    capture_with_highlights_tool,
    get_current_datetime_tool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and context
mcp = FastMCP("dex-browser")
context = Context()
ws_server = None

def add_fake_memories():
    last_night = datetime.now(timezone.utc) - timedelta(hours=12)
    last_week = datetime.now(timezone.utc) - timedelta(days=14)
    print(f"Simulating 'last_night' as: {last_night.isoformat()}")
    print(f"Simulating 'last_week' as: {last_week.isoformat()}")

    # Add memories with URLs
    mem_id1 = db.add_memory(
        text="Darius from League of Legends is a great champion. Although he is kind of overpowered.",
        tags=["topic:league_of_legends", "source:reading"],
        url="https://universe.leagueoflegends.com/en_US/champion/darius/"
    )
    if mem_id1 and mem_id1 in db.memories_data:
        db.memories_data[mem_id1]['timestamp'] = last_week

    mem_id2 = db.add_memory(
        text="I was reading an article about how queen sacrifices can shift the tempo in a middle game.",
        tags=["topic:chess", "source:reading"],
        url="https://www.chess.com/"
    )
    if mem_id2 and mem_id2 in db.memories_data:
        db.memories_data[mem_id2]['timestamp'] = last_night

    mem_id3 = db.add_memory(
        text="The attention mechanism in LLMs is fascinating, attenion is all you need, its so effective!",
        tags=["topic:ai", "source:reading"],
        url=None
    )
    if mem_id3 and mem_id3 in db.memories_data:
        db.memories_data[mem_id3]['timestamp'] = last_night
    
    print(f"Added {len(db.list_all_memories())} fake memories with adjusted timestamps and URLs.")

@mcp.tool()
async def get_selected_text() -> str:
    """Get the user's selected text from the active tab."""
    resp = await get_selected_text_tool(context)
    print(resp)
    return resp

@mcp.tool()
def get_secret_word() -> str:
    """Get the secret word."""
    return secret_word_tool(context)

@mcp.tool()
async def get_browser_state() -> str:
    """Grab the current browser state (Active tab, open tabs, etc.)"""
    resp = await get_browser_state_tool(context)
    print(resp)
    return resp

@mcp.tool()
async def store_memory(text: str, tags: Optional[List[str]] = None, url: Optional[str] = None) -> str:
    """Stores the provided text content as a memory, with optional tags and URL.

    Tooltips: If you need to store a memory from the user's selected text, use the 'get_selected_text' tool first.

    Args:
        text (str): The text content to store.
        tags (List[str], optional): A list of tags to associate with the memory.
        url (str, optional): A URL to associate with the memory.
    """
    tool_params = {"text": text}
    if tags is not None: # Ensure we only add tags if it's provided (even if empty list)
        tool_params["tags"] = tags
    if url is not None:
        tool_params["url"] = url
    
    resp = await store_in_memory_tool(context, tool_params)
    print(resp)
    return resp

@mcp.tool()
async def retrieve_from_memory(
    similarity_query: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> str:
    """Retrieve memories. Optionally filter by semantic similarity, start_time, and/or end_time.
    
    Tooltips: If you need to calculate a date range relative to the current day (e.g., 'last week', 'yesterday'),
    first use the 'get_current_datetime' tool to get the current date. Then, use that information
    to construct the appropriate ISO 8601 'start_time' and 'end_time' strings for this tool.

    Args:
        similarity_query: Optional text to search for similar memories.
        start_time: Optional ISO 8601 datetime string for the beginning of a time range.
        end_time: Optional ISO 8601 datetime string for the end of a time range.
    """
    add_fake_memories()
    
    tool_params = {}
    if similarity_query:
        tool_params["similarity_query"] = similarity_query
    
    # Construct the nested "time" dictionary if either start_time or end_time is provided
    time_dict = {}
    if start_time:
        time_dict["startDateTime"] = start_time
    if end_time:
        time_dict["endDateTime"] = end_time
    
    if time_dict: # If either start_time or end_time was provided, add the "time" dict
        tool_params["time"] = time_dict
    
    # Example: tool_params could be {"similarity_query": "chess", "time": {"startDateTime": "2023-01-01T00:00:00Z"}}
    # Or {"time": {"endDateTime": "2023-01-07T23:59:59Z"}}
    # Or {"similarity_query": "food"}

    resp = await retrieve_from_memory_tool(context, tool_params)
    print(resp)
    return resp

@mcp.tool()
async def get_tabs() -> str:
    """Get all open browser tabs."""
    return await get_tabs_tool(context)


@mcp.tool()
async def screenshot() -> str:
    """Take a screenshot of the active tab."""
    return await screenshot_tool(context)


@mcp.tool()
async def navigate(url: str) -> str:
    """Navigate to a URL in active tab or specified tab."""
    return await navigate_tool(context, {"url": url})


@mcp.tool()
async def navigate_tab(url: str, tab_id: int) -> str:
    """Navigate to a URL in a specific tab."""
    return await navigate_tool(context, {"url": url, "tab_id": tab_id})


@mcp.tool()
async def select_tab(tab_id: int) -> str:
    """Switch to a specific browser tab by ID."""
    return await select_tab_tool(context, {"tab_id": tab_id})


@mcp.tool()
async def new_tab(url: str = None) -> str:
    """Create a new browser tab, optionally with a specific URL."""
    params = {}
    if url:
        params["url"] = url
    return await new_tab_tool(context, params)


@mcp.tool()
async def close_tab(tab_id: int = None) -> str:
    """Close a browser tab by ID, or close the active tab if no ID specified."""
    params = {}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await close_tab_tool(context, params)


@mcp.tool()
async def search_google(query: str, tab_id: int = None) -> str:
    """Perform a Google search in active tab or specified tab."""
    params = {"query": query}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await search_google_tool(context, params)


@mcp.tool()
async def click_element(element_id: str, tab_id: int = None) -> str:
    """Click on a DOM element by its ID."""
    params = {"element_id": element_id}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await click_element_tool(context, params)


@mcp.tool()
async def input_text(element_id: str, text: str, tab_id: int = None) -> str:
    """Type text into a DOM element by its ID."""
    params = {"element_id": element_id, "text": text}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await input_text_tool(context, params)


@mcp.tool()
async def send_keys(keys: str, tab_id: int = None) -> str:
    """Send keyboard shortcuts or key combinations to the page."""
    params = {"keys": keys}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await send_keys_tool(context, params)


@mcp.tool()
async def grab_dom(tab_id: int = None) -> str:
    """Get formatted DOM structure with XPath mappings for elements."""
    params = {}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await grab_dom_tool(context, params)


@mcp.tool()
async def capture_with_highlights(tab_id: int = None) -> str:
    """Take a screenshot with element highlights for better AI understanding."""
    params = {}
    if tab_id is not None:
        params["tab_id"] = tab_id
    return await capture_with_highlights_tool(context, params)

@mcp.tool()
def get_current_datetime() -> str:
    """Gets the current date and time in UTC, formatted as an ISO 8601 string.
    Returns a JSON string: {"current_datetime_utc": "YYYY-MM-DDTHH:MM:SSZ"}.
    Useful for constructing date ranges for other tools.
    """
    return get_current_datetime_tool()

@mcp.tool()
async def plan_actions(task_description: str) -> str:
    """Generate a comprehensive action plan for complex tasks using an expert planning LLM.
    
    This tool creates detailed, multi-phase plans that break down complex requests into 
    specific, actionable steps with tool calls and reasoning. Use this for any complex
    or multi-step request where you need guidance on how to proceed autonomously.
    
    The planner will provide:
    - Phase-based breakdown of the task
    - Specific tool calls with parameters for each step
    - Reasoning and context for each action
    - Comprehensive coverage of the entire workflow
    
    Perfect for research tasks, learning requests, content discovery, and any scenario
    where you need to orchestrate multiple tools to achieve a goal.

    Args:
        task_description (str): A natural language description of the complex task to be planned.
    
    Returns:
        str: A detailed planning document with phases, steps, tool calls, and reasoning.
    """
    logger.info(f"Plan_actions tool called with task: {task_description}")
    
    # Get schemas of other tools, excluding this planning tool itself
    available_tools = get_formatted_tool_schemas(mcp, exclude_tool_name="plan_actions")
    
    if not available_tools:
        logger.warning("No other tools found for the planning agent to use.")
        return "Unable to generate plan - no tools available for task execution."

    comprehensive_plan = await generate_plan_from_llm(task_description, available_tools)

    if not comprehensive_plan or comprehensive_plan.strip() == "":
        logger.error("Planning LLM returned empty or None plan")
        return "Unable to generate a plan for this task. Please try rephrasing your request."

    logger.info(f"Comprehensive plan generated (length: {len(comprehensive_plan)} characters)")
    return comprehensive_plan

async def start_background_services():
    """Start background services like WebSocket server."""
    global ws_server
    try:
        logger.info("Starting Dex MCP Server...")
        ws_server = await start_websocket_server(context)
        logger.info("WebSocket server started successfully")
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        raise


async def cleanup():
    """Clean up resources."""
    global ws_server
    logger.info("Shutting down servers...")
    
    # Close WebSocket connections
    await context.close()
    
    # Close WebSocket server
    if ws_server:
        ws_server.close()
        await ws_server.wait_closed()
    
    logger.info("Shutdown complete")


# Set up signal handlers for graceful shutdown
def setup_signal_handlers():
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Set up signal handlers
    setup_signal_handlers()
    
    # Start background services before running MCP server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start WebSocket server on the newly-created loop
        loop.run_until_complete(start_background_services())

        # ---- Run MCP server in a background thread ----------------------
        import threading

        def run_mcp():
            try:
                logger.info("Starting MCP server with SSE transport…")
                mcp.run(transport="sse")
            except Exception as e:
                logger.error(f"MCP server stopped: {e}")

        mcp_thread = threading.Thread(target=run_mcp, daemon=True)
        mcp_thread.start()

        # ---- Keep the asyncio loop alive for the WebSocket server ------
        loop.run_forever()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Clean up
        try:
            loop.run_until_complete(cleanup())
        except Exception:
            pass
        loop.stop()
        loop.close()

