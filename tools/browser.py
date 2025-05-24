"""Browser tools for interacting with the browser extension."""

from typing import Any, Dict, Optional
from context import Context
import json
from datetime import datetime, timezone, timedelta
from memory.memory_db import MemoryDBHandler

async def get_browser_state_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Gathers information about the current browser state, including open tabs and the active tab.
    
    Params: None
    Returns: A JSON string describing the browser state.
    """
    browser_state = {
        "active_tab": None,
        "all_tabs": [],
        "error": None
    }

    try:
        tabs_result_raw = await context.send_socket_message("get_tabs", {})
        
        if not tabs_result_raw or "tabs" not in tabs_result_raw:
            browser_state["error"] = "No tabs found or unable to fetch tabs via WebSocket."
            return json.dumps(browser_state, indent=2)
            
        tabs_data = tabs_result_raw["tabs"]
        if not tabs_data:
            browser_state["error"] = "No open tabs found via WebSocket."
            return json.dumps(browser_state, indent=2)

        formatted_tabs = []
        active_tab_info = None

        for tab in tabs_data:
            tab_info = {
                "id": tab.get('id'),
                "title": tab.get('title', 'Untitled'),
                "url": tab.get('url'),
                "active": tab.get('active', False)
            }
            formatted_tabs.append(tab_info)
            if tab_info["active"]:
                active_tab_info = tab_info
        
        browser_state["all_tabs"] = formatted_tabs

        if active_tab_info:
            browser_state["active_tab"] = active_tab_info
        elif formatted_tabs:
            print("=====> No active_tab in get_tabs.")
            pass

        if not browser_state["active_tab"] and formatted_tabs:
            print("=====> No active_tab in get_tabs.")

    except Exception as e:
        browser_state["error"] = f"Error gathering browser state: {str(e)}"

    return json.dumps(browser_state, indent=2) 

async def retrieve_from_memory_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """
    Retrieves memories from the MemoryDB based on time range and/or similarity query.
    Params:
        time: Dict with optional 'startDateTime' and 'endDateTime' (ISO 8601 strings).
        similarity_query: Optional string for semantic search.
        limit: Optional integer for the number of results (defaults to 5 in MemoryDBHandler).

    Returns: JSON string of matching memories.
    """
    
    # Helper to parse ISO 8601 datetime strings, nested to keep it local to this tool
    def _parse_datetime_utc(datetime_str: Optional[str]) -> Optional[datetime]:
        if not datetime_str:
            return None
        try:
            # Handle 'Z' for UTC explicitly
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            # If no timezone info, assume UTC. If it has timezone, convert to UTC.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError as e:
            # Optionally, log this error or handle it if the tool needs to communicate parsing failures
            print(f"Error parsing datetime string '{datetime_str}': {e}")
            return None

    if params is None:
        params = {}

    db_handler = MemoryDBHandler() # Get singleton instance

    time_params = params.get("time", {})
    start_time_str = time_params.get("startDateTime")
    end_time_str = time_params.get("endDateTime")
    
    similarity_query = params.get("similarity_query")
    # Use a sensible default for limit, or make it configurable if needed
    limit = params.get("limit", 5) 

    start_time_dt: Optional[datetime] = _parse_datetime_utc(start_time_str)
    end_time_dt: Optional[datetime] = _parse_datetime_utc(end_time_str)

    try:
        results = db_handler.query_memories(
            start_time=start_time_dt,
            end_time=end_time_dt,
            similarity_query_text=similarity_query,
            limit=limit
        )
    except Exception as e:
        # Log the exception from db_handler if necessary
        # print(f"Error querying memories: {e}")
        return json.dumps({"error": f"Failed to query memories: {str(e)}"}, indent=2)


    # Convert datetime objects in results to ISO 8601 strings for JSON serialization
    for memory_item in results: # Renamed to avoid conflict with top-level memory module
        if isinstance(memory_item.get("timestamp"), datetime):
            memory_item["timestamp"] = memory_item["timestamp"].isoformat()

    return json.dumps(results, indent=2)

async def get_selected_text_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Get the selected text from the active tab."""
    return await context.send_socket_message("get_selected_text", {})

async def store_in_memory_tool(context: Context, params: Dict[str, Any]) -> str:
    """Stores the provided text and optional tags/URL as a new memory.
    
    Params: 
        text (str, required): The text content to store as a memory.
        tags (List[str], optional): A list of tags to associate with the memory.
        url (str, optional): A URL to associate with the memory.
    Returns: A message indicating success (with memory ID) or failure.
    """
    if not params or "text" not in params or not params["text"].strip():
        return "❌ Error: 'text' parameter is required and cannot be empty." 

    text_to_store = params["text"].strip()
    user_tags = params.get("tags")
    url_to_store = params.get("url")

    try:
        db_handler = MemoryDBHandler() # Get singleton instance
        
        # Prepare tags
        final_tags = []
        if isinstance(user_tags, list):
            final_tags.extend(user_tags)
        elif user_tags is not None:
            # If tags are provided but not as a list, consider logging a warning or handling as an error
            # For now, we'll just ignore non-list tags if provided that way.
            print(f"Warning: 'tags' parameter was provided but not a list: {user_tags}. Ignoring these tags.")

        # Add a default tag if you still want to differentiate memories added this way
        # e.g., final_tags.insert(0, "source:manual_add") 
        # For now, let's assume the caller provides all necessary tags.

        memory_id = db_handler.add_memory(text=text_to_store, tags=final_tags if final_tags else None, url=url_to_store)
        
        response_parts = [
            f"✅ Text stored as memory. ID: {memory_id}",
            f"Text: '{text_to_store[:100]}...'"
        ]
        if final_tags:
            response_parts.append(f"Tags: {final_tags}")
        if url_to_store:
            response_parts.append(f"URL: {url_to_store}")
            
        return "\n".join(response_parts)

    except Exception as e:
        # Consider logging the full exception e for debugging
        return f"❌ Tool error in store_in_memory_tool: {str(e)}"

def secret_word_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Get the secret word.
    
    Params: None
    """
    return "Secret word is kebab"

async def get_tabs_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Get all open browser tabs.
    
    Params: None
    """
    try:
        result = await context.send_socket_message("get_tabs", {})
        
        if not result or "tabs" not in result:
            return "No tabs found or unable to fetch tabs."
        
        tabs = result["tabs"]
        if not tabs:
            return "No open tabs found."
        
        # Format tabs into readable text
        tab_list = []
        for tab in tabs:
            tab_info = f"Tab {tab.get('id', 'Unknown')}: {tab.get('title', 'Untitled')}"
            if tab.get('url'):
                tab_info += f"\n  URL: {tab['url']}"
            tab_list.append(tab_info)
        
        return f"Found {len(tabs)} open tabs:\n\n" + "\n\n".join(tab_list)
        
    except Exception as e:
        return f"Error getting tabs: {str(e)}"


async def screenshot_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Take a screenshot of the active tab.
    
    Params: None
    """
    try:
        result = await context.send_socket_message("screenshot", {})
        
        if not result:
            return "Failed to take screenshot."
        
        if result.get("success"):
            data_url = result.get("data")
            message = result.get("message", "Screenshot taken")
            if data_url:
                return f"Screenshot captured successfully: {message}\nScreenshot data available (base64 data URL: {len(data_url)} characters)"
            else:
                return f"Screenshot captured successfully: {message}"
        else:
            return f"Failed to take screenshot: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"


async def navigate_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Navigate to a URL in active tab or specified tab.
    
    Params:
        url (str): Required - URL to navigate to
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    if not params or "url" not in params:
        return "Error: URL parameter is required"
    
    url = params["url"]
    tab_id = params.get("tab_id")
    
    payload = {"url": url}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("navigate", payload)
        
        if not result:
            return f"Failed to navigate to {url}"
        
        if result.get("success"):
            message = result.get("message", f"Navigated to {url}")
            action = result.get("action", "go_to_url")
            return f"Successfully navigated to {url}\nAction: {action} - {message}"
        else:
            return f"Failed to navigate to {url}: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error navigating to {url}: {str(e)}"


async def select_tab_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Switch to a specific browser tab by ID.
    
    Params:
        tab_id (int): Required - Tab ID to switch to
    """
    if not params or "tab_id" not in params:
        return "Error: tab_id parameter is required"
    
    tab_id = params["tab_id"]
    
    try:
        result = await context.send_socket_message("select_tab", {"tab_id": tab_id})
        
        if not result:
            return f"Failed to select tab {tab_id}"
        
        if result.get("success"):
            message = result.get("message", "Tab selected")
            action = result.get("action", "select_tab")
            return f"Successfully switched to tab {tab_id}\nAction: {action} - {message}"
        else:
            return f"Failed to select tab {tab_id}: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error selecting tab {tab_id}: {str(e)}"


async def new_tab_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Create a new browser tab, optionally with a specific URL.
    
    Params:
        url (str): Optional - URL to open in new tab, defaults to blank tab
    """
    url = params.get("url") if params else None
    
    payload = {}
    if url:
        payload["url"] = url
    
    try:
        result = await context.send_socket_message("new_tab", payload)
        
        if not result:
            return "Failed to create new tab"
        
        if result.get("success"):
            tab_id = result.get("data", {}).get("id")
            message = result.get("message", "New tab created")
            action = result.get("action", "new_tab")
            
            if url:
                return f"Successfully created new tab (ID: {tab_id}) with URL: {url}\nAction: {action} - {message}"
            else:
                return f"Successfully created new tab (ID: {tab_id})\nAction: {action} - {message}"
        else:
            return f"Failed to create new tab: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error creating new tab: {str(e)}"


async def close_tab_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Close a browser tab by ID, or close the active tab if no ID specified.
    
    Params:
        tab_id (int): Optional - Tab ID to close, defaults to active tab
    """
    tab_id = params.get("tab_id") if params else None
    
    payload = {}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("close_tab", payload)
        
        if not result:
            return f"Failed to close tab {tab_id if tab_id else '(active)'}"
        
        if result.get("success"):
            message = result.get("message", "Tab closed")
            action = result.get("action", "close_tab")
            return f"Successfully closed tab {tab_id if tab_id else '(active)'}\nAction: {action} - {message}"
        else:
            return f"Failed to close tab: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error closing tab: {str(e)}"


async def search_google_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Perform a Google search in active tab or specified tab.
    
    Params:
        query (str): Required - Search query text
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    if not params or "query" not in params:
        return "Error: query parameter is required"
    
    query = params["query"]
    tab_id = params.get("tab_id")
    
    payload = {"query": query}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("search_google", payload)
        
        if not result:
            return f"Failed to search Google for '{query}'"
        
        if result.get("success"):
            message = result.get("message", f"Searched Google for {query}")
            action = result.get("action", "search_google")
            return f"Successfully searched Google for '{query}'\nAction: {action} - {message}"
        else:
            return f"Failed to search Google for '{query}': {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error searching Google for '{query}': {str(e)}"


async def click_element_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Click on a DOM element by its ID.
    
    Params:
        element_id (str): Required - Element ID to click
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    if not params or "element_id" not in params:
        return "Error: element_id parameter is required"
    
    element_id = params["element_id"]
    tab_id = params.get("tab_id")
    
    payload = {"element_id": element_id}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("click_element", payload)
        
        if not result:
            return f"Failed to click element '{element_id}'"
        
        if result.get("success"):
            message = result.get("message", f"Clicked element '{element_id}'")
            action = result.get("action", "click_element")
            return f"Successfully clicked element '{element_id}'\nAction: {action} - {message}"
        else:
            return f"Failed to click element '{element_id}': {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error clicking element '{element_id}': {str(e)}"


async def input_text_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Type text into a DOM element by its ID.
    
    Params:
        element_id (str): Required - Element ID to type into
        text (str): Required - Text to input
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    if not params or "element_id" not in params or "text" not in params:
        return "Error: element_id and text parameters are required"
    
    element_id = params["element_id"]
    text = params["text"]
    tab_id = params.get("tab_id")
    
    payload = {"element_id": element_id, "text": text}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("input_text", payload)
        
        if not result:
            return f"Failed to input text into element '{element_id}'"
        
        if result.get("success"):
            message = result.get("message", f"Input text into element '{element_id}'")
            action = result.get("action", "input_text")
            return f"Successfully input text '{text}' into element '{element_id}'\nAction: {action} - {message}"
        else:
            return f"Failed to input text into element '{element_id}': {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error inputting text into element '{element_id}': {str(e)}"


async def send_keys_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Send keyboard shortcuts or key combinations to the page.
    
    Params:
        keys (str): Required - Key combination (e.g. 'Ctrl+C', 'Enter', 'Tab')
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    if not params or "keys" not in params:
        return "Error: keys parameter is required"
    
    keys = params["keys"]
    tab_id = params.get("tab_id")
    
    payload = {"keys": keys}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("send_keys", payload)
        
        if not result:
            return f"Failed to send keys '{keys}'"
        
        if result.get("success"):
            message = result.get("message", f"Sent keys '{keys}'")
            action = result.get("action", "send_keys")
            return f"Successfully sent keys '{keys}'\nAction: {action} - {message}"
        else:
            return f"Failed to send keys '{keys}': {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error sending keys '{keys}': {str(e)}"


async def grab_dom_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Get formatted DOM structure with XPath mappings for elements.
    
    Params:
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    tab_id = params.get("tab_id") if params else None
    
    payload = {}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("grab_dom", payload)
        
        if not result:
            return "Failed to grab DOM structure"
        
        if result.get("success"):
            # Extract all the GrabDomResult data
            data = result.get("data", {})
            processed_output = data.get("processedOutput", "")
            highlight_to_xpath = data.get("highlightToXPath", {})
            html = data.get("html", "")
            
            response_parts = ["DOM structure captured successfully:"]
            
            if processed_output:
                response_parts.append(f"\nProcessed DOM Output:\n{processed_output}")
            
            if highlight_to_xpath:
                response_parts.append(f"\nFound {len(highlight_to_xpath)} interactive elements with XPath mappings:")
                for highlight_id, xpath in highlight_to_xpath.items():
                    response_parts.append(f"  Element {highlight_id}: {xpath}")
            
            if html:
                response_parts.append(f"\nRaw HTML available ({len(html)} characters)")
            
            return "\n".join(response_parts)
        else:
            return f"Failed to grab DOM: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error grabbing DOM: {str(e)}"


async def capture_with_highlights_tool(context: Context, params: Dict[str, Any] = None) -> str:
    """Take a screenshot with element highlights for better AI understanding.
    
    Params:
        tab_id (int): Optional - Specific tab ID, defaults to active tab
    """
    tab_id = params.get("tab_id") if params else None
    
    payload = {}
    if tab_id is not None:
        payload["tab_id"] = tab_id
    
    try:
        result = await context.send_socket_message("capture_with_highlights", payload)
        
        if not result:
            return "Failed to capture screenshot with highlights"
        
        if result.get("success"):
            data = result.get("data", {})
            data_url = data.get("dataUrl")
            highlight_count = data.get("highlightCount", 0)
            message = result.get("message", "Screenshot with highlights captured")
            
            response_parts = [f"Screenshot with highlights captured successfully: {message}"]
            response_parts.append(f"Found {highlight_count} highlighted interactive elements")
            
            if data_url:
                response_parts.append(f"Screenshot data available (base64 data URL: data_url)")
                response_parts.append("This screenshot includes visual highlights on interactive elements to help identify clickable areas")
            
            return "\n".join(response_parts)
        else:
            return f"Failed to capture screenshot with highlights: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error capturing screenshot with highlights: {str(e)}"

def get_current_datetime_tool(context: Optional[Context] = None, params: Optional[Dict[str, Any]] = None) -> str:
    """Gets the current date and time in UTC, formatted as an ISO 8601 string.
    
    Params: None
    Returns: A JSON string with the current UTC datetime, e.g., {"current_datetime_utc": "YYYY-MM-DDTHH:MM:SSZ"}.
    """
    now_utc = datetime.now(timezone.utc)
    response = {
        "current_datetime_utc": now_utc.isoformat().replace("+00:00", "Z")
    }
    return json.dumps(response, indent=2)