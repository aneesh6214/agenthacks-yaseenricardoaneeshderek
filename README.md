# Dex MCP Server

A Model Context Protocol (MCP) server that enables AI models to perform complete browser automation through the [Dex browser extension](https://chromewebstore.google.com/detail/dex/ignaljemchdlmoimaliinmonecgbdnkk?authuser=3&hl=en). You'll have to login to the account you signed up with for your hackathon to download the extension.

Note: you can choose which MCP tools to expose to the agent, and in fact define your own agents as MCP tool calls (eg, @mcp_tool github_agent(instruction: str)), and you can internally chain together the tool calls already built out to interface with the user's browser.

On mac, use command-J to activate Dex.

## From author

There might be a couple of bugs so would appreciate feedback / reports so I can put out some hotfixes as we go :)

**Hotfixes:**

- Added logging to chrome extension and browser to see the LLM's full response & tool results

- Added ability to refresh MCP server on browser

Update instructions:
1. go to chrome://extensions/ 

2. toggle developer mode

3. update, should be 2.59

## Overview

This MCP server provides browser automation capabilities including tab management, navigation, DOM interaction, and visual analysis.

## Features

### 📑 Tab Management
- **`get_tabs`**: Get all open browser tabs with titles and URLs
- **`select_tab(tab_id)`**: Switch to a specific browser tab
- **`new_tab(url?)`**: Create new tab, optionally with a URL
- **`close_tab(tab_id?)`**: Close specific tab or active tab

### 🧭 Navigation
- **`navigate(url, tab_id?)`**: Navigate to URL in active or specific tab
- **`search_google(query, tab_id?)`**: Perform Google search

### 🖱️ DOM Interaction
- **`click_element(element_id, tab_id?)`**: Click DOM elements by ID
- **`input_text(element_id, text, tab_id?)`**: Type text into form fields
- **`send_keys(keys, tab_id?)`**: Send keyboard shortcuts (Ctrl+C, Enter, etc.)

### 📸 Visual Analysis
- **`screenshot()`**: Capture screenshot of active tab
- **`capture_with_highlights(tab_id?)`**: Screenshot with interactive element highlights
- **`grab_dom(tab_id?)`**: Get formatted DOM structure with XPath mappings

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the MCP server:
```bash
uv run main.py
```

The server will:
- Start an MCP server with SSE transport on the default port
- Start a WebSocket server on `ws://127.0.0.1:8765` for browser extension connections

## Browser Extension Integration

The server connects via WebSocket to `ws://127.0.0.1:8765` and handles these message types:

### Message Types

| Type | Parameters | Description |
|------|------------|-------------|
| `get_tabs` | None | Get all browser tabs |
| `screenshot` | None | Screenshot active tab |
| `navigate` | `url`, `tab_id?` | Navigate to URL |
| `select_tab` | `tab_id` | Switch to tab |
| `new_tab` | `url?` | Create new tab |
| `close_tab` | `tab_id?` | Close tab |
| `search_google` | `query`, `tab_id?` | Google search |
| `click_element` | `element_id`, `tab_id?` | Click DOM element |
| `input_text` | `element_id`, `text`, `tab_id?` | Type into element |
| `send_keys` | `keys`, `tab_id?` | Send keyboard input |
| `grab_dom` | `tab_id?` | Get DOM structure |
| `capture_with_highlights` | `tab_id?` | Screenshot with highlights |

### Extension Response Examples

**get_tabs:**
```json
{
  "result": {
    "action": "get_all_tabs",
    "success": true,
    "data": [
      {"id": 1, "title": "Example", "url": "https://example.com"},
      {"id": 2, "title": "Google", "url": "https://google.com"}
    ]
  }
}
```

**screenshot:**
```json
{
  "result": {
    "action": "screenshot_active_tab",
    "success": true,
    "message": "Screenshot captured",
    "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

**new_tab:**
```json
{
  "result": {
    "action": "new_tab",
    "success": true,
    "message": "New tab opened",
    "data": {"id": 1234}
  }
}
```

**grab_dom:**
```json
{
  "result": {
    "success": true,
    "data": {
      "processedOutput": "Formatted DOM structure...",
      "highlightToXPath": {
        "1": "/html/body/button[1]",
        "2": "/html/body/form/input[1]"
      },
      "html": "<html>...</html>"
    }
  }
}
```

**capture_with_highlights:**
```json
{
  "result": {
    "action": "capture_tab_with_highlights",
    "success": true,
    "message": "Screenshot captured with highlight data",
    "data": {
      "dataUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "highlightCount": 15
    }
  }
}
```

## Tool Parameters Reference

### Required Parameters
- `navigate`: `url`
- `select_tab`: `tab_id`  
- `search_google`: `query`
- `click_element`: `element_id`
- `input_text`: `element_id`, `text`
- `send_keys`: `keys`

### Optional Parameters
- `tab_id`: Most tools support targeting specific tabs (defaults to active tab)
- `url`: `new_tab` can optionally specify initial URL

## Example Workflows

**Multi-step Automation:**
1. `get_tabs()` - See all open tabs
2. `new_tab("https://example.com")` - Open new tab
3. `grab_dom()` - Analyze page structure  
4. `click_element("search-button")` - Interact with page
5. `input_text("search-input", "query")` - Fill forms
6. `send_keys("Enter")` - Submit forms
7. `capture_with_highlights()` - Take annotated screenshot

## Architecture

- **main.py**: Entry point and MCP tool definitions
- **context.py**: WebSocket connection management and message handling
- **ws_server.py**: WebSocket server for browser extension connections
- **tools/browser.py**: Complete browser action implementations

## Development

To add new browser tools:

1. Create the tool function in `tools/browser.py` with proper parameter documentation
2. Add the MCP tool wrapper in `main.py` 
3. Add message handler in browser extension's WebSocket bridge
4. Update this documentation

All tools return formatted strings describing action results and extracted data.

## Logging

The server logs all important events:
- WebSocket connections/disconnections
- Message exchanges with browser extension
- Action successes/failures
- Errors and timeouts

