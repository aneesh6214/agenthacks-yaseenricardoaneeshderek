import json
import os
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. Planning functions may not work.")

# Initialize OpenAI client
aclient = None
if OPENAI_API_KEY:
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_plan_from_llm(task_description: str, available_tools_schema: list) -> str:
    """
    Calls an external OpenAI LLM to generate a comprehensive plan based on the task and available tools.
    Returns a detailed plan as a string with both tool calls and reasoning.
    """
    if not aclient:
        print("Error: OpenAI client not initialized. Missing API key for planning.")
        return "No plan generated - OpenAI client not available."

    system_prompt_content = f"""
You are an expert autonomous planning agent. Your role is to create comprehensive, detailed action plans for complex tasks.

Available tools and their capabilities:
{json.dumps(available_tools_schema, indent=2)}

Your planning approach should be:
1. THOROUGH: Break down complex requests into comprehensive step-by-step plans
2. AUTONOMOUS: Assume the user wants a complete solution, not just the first step
3. CONTEXT-AWARE: Always start by checking memory for relevant information
4. PRACTICAL: Include specific tool calls AND reasoning for each step

Format your response as a detailed plan with this structure:

## COMPREHENSIVE PLAN: [Brief title]

### Phase 1: Information Gathering
- **Step 1**: [Description of what to do]
  - Tool: `tool_name` with params: {{"param": "value"}}
  - Reasoning: [Why this step is important]

### Phase 2: Research & Discovery  
- **Step 2**: [Description]
  - Tool: `tool_name` with params: {{"param": "value"}}
  - Reasoning: [Why this step follows logically]

### Phase 3: Action & Implementation
- **Step 3**: [Description]
  - Tool: `tool_name` with params: {{"param": "value"}}
  - Reasoning: [Expected outcome]

Include as many phases and steps as needed. For learning tasks, always include:
- Memory retrieval to check existing knowledge
- Multiple search strategies (Google, specific sites)
- Resource evaluation and selection
- Content consumption (reading, watching)
- Knowledge storage for future reference

Be specific with tool parameters. Use realistic URLs, search queries, and selectors.
The LLM who sees this plan does not have realtime access to the browser, only the tools.
For example, if you plan to search something on google, make sure you plan to open a new tab also.
"""

    user_prompt_content = f"""
User request: "{task_description}"

Create a comprehensive, autonomous plan that fully addresses this request. Think like an expert research assistant who anticipates all the steps needed for complete success.
"""

    print(f"--- Autonomous Planning Logic ---")
    print(f"Task for planning: {task_description}")

    try:
        chat_completion = await aclient.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_prompt_content}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        
        raw_response_content = chat_completion.choices[0].message.content
        
        if not raw_response_content:
            print("Planning LLM returned empty content.")
            return "No plan could be generated for this task."

        print(f"Generated comprehensive plan (length: {len(raw_response_content)} chars)")
        return raw_response_content.strip()
            
    except openai.APIError as e:
        print(f"OpenAI API Error in planning: {e}")
        return f"Planning error: {str(e)}"
    except Exception as e:
        print(f"Exception during planning LLM call: {e}")
        return f"Planning failed: {str(e)}"

def get_formatted_tool_schemas(mcp_instance, exclude_tool_name: str = None) -> list:
    """
    Introspects registered MCP tools and returns their schemas, formatted for the planning agent.
    Excludes the tool specified by exclude_tool_name (e.g., the planning tool itself).
    """
    available_tools_full_schema = []
    
    tools_to_inspect = {}
    
    if hasattr(mcp_instance, '_tool_manager') and hasattr(mcp_instance._tool_manager, '_tools'):
        tools_to_inspect = mcp_instance._tool_manager._tools
        print(f"Found {len(tools_to_inspect)} tools in _tool_manager._tools")
    elif hasattr(mcp_instance, 'tools') and isinstance(mcp_instance.tools, dict):
        tools_to_inspect = mcp_instance.tools
    elif hasattr(mcp_instance, '_tools') and isinstance(mcp_instance._tools, dict):
        tools_to_inspect = mcp_instance._tools
    else:
        print("Warning: Could not find tools in MCP instance. Checking for list_tools method.")
        if hasattr(mcp_instance, 'list_tools'):
            print("Found list_tools method, but cannot call async method in sync context.")
    
    if not tools_to_inspect:
        print("Warning: Could not dynamically fetch any tool schemas for planning. Planning might be ineffective.")
        return []

    for tool_name, tool_obj in tools_to_inspect.items():
        if exclude_tool_name and tool_name == exclude_tool_name:
            continue
        
        tool_description = getattr(tool_obj, 'description', f"Tool named {tool_name}")
        
        if hasattr(tool_obj, 'target_function') and callable(tool_obj.target_function):
            docstring = tool_obj.target_function.__doc__
            if docstring:
                first_line_of_docstring = docstring.strip().split('\n')[0]
                if first_line_of_docstring: # Ensure it's not empty
                    tool_description = first_line_of_docstring

        params_json_schema = {"type": "object", "properties": {}}
        if hasattr(tool_obj, 'inputSchema'):
            params_json_schema = tool_obj.inputSchema
        elif hasattr(tool_obj, 'parameters_schema'):
            params_json_schema = tool_obj.parameters_schema
        
        if not isinstance(params_json_schema, dict) or "type" not in params_json_schema:
             print(f"Warning: parameters_schema for tool '{tool_name}' is not a valid JSON schema dict. Using default.")
             params_json_schema = {"type": "object", "properties": {}}

        schema_entry = {
            "name": tool_name,
            "description": tool_description,
            "parameters": params_json_schema
        }
        available_tools_full_schema.append(schema_entry)
    
    print(f"Successfully generated {len(available_tools_full_schema)} tool schemas for planning.")
    return available_tools_full_schema