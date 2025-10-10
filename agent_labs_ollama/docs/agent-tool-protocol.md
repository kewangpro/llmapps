# Agent and Tool Protocol Specification

This document defines the contracts, naming conventions, and protocols used in Agent Labs for agent-to-agent communication and tool integration.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Naming Conventions](#naming-conventions)
- [Agent Protocol](#agent-protocol)
- [Tool Protocol](#tool-protocol)
- [MCP Integration](#mcp-integration)
- [Data Contracts](#data-contracts)

## Overview

Agent Labs uses a hierarchical multi-agent orchestration pattern where a central OrchestratorAgent coordinates specialized sub-agents. Each agent can execute tools and return structured results.

### Communication Flow
```
User Query → OrchestratorAgent → Sub-Agent(s) → Tool(s) → Results → Orchestrator → User
```

### Agent Types
- **Orchestrator Agent**: Coordinates and routes queries to specialized agents
- **Specialized Agents**: Domain-specific agents (FlightSearchAgent, WebSearchAgent, etc.)
- **MCP Agents**: External agents via Model Context Protocol

## Architecture

### Multi-Agent Pattern
```python
# Hierarchical orchestration
OrchestratorAgent
  ├── FileSearchAgent
  ├── WebSearchAgent
  ├── FlightSearchAgent
  ├── SystemInfoAgent
  ├── CostAnalysisAgent
  ├── DataProcessingAgent
  ├── ImageAnalysisAgent
  ├── StockAnalysisAgent
  ├── VisualizationAgent
  ├── ForecastAgent
  ├── PresentationAgent
  └── MCPAgent (gateway to external tools)
```

## Naming Conventions

### Tool Naming
All tool components follow consistent naming patterns:

#### Tool ID (Internal Identifier)
- **Format**: `snake_case` (lowercase with underscores)
- **Pattern**: `{category}_{action}` or `{domain}_search`
- **Examples**:
  - `file_search` - file discovery tool
  - `web_search` - web search tool
  - `flight_search` - flight search tool
  - `cost_analysis` - cost analysis tool
  - `image_analysis` - image analysis tool

#### Tool Name (Display Name)
- **Format**: Title Case with spaces
- **Pattern**: Human-readable descriptive name
- **Examples**:
  - `"File Search"` - displayed in UI
  - `"Web Search"` - displayed in UI
  - `"Flight Search"` - displayed in UI

#### Tool Script Filename
- **Format**: `{tool_id}.py`
- **Location**: `tools/` directory
- **Examples**:
  - `tools/file_search.py`
  - `tools/web_search.py`
  - `tools/flight_search.py`

#### Agent Class Name
- **Format**: PascalCase with "Agent" suffix
- **Pattern**: `{ToolName}Agent`
- **Examples**:
  - `FileSearchAgent`
  - `WebSearchAgent`
  - `FlightSearchAgent`

#### Agent Filename
- **Format**: `{tool_id}_agent.py`
- **Location**: `backend/agents/` directory
- **Examples**:
  - `backend/agents/file_search_agent.py`
  - `backend/agents/web_search_agent.py`
  - `backend/agents/flight_search_agent.py`

### MCP Tool Naming
MCP tools follow a different pattern:

- **Tool ID**: `{server_name}:{tool_name}` (with colon separator)
- **Tool Name**: `{Server-Name}:{Tool-Name}` (formatted for display)
- **Examples**:
  - ID: `example_server:get_time`
  - Name: `Example-Server:Get-Time`

## Agent Protocol

### Base Agent Interface

All agents extend `BaseAgent` and implement the `execute()` method:

```python
from .base_agent import BaseAgent
from typing import Dict, Any

class FlightSearchAgent(BaseAgent):
    """Specialized agent for flight search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute agent logic

        Args:
            query (str): Natural language query from user

        Returns:
            Dict[str, Any]: Standardized response with tool, result, success, timestamp
        """
        pass
```

### Agent Response Contract

All agents must return a standardized response dictionary:

```python
{
    "tool": str,              # Tool ID (e.g., "flight_search")
    "success": bool,          # Execution status
    "parameters": dict,       # Extracted parameters (optional)
    "result": dict,           # Tool execution results (optional)
    "error": str,             # Error message if success=False (optional)
    "timestamp": str          # ISO format timestamp
}
```

### Example Agent Implementation

```python
class FlightSearchAgent(BaseAgent):
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # 1. Extract parameters using LLM
            params = self._extract_parameters(query)

            # 2. Execute tool
            tool_result = self._execute_tool_script("flight_search", params)

            # 3. Analyze results with LLM
            analysis = self._analyze_results(tool_result, query)

            # 4. Return standardized response
            return {
                "tool": "flight_search",
                "success": True,
                "parameters": params,
                "result": {
                    "llm_analysis": analysis,
                    "tool_data": tool_result.get("data"),
                    "flights": tool_result.get("flights")
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "tool": "flight_search",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
```

## Tool Protocol

### Tool Definition

Tools are registered in `multi_agent_system.py`:

```python
{
    "id": "flight_search",                    # Internal identifier
    "name": "Flight Search",                  # Display name
    "description": "Search for flight...",    # Full description
    "short_description": "search for...",     # Brief description for prompts
    "category": "general"                     # Tool category
}
```

### Tool Categories
- `general` - General-purpose tools (file search, web search, system info, etc.)
- `analytics` - Data analysis tools (cost analysis, stock analysis, forecasting, etc.)
- `mcp` - External MCP tools

### Tool Script Interface

Tool scripts must accept JSON parameters via command line and output JSON:

```python
#!/usr/bin/env python3
import sys
import json

def main():
    # Read parameters from command line
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
    else:
        params = {}

    # Execute tool logic
    result = execute_tool(params)

    # Output JSON result
    print(json.dumps(result))

def execute_tool(params: dict) -> dict:
    """
    Tool implementation

    Returns:
        dict: {
            "success": bool,
            "data": Any,
            "error": str (optional)
        }
    """
    return {
        "success": True,
        "data": "result data"
    }

if __name__ == "__main__":
    main()
```

### Tool Execution

Agents execute tools using `_execute_tool_script()`:

```python
tool_result = self._execute_tool_script("flight_search", {
    "origin": "San Francisco",
    "destination": "Tokyo",
    "departure_date": "2025-12-20"
})
```

## MCP Integration

### MCP Agent Gateway

The `MCPAgent` acts as a gateway to external MCP servers:

```python
# MCP tool execution
mcp_result = await mcp_agent.execute_mcp_tool(
    tool_name="get_time",
    parameters={"timezone": "UTC"}
)
```

### MCP Tool Discovery

MCP tools are discovered dynamically from configured servers:

```python
# Environment configuration
MCP_SERVERS=example_server
MCP_EXAMPLE_SERVER_URL=http://localhost:8000
MCP_EXAMPLE_SERVER_TOOLS=echo,get_time
MCP_EXAMPLE_SERVER_DESCRIPTION=Example MCP server
```

### MCP Tool Response

MCP tools return the same standardized response format:

```python
{
    "tool": "example_server:get_time",
    "success": True,
    "result": {"time": "2025-10-09T18:00:00Z"},
    "timestamp": "2025-10-09T18:00:00Z"
}
```

## Data Contracts

### Orchestrator Selection Protocol

The orchestrator uses LLM to select appropriate agents:

**Input**: User query + Available tools
**Output**: Ordered list of tool IDs

```python
# Prompt format
"""
Given this user query: "find flights from NYC to London"
Available tools:
- file_search: search for files and directories
- web_search: search the internet for information
- flight_search: search for flight information between cities

Respond with: flight_search
"""

# Expected response formats
"flight_search"                      # Single tool
"stock_analysis → forecast"          # Multiple tools with execution order
"NONE"                              # No tools needed
```

### Parameter Extraction

Agents use LLM to extract structured parameters from natural language:

```python
# Query: "find flights from San Francisco to Tokyo on Dec 20"
# Extracted parameters:
{
    "origin": "San Francisco",
    "destination": "Tokyo",
    "departure_date": "2025-12-20"
}
```

### Result Analysis

Agents use LLM to analyze tool results and generate insights:

```python
{
    "llm_analysis": str,        # Human-readable analysis
    "tool_data": str,           # Formatted data for chaining
    "results": list,            # Structured results array
    "query": dict,              # Query metadata
    "results_count": int        # Number of results
}
```

### Frontend Data Contract

Results displayed in frontend follow specific formats:

#### Generic Results
```typescript
{
  tool: string;              // Tool ID
  success: boolean;
  result: {
    llm_analysis?: string;   // Display as markdown
    tool_data?: string;      // Display as formatted text
    results?: any[];         // Display as list
  }
}
```

#### Specialized Viewers

**Flight Search**:
```typescript
{
  tool: "flight_search",
  result: {
    flights: {
      outbound: Flight[],
      return: Flight[]
    },
    query: {
      origin: string,
      destination: string,
      departure_date: string
    },
    results_count: number
  }
}
```

**Web Search**:
```typescript
{
  tool: "web_search",
  result: {
    results: SearchResult[],
    query: string,
    results_count: number
  }
}
```

**Visualization**:
```typescript
{
  tool: "visualization",
  result: {
    chart_url: string,
    chart_type: string,
    data_summary: string
  }
}
```

## Best Practices

### Agent Development

1. **Inherit from BaseAgent**: All agents should extend the base agent class
2. **Implement execute()**: Follow the standardized execute method signature
3. **Return standard format**: Always return the required response dictionary
4. **Handle errors gracefully**: Catch exceptions and return error responses
5. **Use LLM for extraction**: Leverage LLM for parameter extraction from natural language
6. **Provide analysis**: Include LLM-generated insights in results

### Tool Development

1. **JSON I/O**: Accept JSON input, output JSON results
2. **Error handling**: Return success/error status in response
3. **Validation**: Validate required parameters before execution
4. **Documentation**: Include docstrings and parameter descriptions
5. **Logging**: Use logging for debugging and monitoring

### Naming Consistency

1. **Tool ID**: Always use `snake_case` for internal identifiers
2. **Display Names**: Use Title Case for user-facing names
3. **File Names**: Match tool ID with `_agent.py` suffix for agents
4. **Class Names**: Use PascalCase with Agent suffix
5. **MCP Tools**: Use colon separator for server:tool format

## Version History

- **v1.0** (2025-10-09): Initial protocol specification
  - Defined agent and tool naming conventions
  - Established response contracts
  - Documented MCP integration patterns
  - Specified data contracts for frontend integration
