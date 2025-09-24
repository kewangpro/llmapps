# MCP Category Integration Design

## Overview

This document outlines the design for integrating Model Context Protocol (MCP) servers as a new agent category in the Agent Labs orchestration system. The MCP category will enable dynamic tool discovery and execution from external MCP servers, extending the platform's capabilities beyond the current built-in tools.

## Background

Agent Labs currently uses a multi-agent orchestration pattern where:
- An `OrchestratorAgent` determines which tools to use based on user queries
- Specialized agents execute individual tools via Python scripts in the `tools/` directory
- Tool results are aggregated and presented to users through real-time WebSocket communication

The MCP integration introduces a new category that can dynamically connect to external MCP servers, discover their available tools, and execute them seamlessly within the existing architecture.

## Architecture

### Current System Architecture

```
User Query → OrchestratorAgent → Specialized Agents → Tool Scripts → Results
                ↓
        Real-time WebSocket Updates
```

### Proposed MCP Integration Architecture

```
User Query → OrchestratorAgent → MCP Agent → MCP Server → Tool Results
                ↓                   ↓
        Real-time WebSocket Updates  ↓
                                Dynamic Tool Discovery
```

## Core Components

### 1. MCP Agent (`mcp_agent.py`)

A new specialized agent that follows the existing `BaseAgent` pattern:

```python
class MCPAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.mcp_servers = self._load_mcp_config()

    def execute(self, query: str, tool_name: str = None) -> Dict[str, Any]:
        # Connect to appropriate MCP server
        # Execute the requested tool
        # Return standardized results
```

**Key Responsibilities:**
- Load MCP server configurations from environment variables
- Establish HTTP connections to MCP servers
- Translate between Agent Labs format and MCP protocol
- Handle tool discovery and execution
- Manage connection lifecycle and error handling

### 2. MCP Configuration in `.env`

Extend the existing `.env` file to include MCP server configurations:

```bash
# MCP Server Configurations
MCP_SERVERS=example-server,search-server,data-server

# Example Server Configuration
MCP_EXAMPLE_URL=http://localhost:8000
MCP_EXAMPLE_TOOLS=echo,get_time
MCP_EXAMPLE_DESCRIPTION=Basic MCP server for testing

# Search Server Configuration
MCP_SEARCH_URL=http://localhost:8001
MCP_SEARCH_TOOLS=web_search,search_web
MCP_SEARCH_DESCRIPTION=Web search MCP server

# Data Server Configuration
MCP_DATA_URL=http://localhost:8002
MCP_DATA_TOOLS=analyze_data,process_csv
MCP_DATA_DESCRIPTION=Data processing MCP server
```

### 3. MCP Tool Discovery Service

A service class to manage MCP server connections and tool discovery:

```python
class MCPToolDiscovery:
    def __init__(self):
        self.servers = {}
        self.available_tools = {}

    def discover_tools(self) -> Dict[str, List[Dict]]:
        # Connect to all configured MCP servers
        # Call /list_tools endpoint
        # Cache available tools with metadata

    def get_server_for_tool(self, tool_name: str) -> str:
        # Return which server hosts a specific tool
```

### 4. Integration with Multi-Agent System

Update `MultiAgentSystem.get_available_tools()` to include MCP tools:

```python
@staticmethod
def get_available_tools() -> List[Dict[str, str]]:
    tools = [
        # ... existing tools ...
    ]

    # Add MCP tools
    mcp_discovery = MCPToolDiscovery()
    mcp_tools = mcp_discovery.discover_tools()

    for server_name, server_tools in mcp_tools.items():
        for tool in server_tools:
            tools.append({
                "name": f"mcp_{tool['name']}",
                "description": tool['description'],
                "short_description": tool['description'],
                "category": "mcp",
                "server": server_name
            })

    return tools
```

## MCP Protocol Implementation

### Request Flow

1. **Tool Discovery** (on system startup/refresh):
   ```
   GET /initialize → MCP Server Capabilities
   GET /list_tools → Available Tools
   ```

2. **Tool Execution** (per user request):
   ```
   POST /call_tool → Tool Results
   ```

### Message Format Translation

**From Agent Labs to MCP:**
```python
# Agent Labs internal format
{
    "tool": "search_web",
    "parameters": {"query": "machine learning", "num_results": 5}
}

# MCP format
{
    "name": "search_web",
    "arguments": {"query": "machine learning", "num_results": 5}
}
```

**From MCP to Agent Labs:**
```python
# MCP response format
{
    "result": {
        "content": [{"type": "text", "text": "search results..."}]
    }
}

# Agent Labs internal format
{
    "tool": "search_web",
    "success": True,
    "result": {"content": "search results..."},
    "server": "search-server"
}
```

## Error Handling and Resilience

### Connection Management
- Connection pooling for MCP servers
- Automatic retry with exponential backoff
- Circuit breaker pattern for unhealthy servers
- Graceful degradation when servers are unavailable

### Error Types
- **Server Unavailable**: Remove tools from available list, notify user
- **Tool Execution Error**: Return structured error to orchestrator
- **Timeout**: Configurable timeout per server, fallback behavior
- **Authentication Error**: Server-specific auth handling

## Security Considerations

### Network Security
- HTTPS enforcement for production MCP servers
- Certificate validation for secure connections
- Network isolation and firewall rules

### Authentication
- Support for API key authentication in MCP requests
- Secure storage of credentials in environment variables
- Per-server authentication configuration

### Input Validation
- Sanitize all parameters before sending to MCP servers
- Validate MCP server responses before processing
- Rate limiting for MCP server requests

## Configuration Management

### Environment Variable Schema
```bash
# Required for each MCP server
MCP_{SERVER_NAME}_URL=server_endpoint
MCP_{SERVER_NAME}_TOOLS=comma_separated_tool_list
MCP_{SERVER_NAME}_DESCRIPTION=human_readable_description

# Optional authentication
MCP_{SERVER_NAME}_API_KEY=authentication_key
MCP_{SERVER_NAME}_AUTH_TYPE=bearer|api_key|none

# Optional performance settings
MCP_{SERVER_NAME}_TIMEOUT=30
MCP_{SERVER_NAME}_MAX_RETRIES=3
```

### Dynamic Configuration
- Hot-reloading of MCP server configurations
- Runtime addition/removal of servers via API endpoints
- Health checks and automatic discovery refresh

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `MCPAgent` base class following existing patterns
- [ ] Implement MCP protocol client with HTTP requests
- [ ] Add MCP configuration loading from environment
- [ ] Create basic tool discovery mechanism

### Phase 2: Integration
- [ ] Update `MultiAgentSystem` to include MCP tools
- [ ] Integrate MCP agent with orchestrator decision logic
- [ ] Add MCP category to frontend tool selection
- [ ] Implement real-time WebSocket updates for MCP tools

### Phase 3: Advanced Features
- [ ] Connection pooling and error resilience
- [ ] Authentication support
- [ ] Health monitoring and metrics
- [ ] Dynamic server management API

### Phase 4: Production Readiness
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Comprehensive error handling
- [ ] Documentation and examples

## Testing Strategy

### Unit Tests
- MCP protocol message serialization/deserialization
- Tool discovery and caching logic
- Error handling for various failure scenarios
- Configuration loading and validation

### Integration Tests
- End-to-end tool execution with test MCP server
- WebSocket message flow with MCP tools
- Multi-server scenarios with tool conflicts
- Performance testing with concurrent requests

### Test MCP Server
Use the provided `mcp_server.py` as a reference implementation:
- Echo tool for basic connectivity testing
- Time tool for simple functionality validation
- Web search tool for complex parameter handling

## Monitoring and Observability

### Metrics
- MCP server response times
- Tool execution success/failure rates
- Server availability and health status
- User adoption of MCP tools vs. built-in tools

### Logging
- Structured logging for MCP requests/responses
- Error tracking with server context
- Performance metrics for tool discovery
- User interaction patterns with MCP tools

## Migration and Compatibility

### Backward Compatibility
- Existing built-in tools remain unchanged
- No breaking changes to current API endpoints
- Gradual migration path for tools to MCP servers

### Tool Naming
- Prefix MCP tools with `mcp_` to avoid conflicts
- Maintain tool metadata for proper categorization
- Support for tool aliases and migration helpers

## Future Enhancements

### Advanced Protocol Support
- WebSocket connections for real-time MCP communication
- Streaming responses for long-running tools
- Resource management (files, databases, etc.)

### Tool Composition
- Ability to chain MCP tools together
- Cross-server tool workflows
- Dependency management between tools

### Management Interface
- Web UI for MCP server management
- Tool usage analytics and optimization
- Server health monitoring dashboard

## Conclusion

The MCP category integration provides a scalable way to extend Agent Labs capabilities through external tool providers. By following the existing architecture patterns and maintaining compatibility with current functionality, this design enables seamless adoption of MCP servers while preserving the user experience and system reliability.

The modular design allows for incremental implementation and testing, ensuring stability throughout the development process. The configuration-driven approach provides flexibility for different deployment scenarios while maintaining security and performance standards.