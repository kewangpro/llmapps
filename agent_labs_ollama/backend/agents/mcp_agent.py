"""
MCP (Model Context Protocol) Agent
Handles communication with external MCP servers
"""

import json
import os
import httpx
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import logging

logger = logging.getLogger("MCPAgent")


class MCPToolDiscovery:
    """Service for discovering and managing MCP server tools"""

    def __init__(self):
        self.servers = {}
        self.available_tools = {}
        self._load_mcp_config()

    def _load_mcp_config(self):
        """Load MCP server configurations from environment variables"""
        # Get list of configured MCP servers
        servers_list = os.environ.get("MCP_SERVERS", "").strip()
        if not servers_list:
            logger.info("No MCP servers configured")
            return

        server_names = [name.strip() for name in servers_list.split(",") if name.strip()]

        for server_name in server_names:
            server_config = self._load_server_config(server_name)
            if server_config:
                self.servers[server_name] = server_config
                logger.info(f"Configured MCP server: {server_name} at {server_config['url']}")

    def _load_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration for a specific MCP server"""
        url_key = f"MCP_{server_name.upper()}_URL"
        tools_key = f"MCP_{server_name.upper()}_TOOLS"
        desc_key = f"MCP_{server_name.upper()}_DESCRIPTION"

        url = os.environ.get(url_key)
        if not url:
            logger.warning(f"No URL configured for MCP server {server_name} (missing {url_key})")
            return None

        tools = os.environ.get(tools_key, "").strip()
        description = os.environ.get(desc_key, f"MCP server: {server_name}")

        # Optional authentication
        api_key = os.environ.get(f"MCP_{server_name.upper()}_API_KEY")
        auth_type = os.environ.get(f"MCP_{server_name.upper()}_AUTH_TYPE", "none")

        # Optional performance settings
        timeout = int(os.environ.get(f"MCP_{server_name.upper()}_TIMEOUT", "30"))
        max_retries = int(os.environ.get(f"MCP_{server_name.upper()}_MAX_RETRIES", "3"))

        return {
            "url": url,
            "tools": [tool.strip() for tool in tools.split(",") if tool.strip()] if tools else [],
            "description": description,
            "api_key": api_key,
            "auth_type": auth_type,
            "timeout": timeout,
            "max_retries": max_retries
        }

    async def discover_tools(self) -> Dict[str, List[Dict]]:
        """Connect to all configured MCP servers and discover their tools"""
        all_tools = {}

        for server_name, server_config in self.servers.items():
            try:
                server_tools = await self._discover_server_tools(server_name, server_config)
                all_tools[server_name] = server_tools
                self.available_tools[server_name] = server_tools
                logger.info(f"Discovered {len(server_tools)} tools from {server_name}")
            except Exception as e:
                logger.error(f"Failed to discover tools from {server_name}: {e}")
                all_tools[server_name] = []

        return all_tools

    async def _discover_server_tools(self, server_name: str, server_config: Dict[str, Any]) -> List[Dict]:
        """Discover tools from a specific MCP server"""
        timeout = httpx.Timeout(server_config["timeout"])
        headers = {}

        # Add authentication if configured
        if server_config["auth_type"] == "bearer" and server_config["api_key"]:
            headers["Authorization"] = f"Bearer {server_config['api_key']}"
        elif server_config["auth_type"] == "api_key" and server_config["api_key"]:
            headers["X-API-Key"] = server_config["api_key"]

        async with httpx.AsyncClient(timeout=timeout) as client:
            # First, initialize the server
            try:
                init_response = await client.post(
                    f"{server_config['url']}/initialize",
                    json={"clientInfo": {"name": "agent-labs", "version": "1.0.0"}},
                    headers=headers
                )
                init_response.raise_for_status()
                logger.debug(f"Initialized MCP server {server_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {server_name}, trying direct tool discovery: {e}")

            # Discover tools
            tools_response = await client.post(
                f"{server_config['url']}/list_tools",
                json={},
                headers=headers
            )
            tools_response.raise_for_status()

            tools_data = tools_response.json()
            tools = tools_data.get("tools", [])

            # Filter tools if specific ones are configured
            if server_config["tools"]:
                tools = [tool for tool in tools if tool.get("name") in server_config["tools"]]

            return tools

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which server hosts a specific tool"""
        logger.info(f"🔍 Looking for tool '{tool_name}' in available_tools: {list(self.available_tools.keys())}")
        for server_name, tools in self.available_tools.items():
            logger.info(f"🔍 Server {server_name} has {len(tools)} tools: {[t.get('name') for t in tools]}")
            for tool in tools:
                if tool.get("name") == tool_name:
                    logger.info(f"✅ Found tool '{tool_name}' on server '{server_name}'")
                    return server_name
        logger.warning(f"❌ Tool '{tool_name}' not found on any server")
        return None


class MCPAgent(BaseAgent):
    """Agent for executing tools via MCP servers"""

    def __init__(self):
        super().__init__()
        self.discovery = MCPToolDiscovery()
        # Ensure tools are discovered on initialization
        try:
            import asyncio
            if not self.discovery.available_tools:
                # Try to discover tools synchronously if not already done
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule for later if we're in an async context
                        pass
                    else:
                        self.discovery.available_tools = loop.run_until_complete(self.discovery.discover_tools())
                except RuntimeError:
                    # No event loop exists, create one
                    self.discovery.available_tools = asyncio.run(self.discovery.discover_tools())
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools discovery: {e}")

    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on an MCP server"""
        try:
            # Ensure tools are discovered first
            if not self.discovery.available_tools:
                logger.info("🔍 MCP tools not yet discovered, discovering now...")
                await self.discovery.discover_tools()

            # Find which server hosts this tool
            server_name = self.discovery.get_server_for_tool(tool_name)
            if not server_name:
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' not found on any configured MCP server"
                }

            server_config = self.discovery.servers[server_name]

            # Execute the tool
            result = await self._execute_on_server(server_name, server_config, tool_name, parameters)
            return result

        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": f"MCP tool execution error: {str(e)}"
            }

    async def _execute_on_server(self, server_name: str, server_config: Dict[str, Any],
                                tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on a specific MCP server"""
        timeout = httpx.Timeout(server_config["timeout"])
        headers = {"Content-Type": "application/json"}

        # Add authentication if configured
        if server_config["auth_type"] == "bearer" and server_config["api_key"]:
            headers["Authorization"] = f"Bearer {server_config['api_key']}"
        elif server_config["auth_type"] == "api_key" and server_config["api_key"]:
            headers["X-API-Key"] = server_config["api_key"]

        # Prepare MCP request
        mcp_request = {
            "name": tool_name,
            "arguments": parameters
        }

        max_retries = server_config["max_retries"]
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{server_config['url']}/call_tool",
                        json=mcp_request,
                        headers=headers
                    )
                    response.raise_for_status()

                    result_data = response.json()

                    # Check for MCP-level errors
                    if "error" in result_data:
                        return {
                            "tool": tool_name,
                            "success": False,
                            "error": f"MCP server error: {result_data['error']}",
                            "server": server_name
                        }

                    # Extract result content
                    mcp_result = result_data.get("result", {})
                    content = mcp_result.get("content", [])

                    # Convert MCP content to Agent Labs format
                    result_text = ""
                    if content and isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                result_text += item.get("text", "")

                    return {
                        "tool": tool_name,
                        "success": True,
                        "result": {"content": result_text},
                        "server": server_name,
                        "mcp_result": mcp_result  # Include original MCP result for debugging
                    }

            except httpx.TimeoutException as e:
                last_error = f"Timeout after {server_config['timeout']} seconds"
                logger.warning(f"Timeout on attempt {attempt + 1} for {tool_name} on {server_name}")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.warning(f"HTTP error on attempt {attempt + 1} for {tool_name} on {server_name}: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1} for {tool_name} on {server_name}: {last_error}")

            if attempt < max_retries:
                # Exponential backoff
                import asyncio
                await asyncio.sleep(2 ** attempt)

        # All retries failed
        return {
            "tool": tool_name,
            "success": False,
            "error": f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
            "server": server_name
        }

    def execute(self, query: str, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute method for compatibility with BaseAgent (synchronous wrapper)"""
        import asyncio

        if tool_name and parameters:
            # Direct tool execution
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.execute_mcp_tool(tool_name, parameters))
            except RuntimeError:
                # If no event loop exists, create one
                return asyncio.run(self.execute_mcp_tool(tool_name, parameters))
        else:
            return {
                "success": False,
                "error": "MCP agent requires tool_name and parameters for execution"
            }