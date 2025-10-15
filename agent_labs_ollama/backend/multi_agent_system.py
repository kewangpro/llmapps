"""
Multi-Agent System for Agent Labs
Orchestrator pattern with specialized sub-agents for each tool
"""

import logging
import asyncio
from typing import Dict, List, Any
from agents import OrchestratorAgent
from agents.mcp_agent import MCPToolDiscovery
from llm_config import llm_config

# Configure logging for multi-agent system with timestamps
import os
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MultiAgentSystem")


class MultiAgentSystem:
    """Main multi-agent system interface"""

    def __init__(self):
        # LLM configuration will be set by frontend model selection
        self.orchestrator = OrchestratorAgent()

    async def execute_query_with_callback(self, query: str, selected_tools: List[str] = None, attached_file: Dict = None, callback=None) -> Dict[str, Any]:
        """Execute a query with real-time callback for messaging"""
        return await self.orchestrator.execute_with_callback(query, selected_tools, attached_file, callback)

    @staticmethod
    def get_available_tools() -> List[Dict[str, str]]:
        """Get list of available tools including MCP tools"""
        # Built-in tools (id = name for consistency)
        # Order: general -> search -> analytics
        tools = [
            {"id": "system_info", "name": "System Info", "description": "Get system information including CPU, memory, disk usage", "short_description": "get system information (CPU, memory, disk, network)", "category": "general"},
            {"id": "presentation", "name": "Presentation", "description": "Generate PowerPoint presentations from text or files", "short_description": "generate PowerPoint presentations from text or files", "category": "general"},
            {"id": "visualization", "name": "Visualization", "description": "Create charts and visualizations from data", "short_description": "create charts and visualizations from data", "category": "general"},
            {"id": "file_search", "name": "File Search", "description": "Search for files and directories in the filesystem", "short_description": "search for files and directories", "category": "search"},
            {"id": "web_search", "name": "Web Search", "description": "Search the web for current information and news", "short_description": "search the internet for information", "category": "search"},
            {"id": "flight_search", "name": "Flight Search", "description": "Search for flights between cities", "short_description": "search for flights between cities", "category": "search"},
            {"id": "hotel_search", "name": "Hotel Search", "description": "Search for hotel accommodations", "short_description": "search for hotel accommodations", "category": "search"},
            {"id": "cost_analysis", "name": "Cost Analysis", "description": "Analyze cost data, COGS, and spending patterns", "short_description": "analyze cost data, COGS, and spending patterns", "category": "analytics"},
            {"id": "data_processing", "name": "Data Processing", "description": "Process, analyze, and transform data", "short_description": "process, analyze, or transform data", "category": "analytics"},
            {"id": "image_analysis", "name": "Image Analysis", "description": "Analyze image files for content, objects, text, and metadata", "short_description": "analyze image files for content, text, and metadata", "category": "analytics"},
            {"id": "stock_analysis", "name": "Stock Analysis", "description": "Analyze stock market data and performance using Yahoo Finance", "short_description": "analyze stock market data and performance using Yahoo Finance", "category": "analytics"},
            {"id": "forecast", "name": "Forecast", "description": "Predict future values using LSTM neural networks for time series data", "short_description": "forecast future trends using LSTM neural networks", "category": "analytics"}
        ]

        # Add MCP tools
        try:
            mcp_discovery = MCPToolDiscovery()
            # First check if any servers are configured
            if not mcp_discovery.servers:
                logger.info("No MCP servers configured")
                return tools

            # Run async discovery in sync context
            mcp_tools = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create new thread/event loop
                    import concurrent.futures
                    import threading

                    def run_discovery():
                        return asyncio.run(mcp_discovery.discover_tools())

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_discovery)
                        mcp_tools = future.result(timeout=10)  # 10 second timeout
                else:
                    mcp_tools = loop.run_until_complete(mcp_discovery.discover_tools())
            except RuntimeError:
                # No event loop exists, create one
                mcp_tools = asyncio.run(mcp_discovery.discover_tools())

            if mcp_tools:
                for server_name, server_tools in mcp_tools.items():
                    for tool in server_tools:
                        tool_name = tool.get("name", "")
                        if tool_name:
                            # Format server name for display (replace underscores with hyphens, capitalize)
                            display_server_name = server_name.replace("_", "-").title()
                            display_tool_name = tool_name.capitalize()
                            display_name = f"{display_server_name}:{display_tool_name}"

                            # Use server:tool format for ID, formatted display name for name
                            tool_id = f"{server_name}:{tool_name}"

                            tools.append({
                                "id": tool_id,
                                "name": display_name,
                                "description": tool.get("description", f"MCP tool: {tool_name}"),
                                "short_description": tool.get("description", f"MCP tool: {tool_name}"),
                                "category": "mcp"
                            })

                logger.info(f"Added {sum(len(server_tools) for server_tools in mcp_tools.values())} MCP tools")
            else:
                logger.info("No MCP tools discovered")

        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")

        return tools