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

    def execute_query(self, query: str, selected_tools: List[str] = None, attached_file: Dict = None) -> Dict[str, Any]:
        """Execute a query using the multi-agent system"""
        return self.orchestrator.execute(query, selected_tools, attached_file)

    async def execute_query_with_callback(self, query: str, selected_tools: List[str] = None, attached_file: Dict = None, callback=None) -> Dict[str, Any]:
        """Execute a query with real-time callback for messaging"""
        return await self.orchestrator.execute_with_callback(query, selected_tools, attached_file, callback)

    @staticmethod
    def get_available_tools() -> List[Dict[str, str]]:
        """Get list of available tools including MCP tools"""
        # Built-in tools
        tools = [
            {"name": "file_search", "description": "Search for files and directories in the filesystem", "short_description": "search for files and directories", "category": "general"},
            {"name": "web_search", "description": "Search the web for current information and news", "short_description": "search the internet for information", "category": "general"},
            {"name": "system_info", "description": "Get system information including CPU, memory, disk usage", "short_description": "get system information (CPU, memory, disk, network)", "category": "general"},
            {"name": "presentation", "description": "Generate PowerPoint presentations from text or files", "short_description": "generate PowerPoint presentations from text or files", "category": "general"},
            {"name": "cost_analysis", "description": "Analyze cost data, COGS, and spending patterns", "short_description": "analyze cost data, COGS, and spending patterns", "category": "analytics"},
            {"name": "data_processing", "description": "Process, analyze, and transform data", "short_description": "process, analyze, or transform data", "category": "analytics"},
            {"name": "image_analysis", "description": "Analyze image files for content, objects, text, and metadata", "short_description": "analyze image files for content, text, and metadata", "category": "analytics"},
            {"name": "stock_analysis", "description": "Analyze stock market data and performance using Yahoo Finance", "short_description": "analyze stock market data and performance using Yahoo Finance", "category": "analytics"},
            {"name": "visualization", "description": "Create charts and visualizations from data", "short_description": "create charts and visualizations from data", "category": "general"},
            {"name": "forecast", "description": "Predict future values using LSTM neural networks for time series data", "short_description": "forecast future trends using LSTM neural networks", "category": "analytics"}
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
                            tools.append({
                                "name": f"mcp_{tool_name}",
                                "description": tool.get("description", f"MCP tool: {tool_name}"),
                                "short_description": tool.get("description", f"MCP tool: {tool_name}"),
                                "category": "mcp",
                                "server": server_name
                            })

                logger.info(f"Added {sum(len(server_tools) for server_tools in mcp_tools.values())} MCP tools")
            else:
                logger.info("No MCP tools discovered")

        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")

        return tools