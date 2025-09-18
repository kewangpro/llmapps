"""
Multi-Agent System for Agent Labs
Orchestrator pattern with specialized sub-agents for each tool
"""

import logging
from typing import Dict, List, Any
from agents import OrchestratorAgent

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

    def __init__(self, model: str = "gemma3:latest"):
        self.orchestrator = OrchestratorAgent(model)

    def execute_query(self, query: str, selected_tools: List[str] = None, attached_file: Dict = None) -> Dict[str, Any]:
        """Execute a query using the multi-agent system"""
        return self.orchestrator.execute(query, selected_tools, attached_file)

    async def execute_query_with_callback(self, query: str, selected_tools: List[str] = None, attached_file: Dict = None, callback=None) -> Dict[str, Any]:
        """Execute a query with real-time callback for messaging"""
        return await self.orchestrator.execute_with_callback(query, selected_tools, attached_file, callback)

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {"name": "file_search", "description": "Search for files and directories in the filesystem"},
            {"name": "web_search", "description": "Search the web for current information and news"},
            {"name": "system_info", "description": "Get system information including CPU, memory, disk usage"},
            {"name": "code_analysis", "description": "Analyze code files for quality, security, and performance"},
            {"name": "data_processing", "description": "Process, analyze, and transform data"},
            {"name": "presentation", "description": "Generate PowerPoint presentations from text or files"},
            {"name": "image_analysis", "description": "Analyze image files for content, objects, text, and metadata"}
        ]