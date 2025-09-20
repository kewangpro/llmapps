"""
Cost Analysis Agent - Specialized agent for cost analysis operations
"""

import json
import logging
from .base_agent import BaseAgent
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class CostAnalysisAgent(BaseAgent):
    """Specialized agent for cost analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute cost analysis with intelligent parameter extraction"""
        try:
            logger.info(f"📊 CostAnalysisAgent analyzing: '{query}'")
            # Use default COGS file unless a different file is attached
            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Using attached file: {file_path}")
            else:
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                file_path = os.path.join(project_root, "data", "Any_COGS_1-8.csv")
                clean_query = query
                logger.info(f"📊 Using default COGS file: {file_path}")
            try:
                params = {"file_path": file_path, "query": clean_query}
                result = self._execute_tool_script("cost_analysis", params)
                return {
                    "agent": "CostAnalysisAgent",
                    "tool": "cost_analysis",
                    "parameters": params,
                    "result": result,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"📊 Failed to analyze COGS data: {str(e)}")
                return {
                    "agent": "CostAnalysisAgent",
                    "success": False,
                    "error": f"Failed to analyze COGS data: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"📊 CostAnalysisAgent error: {str(e)}")
            return {
                "agent": "CostAnalysisAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
