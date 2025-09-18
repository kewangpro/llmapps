"""
System Info Agent - Specialized agent for system information operations
"""

import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class SystemInfoAgent(BaseAgent):
    """Specialized agent for system information operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute system info with intelligent metric selection"""
        try:
            logger.info(f"💻 SystemInfoAgent analyzing: '{query}'")
            prompt = f"""Determine what system metric to check for: "{query}"
Choose the best metric:
- "overview" for general system information
- "cpu" for CPU usage and details
- "memory" for RAM usage
- "disk" for storage information
- "network" for network details
Respond with just the metric name."""
            metric = self.llm.call(prompt).strip().lower()
            logger.info(f"💻 Selected metric: '{metric}'")

            # Validate metric
            valid_metrics = ["overview", "cpu", "memory", "disk", "network"]
            if metric not in valid_metrics:
                metric = "overview"
                logger.info(f"💻 Invalid metric, defaulting to: '{metric}'")

            params = {"metric": metric}
            logger.info(f"💻 Executing system_info tool with metric: {metric}")

            # Execute system info tool
            result = self._execute_tool_script("system_info", params)
            logger.info(f"💻 System info retrieved: {len(str(result))} characters")

            return {
                "agent": "SystemInfoAgent",
                "tool": "system_info",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"💻 SystemInfoAgent error: {str(e)}")
            return {
                "agent": "SystemInfoAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }