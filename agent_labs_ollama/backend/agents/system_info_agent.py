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
- "overview" for general system information, OS version, hardware specs, overall status
- "cpu" for CPU usage and performance details only
- "memory" for RAM usage only
- "disk" for storage information only
- "network" for network configuration only

For queries about "system info", "system specs", "what is my system", or checking OS versions, use "overview".
Respond with just the metric name."""
            metric = self.llm.call(prompt).strip().lower()
            logger.info(f"💻 Selected metric: '{metric}'")

            # Validate metric and apply intelligent defaults
            valid_metrics = ["overview", "cpu", "memory", "disk", "network"]
            if metric not in valid_metrics:
                metric = "overview"
                logger.info(f"💻 Invalid metric, defaulting to: '{metric}'")

            # For queries about general system info or OS version, force overview
            general_queries = ["system info", "system", "my system", "what is my", "find my system", "os version", "version"]
            if any(phrase in query.lower() for phrase in general_queries):
                metric = "overview"
                logger.info(f"💻 General system query detected, using: '{metric}'")

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