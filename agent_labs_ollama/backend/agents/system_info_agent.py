"""
System Info Agent - Specialized agent for system information operations
"""

import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("SystemInfoAgent")


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
            tool_result = self._execute_tool_script("system_info", params)
            logger.info(f"💻 System info retrieved: {len(str(tool_result))} characters")

            if not tool_result.get("success", False):
                return {
                    "agent": "SystemInfoAgent",
                    "success": False,
                    "error": f"System info tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Use LLM to analyze the system information
            llm_analysis = self._analyze_system_info_with_llm(tool_result, query)

            # Format for downstream agents
            formatted_tool_data = self._format_tool_data(tool_result)

            result = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis       # LLM insights
            }

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

    def _analyze_system_info_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze the system information"""
        try:
            # Extract system info from tool result
            system_data = tool_result.get("system_info", {})

            analysis_prompt = f"""Analyze this system information and provide insights for the user query: "{original_query}"

System Information:
{system_data}

Please provide a well-formatted analysis including:
1. Overview of the system configuration
2. Key hardware and software details
3. Current system status and performance insights
4. Answer to the specific user query: "{original_query}"

Format your response with:
- Clear section headers
- Important details highlighted
- Organized layout that's easy to read

Focus on the information most relevant to the user's question."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"💻 Generated LLM analysis for system info")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"💻 Error in LLM analysis: {str(e)}")
            return f"System information retrieved but LLM analysis failed: {str(e)}"

    def _format_tool_data(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as text for downstream agents"""
        try:
            system_data = tool_result.get("system_info", {})
            if isinstance(system_data, dict):
                # Convert dict to readable text format
                formatted_text = ""
                for key, value in system_data.items():
                    formatted_text += f"{key}: {value}\n"
                return formatted_text
            else:
                return str(system_data)
        except Exception as e:
            logger.error(f"💻 Error formatting tool data: {str(e)}")
            return f"Error formatting system data: {str(e)}"