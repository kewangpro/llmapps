"""
Code Analysis Agent - Specialized agent for code analysis operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class CodeAnalysisAgent(BaseAgent):
    """Specialized agent for code analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute code analysis with intelligent parameter extraction"""
        try:
            prompt = f"""Extract code analysis parameters from: "{query}"
Determine:
1. File path (if mentioned, otherwise use ".")
2. Analysis type: "general", "security", "performance", or "style"
Respond with JSON only:
{{"file_path": "path", "analysis_type": "type"}}"""
            response = self.llm.call(prompt)
            try:
                params = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback parameters
                params = {"file_path": ".", "analysis_type": "general"}

            # Execute code analysis tool
            result = self._execute_tool_script("code_analysis", params)

            return {
                "agent": "CodeAnalysisAgent",
                "tool": "code_analysis",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "agent": "CodeAnalysisAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }