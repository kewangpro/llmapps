"""
Presentation Agent - Specialized agent for generating PowerPoint presentations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class PresentationAgent(BaseAgent):
    """Specialized agent for generating PowerPoint presentations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute presentation generation with intelligent parameter extraction"""
        try:
            logger.info(f"🎨 PresentationAgent analyzing: '{query}'")

            # Extract parameters from query using LLM
            prompt = f"""Extract presentation generation parameters from: "{query}"

Determine:
1. Input text content (the main content to convert to slides)
2. Presentation title (if mentioned, otherwise generate one)
3. Output filename (if mentioned, otherwise use default)

Respond with JSON only:
{{"input_text": "content", "title": "title", "output_filename": "filename.pptx"}}

If the query contains file content (after [Attached file:]), use that as input_text."""

            response = self.llm.call(prompt)
            logger.info(f"🎨 Parameter extraction: {response.strip()}")

            try:
                params = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback parameters
                params = {"input_text": query, "title": "Generated Presentation", "output_filename": "presentation.pptx"}

            # Execute presentation generation tool
            result = self._execute_tool_script("presentation", params)

            return {
                "agent": "PresentationAgent",
                "tool": "presentation",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🎨 PresentationAgent error: {str(e)}")
            return {
                "agent": "PresentationAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }