"""
Web Search Agent - Specialized agent for web search operations
"""

import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class WebSearchAgent(BaseAgent):
    """Specialized agent for web search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute web search with query optimization"""
        try:
            logger.info(f"🌐 WebSearchAgent analyzing: '{query}'")
            prompt = f"""Optimize this search query for web search: "{query}"
Create a clear, focused search query that will get the best results.
Respond with just the optimized query, no additional text."""
            optimized_query = self.llm.call(prompt).strip()

            # Remove any surrounding quotes that the LLM might have added
            if optimized_query.startswith('"') and optimized_query.endswith('"'):
                optimized_query = optimized_query[1:-1]
            if optimized_query.startswith("'") and optimized_query.endswith("'"):
                optimized_query = optimized_query[1:-1]

            logger.info(f"🌐 Query optimization: '{query}' → '{optimized_query}'")

            # Use original query if optimization fails
            if not optimized_query or len(optimized_query) > 200:
                optimized_query = query
                logger.info(f"🌐 Using original query: '{optimized_query}'")

            params = {"query": optimized_query}
            logger.info(f"🌐 Executing web_search tool with: {optimized_query}")

            # Execute web search tool
            result = self._execute_tool_script("web_search", params)
            logger.info(f"🌐 Web search completed: {len(str(result))} characters")

            return {
                "agent": "WebSearchAgent",
                "tool": "web_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"🌐 WebSearchAgent error: {str(e)}")
            return {
                "agent": "WebSearchAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }