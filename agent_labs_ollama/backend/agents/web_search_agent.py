"""
Web Search Agent - Specialized agent for web search operations
"""

import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("WebSearchAgent")


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
            tool_result = self._execute_tool_script("web_search", params)
            logger.info(f"🌐 Web search completed: {len(str(tool_result))} characters")

            if not tool_result.get("success", False):
                return {
                    "tool": "web_search",
                    "success": False,
                    "error": f"Web search tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Use LLM to analyze the search results
            llm_analysis = self._analyze_search_results_with_llm(tool_result, query)

            # Format for downstream agents
            formatted_tool_data = self._format_tool_data(tool_result)

            result = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis       # LLM insights
            }

            return {
                "tool": "web_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"🌐 WebSearchAgent error: {str(e)}")
            return {
                "tool": "web_search",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_search_results_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze web search results and provide insights"""
        try:
            # Extract relevant information from tool result
            search_results = tool_result.get("results", [])
            query = tool_result.get("query", original_query)

            analysis_prompt = f"""Analyze these web search results and provide insights for the user query: "{original_query}"

Search Query Used: {query}
Number of Results: {len(search_results)}

Search Results:
{self._format_search_results_for_analysis(search_results)}

Please provide:
1. Key findings from the search results
2. Relevant information that answers the user's query
3. Summary of the most important points
4. Any recommendations or next steps if applicable

Format your response with:
- Clear section headers
- Important details highlighted
- Organized layout that's easy to read

Focus on the information most relevant to the user's question."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"🌐 Generated LLM analysis for web search results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"🌐 Error in LLM analysis: {str(e)}")
            return f"Web search completed but LLM analysis failed: {str(e)}"

    def _format_search_results_for_analysis(self, results: list) -> str:
        """Format search results for LLM analysis"""
        try:
            formatted = ""
            for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet")
                url = result.get("url", "No URL")
                formatted += f"{i}. {title}\n   {snippet}\n   Source: {url}\n\n"
            return formatted
        except Exception as e:
            logger.error(f"🌐 Error formatting search results: {str(e)}")
            return "Error formatting search results"

    def _format_tool_data(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as text for downstream agents"""
        try:
            search_results = tool_result.get("results", [])
            query = tool_result.get("query", "")

            if not search_results:
                return f"Search query: {query}\nNo results found"

            formatted_text = f"Search query: {query}\nResults count: {len(search_results)}\n\n"

            for i, result in enumerate(search_results[:3], 1):  # Top 3 for downstream
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet")
                url = result.get("url", "No URL")
                formatted_text += f"{i}. {title}\n{snippet}\nURL: {url}\n\n"

            return formatted_text
        except Exception as e:
            logger.error(f"🌐 Error formatting tool data: {str(e)}")
            return f"Error formatting search data: {str(e)}"