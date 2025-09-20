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

            # Use LLM to analyze user intent and extract analysis type
            analysis_intent = self._analyze_intent_with_llm(query)
            logger.info(f"📊 Analysis intent: {analysis_intent}")

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
                params = {
                    "file_path": file_path,
                    "query": clean_query,
                    "analysis_intent": analysis_intent
                }
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

    def _analyze_intent_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze user intent for cost analysis"""
        prompt = f"""Analyze this cost analysis query and determine what type of analysis and chart is needed.

Query: "{query}"

Determine:
1. What dimension to analyze: business_unit, aws_product, service_group, or monthly_totals
2. What time period: per_month, quarterly, yearly, or total
3. What chart type: line, bar, or pie
4. What should be the primary focus

Guidelines:
- "business unit" or "per business unit" → business_unit
- "aws product" or "per aws product" → aws_product
- "service group" or "per service group" → service_group
- "per month" or "monthly" or "over time" → monthly_totals if no other dimension specified
- "trends over time" → line chart
- "comparison" → bar chart
- "breakdown" or "distribution" → pie chart

RESPOND WITH VALID JSON ONLY:
{{
    "dimension": "business_unit|aws_product|service_group|monthly_totals",
    "time_period": "per_month|quarterly|yearly|total",
    "chart_type": "line|bar|pie",
    "focus": "brief description of what to analyze"
}}"""

        try:
            response = self.llm.call(prompt)
            logger.info(f"📊 Raw intent analysis: '{response}'")

            # Clean up and extract JSON
            response_cleaned = response.strip()
            json_start = response_cleaned.find('{')
            json_end = response_cleaned.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_part = response_cleaned[json_start:json_end]
                return json.loads(json_part)
            else:
                logger.warning("📊 Failed to parse LLM intent analysis, using defaults")
                return {
                    "dimension": "business_unit",
                    "time_period": "per_month",
                    "chart_type": "line",
                    "focus": "general cost analysis"
                }
        except Exception as e:
            logger.error(f"📊 Error in intent analysis: {str(e)}")
            return {
                "dimension": "business_unit",
                "time_period": "per_month",
                "chart_type": "line",
                "focus": "general cost analysis"
            }
