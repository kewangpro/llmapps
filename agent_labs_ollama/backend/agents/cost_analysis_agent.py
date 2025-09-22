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

            # Step 1: Call the cost analysis tool to process the file (no LLM, just data processing)
            try:
                tool_params = {"file_path": file_path}
                tool_result = self._execute_tool_script("cost_analysis", tool_params)

                if not tool_result.get("success", False):
                    return {
                        "agent": "CostAnalysisAgent",
                        "success": False,
                        "error": f"Cost analysis tool failed: {tool_result.get('error', 'Unknown error')}",
                        "timestamp": datetime.now().isoformat()
                    }

                logger.info(f"📊 Tool processed COGS data successfully")

                # Step 2: Use LLM to analyze the cost data returned by the tool
                llm_analysis = self._analyze_cost_data_with_llm(tool_result, clean_query)

                # Format tool_data as CSV for downstream agents
                formatted_tool_data = self._format_tool_data_as_csv(tool_result, clean_query)

                result = {
                    "tool_data": formatted_tool_data,  # CSV formatted data for chaining
                    "llm_analysis": llm_analysis       # LLM insights
                }

                return {
                    "agent": "CostAnalysisAgent",
                    "tool": "cost_analysis",
                    "parameters": {"file_path": file_path},
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

    def _analyze_cost_data_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze the cost data returned by the tool"""
        try:
            # Extract key information from tool result
            data_summary = tool_result.get("data_summary", {})
            cost_insights = tool_result.get("cost_insights", {})
            recommendations = tool_result.get("recommendations", [])
            monthly_data = tool_result.get("monthly_data", {})

            # Format the data for LLM analysis
            analysis_prompt = f"""Analyze this COGS cost data and provide insights for the user query: "{original_query}"

Data Summary:
- Total rows: {data_summary.get('total_rows', 0)}
- Total columns: {data_summary.get('total_columns', 0)}
- Date range: {data_summary.get('date_range', {})}
- Columns: {data_summary.get('columns', [])}

Cost Insights:
{json.dumps(cost_insights, indent=2)}

Tool Recommendations:
{recommendations}

Monthly Data Available:
- Total months: {len(monthly_data.get('months', []))}
- Business units: {len(monthly_data.get('business_unit_costs', {}))}
- AWS products: {len(monthly_data.get('aws_product_costs', {}))}
- Service groups: {len(monthly_data.get('service_group_costs', {}))}

Please provide a comprehensive cost analysis including:
1. Overview of spending patterns and trends
2. Key insights about cost drivers and business units
3. Analysis of any notable patterns or concerns
4. Strategic recommendations for cost optimization
5. Answer to the specific user query: "{original_query}"

Focus on actionable insights and business value."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📊 Generated LLM analysis for COGS data")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"📊 Error in LLM analysis: {str(e)}")
            return f"Cost analysis completed but LLM analysis failed: {str(e)}"

    def _format_tool_data_as_csv(self, tool_result: Dict[str, Any], query: str) -> str:
        """Format tool result as CSV string for downstream agents"""
        try:
            monthly_data = tool_result.get("monthly_data", {})
            if not monthly_data or not monthly_data.get("months"):
                return "No monthly cost data available"

            months = monthly_data.get("months", [])

            # Determine format based on query - default to monthly totals
            if "business unit" in query.lower():
                # Business unit trends format
                business_unit_costs = monthly_data.get("business_unit_costs", {})
                if business_unit_costs:
                    csv_data = "month,business_unit,cost\n"
                    for unit, unit_costs in business_unit_costs.items():
                        for month in months:
                            cost = unit_costs.get(month, 0)
                            csv_data += f"{month},{unit},{cost}\n"
                    logger.info(f"📊 Formatted business unit cost data as CSV ({len(business_unit_costs)} units)")
                    return csv_data
                else:
                    return "No business unit cost data available"

            elif "aws product" in query.lower():
                # AWS product trends format
                aws_product_costs = monthly_data.get("aws_product_costs", {})
                if aws_product_costs:
                    csv_data = "month,aws_product,cost\n"
                    for product, product_costs in aws_product_costs.items():
                        for month in months:
                            cost = product_costs.get(month, 0)
                            csv_data += f"{month},{product},{cost}\n"
                    logger.info(f"📊 Formatted AWS product cost data as CSV ({len(aws_product_costs)} products)")
                    return csv_data
                else:
                    return "No AWS product cost data available"

            elif "service group" in query.lower():
                # Service group trends format
                service_group_costs = monthly_data.get("service_group_costs", {})
                if service_group_costs:
                    csv_data = "month,service_group,cost\n"
                    for group, group_costs in service_group_costs.items():
                        for month in months:
                            cost = group_costs.get(month, 0)
                            csv_data += f"{month},{group},{cost}\n"
                    logger.info(f"📊 Formatted service group cost data as CSV ({len(service_group_costs)} groups)")
                    return csv_data
                else:
                    return "No service group cost data available"

            else:
                # Default to monthly totals
                total_costs = monthly_data.get("total_costs", {})
                if total_costs:
                    csv_data = "month,total_cost\n"
                    for month in months:
                        cost = total_costs.get(month, 0)
                        csv_data += f"{month},{cost}\n"
                    logger.info(f"📊 Formatted monthly total cost data as CSV ({len(months)} months)")
                    return csv_data
                else:
                    return "No monthly cost data available"

        except Exception as e:
            logger.error(f"📊 Failed to format tool data as CSV: {str(e)}")
            return f"Error formatting cost data: {str(e)}"
