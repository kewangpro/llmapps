"""
Visualization Agent - Specialized agent for creating charts and visualizations
"""

import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("VisualizationAgent")


class VisualizationAgent(BaseAgent):
    """Specialized agent for creating charts and visualizations from data"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute visualization with intelligent chart type and parameter selection"""
        try:
            logger.info(f"📊 VisualizationAgent analyzing: '{query}'")
            
            # Check for attached file path in query (added from orchestrator)
            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                # Remove the FILE_PATH marker from the query for chart type detection
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Found attached file: {file_path}")
                
                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    logger.info(f"📄 Read file content: {len(file_content)} characters")
                    
                    # Determine chart type and options from clean query
                    chart_params = self._determine_chart_parameters(clean_query, file_content)
                    
                    # Execute visualization tool
                    params = {
                        "data": file_content,
                        "chart_type": chart_params["chart_type"],
                        "options": chart_params["options"]
                    }
                    result = self._execute_tool_script("visualization", params)
                    
                    response = {
                        "agent": "VisualizationAgent",
                        "tool": "visualization",
                        "parameters": params,
                        "result": result,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"📄 Failed to read file {file_path}: {str(e)}")
                    return {
                        "agent": "VisualizationAgent",
                        "success": False,
                        "error": f"Failed to read file {file_path}: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Handle direct data input in query
            elif self._contains_data(query):
                logger.info("📊 Processing direct data input")
                
                # Extract data and chart requirements from query
                chart_params = self._extract_chart_parameters(query)
                
                params = {
                    "data": chart_params["data"],
                    "chart_type": chart_params["chart_type"],
                    "options": chart_params["options"]
                }
                
                # Execute visualization tool
                result = self._execute_tool_script("visualization", params)
                
                response = {
                    "agent": "VisualizationAgent",
                    "tool": "visualization", 
                    "parameters": params,
                    "result": result,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                return response
            
            else:
                # Query doesn't contain data or file - provide guidance
                return {
                    "agent": "VisualizationAgent",
                    "success": False,
                    "error": "No data provided. Please attach a data file or include data in your query.",
                    "available_chart_types": [
                        "line", "bar", "scatter", "pie", "histogram", 
                        "box", "heatmap", "area", "bubble", "treemap"
                    ],
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"📊 VisualizationAgent error: {str(e)}")
            return {
                "agent": "VisualizationAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _contains_data(self, query: str) -> bool:
        """Check if query contains structured data"""
        # Look for data indicators
        data_indicators = [
            '[{', '{"', 'csv', 'json', 'data:', 'values:',
            ',', '\t', 'column', 'row'
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in data_indicators)

    def _determine_chart_parameters(self, query: str, data: str) -> Dict[str, Any]:
        """Determine chart type and options from query and data analysis"""
        try:
            # Let LLM decide chart type based on query and data context - no keyword detection
            
            # Analyze the data to understand its structure
            data_preview = data[:500] if len(data) > 500 else data
            
            prompt = f"""You are a data visualization expert. Analyze the user's query and data to determine the BEST chart type and configuration.

USER QUERY: "{query}"

DATA STRUCTURE (first 500 characters):
{data_preview}

CHART TYPE SELECTION RULES:
- Cost/spending trends over time = LINE CHART (multiple lines for different categories)
- Stock price trends over time = LINE CHART (multiple lines for price and moving averages)
- Comparing categories/values = BAR CHART
- Distribution/parts of whole = PIE CHART
- Correlation between variables = SCATTER PLOT
- Statistical distribution = HISTOGRAM or BOX PLOT

MULTI-LINE CHART DETECTION:
- Time series data with categories (date,category,value format) → USE LINE CHART
- x-axis = time/date column, y-axis = value column, color = category column
- Examples: cost by business unit, stock price with moving averages, sales by region
- This automatically creates multiple lines (one per category)

Your task:
1. Determine the BEST chart type for this specific query and data
2. Find the right columns in the data
3. Create an appropriate title

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT:
{{
    "chart_type": "line|bar|pie|scatter|histogram|box|heatmap|area",
    "options": {{
        "title": "generic descriptive title (do not include specific entity names - use generic terms like 'by Business Unit', 'by Product', etc.)",
        "x_column": "exact_column_name_from_data",
        "y_column": "exact_column_name_from_data",
        "color_column": "exact_column_name_from_data_or_null"
    }}
}}

Use only column names that exist in the data. Use null if a column doesn't exist."""

            response = self.llm.call(prompt)
            logger.info(f"📊 Raw LLM response: '{response}'")
            
            # Clean up the response - sometimes LLM adds extra text
            response_cleaned = response.strip()
            
            # Try to extract JSON from the response
            json_start = response_cleaned.find('{')
            json_end = response_cleaned.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_part = response_cleaned[json_start:json_end]
                logger.info(f"📊 Extracted JSON: '{json_part}'")
                
                try:
                    result = json.loads(json_part)
                    # Trust the LLM's chart type decision
                    logger.info(f"📊 Successfully parsed chart parameters: {result}")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"📊 JSON decode error: {e}")
            
            # Fallback to bar chart if LLM response invalid
            logger.warning(f"📊 Failed to parse chart parameters, using default bar chart")
            return {
                "chart_type": "bar",
                "options": {"title": "Data Visualization"}
            }
                
        except Exception as e:
            logger.error(f"📊 Error determining chart parameters: {str(e)}")
            # Try to detect chart type even in error case
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ["line chart", "trends over time", "time series"]):
                chart_type = "line"
            else:
                chart_type = "bar"
            return {
                "chart_type": chart_type, 
                "options": {"title": "Data Visualization"}
            }

    def _extract_chart_parameters(self, query: str) -> Dict[str, Any]:
        """Extract chart parameters from query containing data"""
        try:
            # Check for DATA_START/DATA_END markers first
            if "DATA_START" in query and "DATA_END" in query:
                start_marker = query.find("DATA_START") + len("DATA_START")
                end_marker = query.find("DATA_END")
                data = query[start_marker:end_marker].strip()

                # Extract chart type from the instruction part
                instruction = query[:query.find("DATA_START")].lower()
                chart_type = "line"  # default
                if "line chart" in instruction or "line" in instruction:
                    chart_type = "line"
                elif "bar chart" in instruction or "bar" in instruction:
                    chart_type = "bar"
                elif "scatter" in instruction:
                    chart_type = "scatter"

                # Extract column information
                x_column = None
                y_column = None
                if "x-axis" in instruction and "date" in instruction:
                    x_column = "date"
                if "y-axis" in instruction and "price" in instruction:
                    y_column = "price"

                logger.info(f"📊 Extracted chart type: {chart_type}, data: {len(data)} chars")
                return {
                    "data": data,
                    "chart_type": chart_type,
                    "options": {
                        "title": "Stock Price Chart",
                        "x_column": x_column,
                        "y_column": y_column
                    }
                }

            # Fallback to LLM extraction for other formats
            prompt = f"""Extract chart parameters from this query that contains both data and visualization request.

Query: "{query}"

The query contains:
1. Data (JSON, CSV, or other format)
2. Visualization request

Extract and respond with JSON only:
{{
    "data": "extracted_data_string",
    "chart_type": "chart_type_name",
    "options": {{
        "title": "Chart Title",
        "x_column": "column_name_or_null",
        "y_column": "column_name_or_null"
    }}
}}

Available chart types: line, bar, scatter, pie, histogram, box, heatmap, area, bubble, treemap

If data format is ambiguous, assume CSV format."""

            response = self.llm.call(prompt)
            logger.info(f"📊 Parameter extraction: {response.strip()}")

            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback - try to extract data manually
                logger.warning("📊 Failed to parse parameters, attempting manual extraction")

                # Simple extraction - look for JSON-like patterns
                if '[{' in query or '{"' in query:
                    # Find JSON data
                    start = query.find('[{') if '[{' in query else query.find('{"')
                    if start != -1:
                        # Try to find the end of JSON
                        bracket_count = 0
                        end = start
                        for i, char in enumerate(query[start:], start):
                            if char in '[{':
                                bracket_count += 1
                            elif char in ']}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end = i + 1
                                    break

                        data = query[start:end]
                        return {
                            "data": data,
                            "chart_type": "bar",
                            "options": {"title": "Data Visualization"}
                        }

                # Fallback for other formats
                return {
                    "data": query,
                    "chart_type": "bar",
                    "options": {"title": "Data Visualization"}
                }

        except Exception as e:
            logger.error(f"📊 Error extracting chart parameters: {str(e)}")
            return {
                "data": query,
                "chart_type": "bar",
                "options": {"title": "Data Visualization"}
            }