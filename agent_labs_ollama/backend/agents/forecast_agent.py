"""
Forecast Agent - Specialized agent for time series forecasting using LSTM
"""

import json
import logging
import os
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("ForecastAgent")


class ForecastAgent(BaseAgent):
    """Specialized agent for time series forecasting operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute forecast with intelligent parameter extraction and preprocessing"""
        try:
            logger.info(f"📈 ForecastAgent analyzing: '{query}'")

            # Check if file is attached (user upload) or file path is provided (orchestrator)
            if "[Attached file:" in query or "FILE_PATH:" in query:
                return self._process_file_forecast(query)
            else:
                return self._process_text_forecast(query)

        except Exception as e:
            logger.error(f"📈 ForecastAgent error: {str(e)}")
            return {
                "agent": "ForecastAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _process_file_forecast(self, query: str) -> Dict[str, Any]:
        """Process forecast request with attached file or file path"""
        try:
            # Handle FILE_PATH: pattern (from orchestrator)
            if "FILE_PATH:" in query:
                file_path_start = query.find("FILE_PATH:") + len("FILE_PATH:")
                file_path = query[file_path_start:].strip().split()[0]

                logger.info(f"📈 Reading file from orchestrator: {file_path}")

                # Read file content
                with open(file_path, 'r') as f:
                    file_content = f.read()

                file_name = os.path.basename(file_path)

            # Handle [Attached file:] pattern (from user upload)
            elif "[Attached file:" in query:
                # Extract file content
                lines = query.split('\n')
                file_content_start = -1
                file_name = ""

                for i, line in enumerate(lines):
                    if line.startswith('[Attached file:'):
                        file_name = line.replace('[Attached file: ', '').replace(']', '')
                        file_content_start = i + 1
                        break

                if file_content_start == -1:
                    raise ValueError("No file content found in query")

                file_content = '\n'.join(lines[file_content_start:])
            else:
                raise ValueError("No file or file path found in query")

            # Use LLM to analyze the file and extract parameters
            analysis_prompt = f"""Analyze this time series data file and extract forecasting parameters:

Filename: {file_name}
Content preview:
{file_content[:2000]}

This is a generic time series dataset. Analyze the CSV structure and identify:
1. The date/time column name (usually 'date', 'time', 'timestamp')
2. The numeric value column to forecast
   - If format is "date,value" -> use "value"
   - If format is "date,series,value" -> use "value" (the actual numeric column)
   - If format has multiple numeric columns, pick the main one
3. Suggest appropriate forecast periods (default 30)
4. Suggest LSTM time steps (default 60)

Format your response as JSON:
{{
    "date_column": "actual_column_name_from_headers",
    "value_column": "actual_column_name_from_headers",
    "forecast_periods": 30,
    "time_steps": 60,
    "data_description": "brief description"
}}

IMPORTANT: Use the EXACT column names from the CSV headers, not conceptual names."""

            logger.info("📈 Analyzing file structure with LLM...")
            analysis_response = self.llm.call(analysis_prompt)

            try:
                # Clean up response and parse JSON
                clean_response = analysis_response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()

                params = json.loads(clean_response)
                logger.info(f"📈 Extracted parameters: {params}")

            except json.JSONDecodeError:
                logger.warning("📈 Failed to parse LLM response as JSON, using defaults")
                params = {
                    "date_column": None,
                    "value_column": None,
                    "forecast_periods": 30,
                    "time_steps": 60,
                    "data_description": "Auto-detected time series data"
                }

            # Prepare parameters for forecast tool
            tool_params = {
                "data": file_content,
                "forecast_periods": params.get("forecast_periods", 30),
                "time_steps": params.get("time_steps", 60),
                "date_column": params.get("date_column"),
                "value_column": params.get("value_column")
            }

            logger.info(f"📈 Executing forecast tool with parameters: {tool_params}")

            # Execute forecast tool
            tool_result = self._execute_tool_script("forecast", tool_params)

            if not tool_result.get("success", False):
                return {
                    "agent": "ForecastAgent",
                    "success": False,
                    "error": f"Forecast tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Generate LLM analysis of the forecast results
            llm_analysis = self._analyze_forecast_with_llm(tool_result, query, params)

            # Format for downstream agents (use tool_params which contains the historical data)
            formatted_tool_data = self._format_tool_data(tool_result, tool_params)

            # Merge tool result with agent analysis for frontend
            result = {
                **tool_result,  # Include all tool result data (forecast_file_data, model_metrics, etc.)
                "tool_data": formatted_tool_data,
                "llm_analysis": llm_analysis
            }

            logger.info(f"📈 Final result keys: {list(result.keys())}")
            if 'forecast_file_data' in result:
                logger.info(f"📈 forecast_file_data found with keys: {list(result['forecast_file_data'].keys())}")
            else:
                logger.warning("📈 forecast_file_data NOT found in result!")

            return {
                "agent": "ForecastAgent",
                "tool": "forecast",
                "parameters": tool_params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"📈 File forecast processing error: {str(e)}")
            return {
                "agent": "ForecastAgent",
                "success": False,
                "error": f"Failed to process file forecast: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _process_text_forecast(self, query: str) -> Dict[str, Any]:
        """Process forecast request without file (extract from query)"""
        try:
            # Check if this is a context-aware query from orchestrator with stock data
            if "date,series,value" in query and "Price," in query:
                return self._process_stock_price_data(query)

            # Use LLM to understand the forecast request
            extraction_prompt = f"""Extract forecast parameters from this query: "{query}"

This tool performs time series forecasting using LSTM neural networks. It requires actual historical data points with dates and values.

Extract:
1. Any embedded CSV time series data (with date and value columns)
2. Number of periods to forecast (default 30)
3. Any specific column names mentioned
4. Time steps for LSTM (default 60)

Format response as JSON:
{{
    "has_data": true/false,
    "data": "csv_data_if_found",
    "forecast_periods": 30,
    "time_steps": 60,
    "date_column": null,
    "value_column": null,
    "needs_file": true/false
}}"""

            response = self.llm.call(extraction_prompt)

            try:
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                params = json.loads(clean_response.strip())
            except json.JSONDecodeError:
                params = {"has_data": False, "needs_file": True}

            if not params.get("has_data", False):
                return {
                    "agent": "ForecastAgent",
                    "success": False,
                    "error": "No time series data provided. Forecasting requires actual historical data (CSV format with date and value columns). Please attach a CSV file with time series data to make predictions.",
                    "timestamp": datetime.now().isoformat()
                }

            # Execute forecast with extracted data
            tool_params = {
                "data": params.get("data", ""),
                "forecast_periods": params.get("forecast_periods", 30),
                "time_steps": params.get("time_steps", 60),
                "date_column": params.get("date_column"),
                "value_column": params.get("value_column")
            }

            tool_result = self._execute_tool_script("forecast", tool_params)

            if not tool_result.get("success", False):
                return {
                    "agent": "ForecastAgent",
                    "success": False,
                    "error": f"Forecast tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            llm_analysis = self._analyze_forecast_with_llm(tool_result, query, params)
            formatted_tool_data = self._format_tool_data(tool_result, tool_params)

            # Merge tool result with agent analysis for frontend
            result = {
                **tool_result,  # Include all tool result data (forecast_file_data, model_metrics, etc.)
                "tool_data": formatted_tool_data,
                "llm_analysis": llm_analysis
            }

            return {
                "agent": "ForecastAgent",
                "tool": "forecast",
                "parameters": tool_params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"📈 Text forecast processing error: {str(e)}")
            return {
                "agent": "ForecastAgent",
                "success": False,
                "error": f"Failed to process forecast request: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _process_stock_price_data(self, query: str) -> Dict[str, Any]:
        """Process stock price data for forecasting"""
        try:
            logger.info("📈 Processing stock price data for forecasting...")

            # Extract the CSV data from the query
            lines = query.split('\n')
            csv_data_lines = []

            # Look for the CSV data section
            in_data_section = False
            for line in lines:
                if line.startswith("date,series,value"):
                    in_data_section = True
                    # Convert to standard date,value format for forecasting
                    csv_data_lines.append("date,value")
                    continue

                if in_data_section and "Price," in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        date = parts[0]
                        value = parts[2]
                        csv_data_lines.append(f"{date},{value}")
                elif in_data_section and line.strip() == "":
                    break
                elif in_data_section and not line.startswith("2024") and not line.startswith("2023"):
                    # End of data section
                    break

            if not csv_data_lines or len(csv_data_lines) < 2:
                return {
                    "agent": "ForecastAgent",
                    "success": False,
                    "error": "No valid stock price data found for forecasting. Need historical price data with dates and values.",
                    "timestamp": datetime.now().isoformat()
                }

            # Convert to CSV string
            csv_data = '\n'.join(csv_data_lines)

            logger.info(f"📈 Extracted {len(csv_data_lines)-1} price data points for forecasting")

            # Execute forecast with stock price data
            tool_params = {
                "data": csv_data,
                "forecast_periods": 30,  # Default for stock forecasting
                "time_steps": 60,
                "date_column": "date",
                "value_column": "value"
            }

            logger.info(f"📈 Executing forecast tool with stock price data: {len(csv_data)} chars")

            tool_result = self._execute_tool_script("forecast", tool_params)

            if not tool_result.get("success", False):
                return {
                    "agent": "ForecastAgent",
                    "success": False,
                    "error": f"Forecast tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Generate LLM analysis focused on stock price forecasting
            llm_analysis = self._analyze_stock_forecast_with_llm(tool_result, query)
            formatted_tool_data = self._format_tool_data(tool_result, tool_params)

            result = {
                "tool_data": formatted_tool_data,
                "llm_analysis": llm_analysis
            }

            return {
                "agent": "ForecastAgent",
                "tool": "forecast",
                "parameters": tool_params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"📈 Stock price forecasting error: {str(e)}")
            return {
                "agent": "ForecastAgent",
                "success": False,
                "error": f"Failed to process stock price forecast: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_stock_forecast_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Generate LLM analysis specifically for stock price forecasting"""
        try:
            forecast_data = tool_result.get("forecast_data", [])
            model_metrics = tool_result.get("model_metrics", {})
            data_info = tool_result.get("data_info", {})

            if not forecast_data:
                return "Stock price forecast could not be generated due to insufficient data."

            # Get first and last predictions for trend analysis
            first_prediction = forecast_data[0] if forecast_data else {}
            last_prediction = forecast_data[-1] if forecast_data else {}

            analysis_prompt = f"""Analyze this stock price forecast for the query: "{original_query}"

Historical Data: {data_info.get('historical_points', 'N/A')} price points
Forecast Period: {len(forecast_data)} days

Model Performance:
- Mean Absolute Error: {model_metrics.get('mae', 'N/A'):.2f}
- RMSE: {model_metrics.get('rmse', 'N/A'):.2f}

Forecast Results:
- First predicted price ({first_prediction.get('date', 'N/A')}): ${first_prediction.get('predicted_value', 0):.2f}
- Last predicted price ({last_prediction.get('date', 'N/A')}): ${last_prediction.get('predicted_value', 0):.2f}
- Price trend: {"Upward" if last_prediction.get('predicted_value', 0) > first_prediction.get('predicted_value', 0) else "Downward"}

Preview of predictions:
{self._format_forecast_preview(forecast_data[:5])}

Provide a focused stock analysis including:
1. **Price Forecast Summary** - Expected price movement and trend direction
2. **Model Reliability** - How trustworthy are these predictions based on metrics
3. **Key Price Levels** - Important predicted prices to watch
4. **Investment Implications** - What this means for potential investors
5. **Risk Factors** - Limitations of the forecast model

Keep the analysis practical and investment-focused."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📈 Generated stock-focused LLM analysis")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"📈 Error in stock forecast analysis: {str(e)}")
            return f"Stock price forecast completed but analysis failed: {str(e)}"

    def _analyze_forecast_with_llm(self, tool_result: Dict[str, Any], original_query: str, params: Dict[str, Any]) -> str:
        """Use LLM to analyze forecast results and provide insights"""
        try:
            forecast_data = tool_result.get("forecast_data", [])
            model_metrics = tool_result.get("model_metrics", {})
            data_info = tool_result.get("data_info", {})

            analysis_prompt = f"""Analyze these time series forecast results for the user query: "{original_query}"

Forecast Parameters:
- Historical data points: {data_info.get('historical_points', 'N/A')}
- Forecast periods: {data_info.get('forecast_periods', 'N/A')}
- Date column: {data_info.get('date_column', 'Auto-detected')}
- Value column: {data_info.get('value_column', 'Auto-detected')}

Model Performance:
- Mean Absolute Error (MAE): {model_metrics.get('mae', 'N/A')}
- Root Mean Square Error (RMSE): {model_metrics.get('rmse', 'N/A')}

Forecast Results (first 5 predictions):
{self._format_forecast_preview(forecast_data[:5])}

Please provide:
1. **Forecast Summary** - Key insights about the predicted trends
2. **Model Performance** - How reliable are these predictions based on the metrics
3. **Key Trends** - What patterns or trends are visible in the forecast
4. **Recommendations** - Suggested actions based on the forecast
5. **Confidence Assessment** - How confident should the user be in these predictions

Format with clear sections and actionable insights."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📈 Generated LLM analysis for forecast results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"📈 Error in LLM analysis: {str(e)}")
            return f"Forecast completed successfully but analysis failed: {str(e)}"

    def _format_forecast_preview(self, forecast_data: list) -> str:
        """Format forecast data for LLM analysis"""
        if not forecast_data:
            return "No forecast data available"

        formatted = ""
        for i, item in enumerate(forecast_data, 1):
            date = item.get("date", "N/A")
            value = item.get("predicted_value", "N/A")
            if isinstance(value, (int, float)):
                value = f"{value:.2f}"
            formatted += f"{i}. {date}: {value}\n"

        return formatted

    def _format_tool_data(self, tool_result: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Format tool result as CSV data for downstream agents (especially visualization)"""
        try:
            forecast_data = tool_result.get("forecast_data", [])

            if not forecast_data:
                return "Time series forecast failed to generate predictions"

            # Format as CSV for visualization: combine historical + forecast data
            # Create CSV with date,series,value format where series indicates Historical vs Forecast
            csv_data = "date,series,value\n"

            # Get original data from params if available
            original_data = params.get("data", "")
            historical_count = 0
            if original_data and "date,series,value" in original_data:
                # Add historical data first (mark as Historical)
                lines = original_data.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            date, series, value = parts[0], parts[1], parts[2]
                            # Only include Price series for historical data (not moving averages)
                            if series == 'Price':
                                csv_data += f"{date},Historical,{value}\n"
                                historical_count += 1

            # Add forecast data (mark as Forecast)
            forecast_count = 0
            for item in forecast_data:
                date = item.get("date", "")
                value = item.get("predicted_value", 0)
                csv_data += f"{date},Forecast,{value:.2f}\n"
                forecast_count += 1

            logger.info(f"📈 Formatted tool data: {historical_count} historical points, {forecast_count} forecast points")
            logger.info(f"📈 Sample CSV output (first 3 lines):\n{chr(10).join(csv_data.split(chr(10))[:4])}")

            return csv_data

        except Exception as e:
            logger.error(f"📈 Error formatting tool data: {str(e)}")
            return f"Error formatting forecast results: {str(e)}"