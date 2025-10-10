"""
Flight Agent - Specialized agent for flight search operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("FlightSearchAgent")


class FlightSearchAgent(BaseAgent):
    """Specialized agent for flight search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute flight search with intelligent parameter extraction"""
        try:
            logger.info(f"✈️  FlightSearchAgent analyzing: '{query}'")

            # Get current date for context
            from datetime import datetime, timedelta
            today = datetime.now()
            current_year = today.year
            tomorrow = today + timedelta(days=1)

            # Use LLM to extract flight search parameters
            prompt = f"""Extract flight search parameters from this query: "{query}"

CURRENT DATE CONTEXT:
- Today's date: {today.strftime('%Y-%m-%d')} ({today.strftime('%A, %B %d, %Y')})
- Tomorrow's date: {tomorrow.strftime('%Y-%m-%d')}
- Current year: {current_year}

Extract the following information:
- origin: Departure city or airport code (e.g., "San Francisco", "SFO", "New York")
- destination: Arrival city or airport code (e.g., "Tokyo", "NRT", "London")
- departure_date: Departure date in YYYY-MM-DD format
- return_date: Return date in YYYY-MM-DD format (optional, only if round trip is mentioned)

Examples:
- "find flights from New York to London on 2024-12-15" → {{"origin": "New York", "destination": "London", "departure_date": "2024-12-15"}}
- "flights from SFO to Tokyo tomorrow" → {{"origin": "SFO", "destination": "Tokyo", "departure_date": "{tomorrow.strftime('%Y-%m-%d')}"}}
- "round trip from LAX to Paris, leaving Jan 10 returning Jan 20" → {{"origin": "LAX", "destination": "Paris", "departure_date": "{current_year}-01-10", "return_date": "{current_year}-01-20"}}
- "search flights San Francisco to New York next Monday" → Calculate next Monday from today's date
- "one way ticket Miami to Seattle in 3 days" → Calculate 3 days from today

IMPORTANT:
1. Always convert dates to YYYY-MM-DD format
2. For relative dates (tomorrow, next week, in 3 days), calculate from today's date: {today.strftime('%Y-%m-%d')}
3. If year is not mentioned, use current year: {current_year}
4. For month names without year (January, Feb, etc.), use {current_year}
5. Only include return_date if explicitly mentioned or if "round trip" is indicated
6. Extract city names or airport codes exactly as mentioned

Respond with JSON only:
{{"origin": "city or code", "destination": "city or code", "departure_date": "YYYY-MM-DD", "return_date": "YYYY-MM-DD or null"}}"""

            response = self.llm.call(prompt).strip()
            logger.info(f"✈️  Parameter extraction response: {response}")

            try:
                # Clean up the response - remove markdown formatting
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.startswith("```"):
                    clean_response = clean_response[3:]  # Remove ```
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()

                params = json.loads(clean_response)
                logger.info(f"✈️  Successfully parsed parameters: {params}")

                # Validate required parameters
                if not params.get("origin"):
                    return {
                        "tool": "flight_search",
                        "success": False,
                        "error": "Could not extract origin (departure) city from query",
                        "timestamp": datetime.now().isoformat()
                    }

                if not params.get("destination"):
                    return {
                        "tool": "flight_search",
                        "success": False,
                        "error": "Could not extract destination (arrival) city from query",
                        "timestamp": datetime.now().isoformat()
                    }

                if not params.get("departure_date"):
                    return {
                        "tool": "flight_search",
                        "success": False,
                        "error": "Could not extract departure date from query",
                        "timestamp": datetime.now().isoformat()
                    }

                # Remove null return_date
                if params.get("return_date") == "null" or params.get("return_date") is None:
                    params.pop("return_date", None)

            except json.JSONDecodeError as e:
                logger.error(f"✈️  JSON parsing error: {e}")
                return {
                    "tool": "flight_search",
                    "success": False,
                    "error": f"Failed to extract flight parameters from query. Please specify origin, destination, and departure date clearly.",
                    "timestamp": datetime.now().isoformat()
                }

            logger.info(f"✈️  Executing flight tool with: {params}")

            # Execute flight search tool
            tool_result = self._execute_tool_script("flight_search", params)
            logger.info(f"✈️  Tool execution completed: {len(str(tool_result))} characters")

            if not tool_result.get("success", False):
                return {
                    "tool": "flight_search",
                    "success": False,
                    "error": f"Flight search tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Use LLM to analyze the search results
            llm_analysis = self._analyze_flight_results_with_llm(tool_result, query)

            # Get tool_data from the tool result
            formatted_tool_data = tool_result.get("tool_data", self._format_tool_data(tool_result))

            # Get flight data and convert to results array for consistency with other tools
            flights_data = tool_result.get("flights", {"outbound": [], "return": []})

            # Combine outbound and return flights into a single results array
            all_flights = flights_data.get("outbound", []) + flights_data.get("return", [])

            result = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis,      # LLM insights
                "results": all_flights,            # All flights in standard results format
                "query": tool_result.get("query", {}),      # Query info for FlightCard
                "results_count": len(all_flights),  # Count for FlightCard
                "flights": flights_data  # Also include structured flights for FlightCard
            }

            return {
                "tool": "flight_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"✈️  FlightSearchAgent error: {str(e)}")
            return {
                "tool": "flight_search",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_flight_results_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze flight search results and provide insights"""
        try:
            # Extract relevant information from tool result
            flights_data = tool_result.get("flights", {"outbound": [], "return": []})
            outbound_flights = flights_data.get("outbound", [])
            return_flights = flights_data.get("return", [])
            query_info = tool_result.get("query", {})
            origin = query_info.get("origin", "Unknown")
            destination = query_info.get("destination", "Unknown")
            departure_date = query_info.get("departure_date", "Unknown")
            return_date = query_info.get("return_date")
            trip_type = query_info.get("trip_type", "one way")

            analysis_prompt = f"""Analyze these flight search results and provide helpful insights for the user query: "{original_query}"

Flight Search Details:
- Route: {origin} → {destination}
- Departure Date: {departure_date}
{"- Return Date: " + return_date if return_date else "- Trip Type: One Way"}
- Outbound Flights Found: {len(outbound_flights)}
{"- Return Flights Found: " + str(len(return_flights)) if return_flights else ""}

Outbound Flight Options:
{self._format_flight_results_for_analysis(outbound_flights)}

{"Return Flight Options:" if return_flights else ""}
{self._format_flight_results_for_analysis(return_flights) if return_flights else ""}

Please provide:
1. **Summary** - Brief overview of available flight options and price ranges
2. **Key Findings** - Highlight the cheapest options, nonstop flights, and airline choices
3. **Recommendations** - Suggest the best value flights based on price, duration, and stops
4. **Booking Tips** - Any relevant advice about timing, prices, or airlines for this route

Format your response with:
- Clear section headers using **bold** markdown
- Well-organized information
- Bullet points for easy reading
- Highlight important details like prices and flight times

Focus on providing actionable information to help the user choose and book the best flight."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"✈️  Generated LLM analysis for flight search results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"✈️  Error in LLM analysis: {str(e)}")
            return f"Flight search completed but LLM analysis failed: {str(e)}"

    def _format_flight_results_for_analysis(self, results: list) -> str:
        """Format flight results for LLM analysis"""
        try:
            if not results:
                return "No flight options available"

            formatted = ""
            for i, flight in enumerate(results, 1):
                airline = flight.get("airline", "Unknown")
                price = flight.get("price", "N/A")
                duration = flight.get("duration", "N/A")
                stops = flight.get("stops", "N/A")
                departure_time = flight.get("departure_time", "N/A")
                arrival_time = flight.get("arrival_time", "N/A")

                stops_text = "Nonstop" if stops == 0 else f"{stops} stop(s)" if isinstance(stops, int) else str(stops)

                formatted += f"{i}. {airline} - {price}\n"
                formatted += f"   Time: {departure_time} → {arrival_time}\n"
                formatted += f"   Duration: {duration}, {stops_text}\n\n"

            return formatted
        except Exception as e:
            logger.error(f"✈️  Error formatting flight results: {str(e)}")
            return "Error formatting flight results"

    def _format_tool_data(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as text for downstream agents"""
        try:
            flight_results = tool_result.get("results", [])
            query_info = tool_result.get("query", {})
            origin = query_info.get("origin", "Unknown")
            destination = query_info.get("destination", "Unknown")
            departure_date = query_info.get("departure_date", "Unknown")
            return_date = query_info.get("return_date")

            if not flight_results:
                return f"Flight Search: {origin} to {destination} on {departure_date}\nNo results found"

            formatted_text = f"Flight Search Results:\n"
            formatted_text += f"Route: {origin} → {destination}\n"
            formatted_text += f"Departure: {departure_date}\n"
            if return_date:
                formatted_text += f"Return: {return_date}\n"
            formatted_text += f"Results Found: {len(flight_results)}\n\n"

            # Show first few results
            for i, result in enumerate(flight_results[:5], 1):
                title = result.get("title", "No title")
                source = result.get("source", "Unknown")
                url = result.get("url", "No URL")
                formatted_text += f"{i}. {title}\n"
                formatted_text += f"   Source: {source}\n"
                formatted_text += f"   URL: {url}\n\n"

            if len(flight_results) > 5:
                formatted_text += f"... and {len(flight_results) - 5} more results\n"

            return formatted_text
        except Exception as e:
            logger.error(f"✈️  Error formatting tool data: {str(e)}")
            return f"Error formatting flight search results: {str(e)}"
