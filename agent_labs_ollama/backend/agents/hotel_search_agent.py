"""
Hotel Search Agent - Specialized agent for hotel search operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("HotelSearchAgent")


class HotelSearchAgent(BaseAgent):
    """Specialized agent for hotel search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute hotel search with intelligent parameter extraction"""
        try:
            logger.info(f"🏨 HotelSearchAgent analyzing: '{query}'")

            # Get current date for context
            from datetime import datetime, timedelta
            today = datetime.now()
            current_year = today.year
            tomorrow = today + timedelta(days=1)

            # Use LLM to extract hotel search parameters
            prompt = f"""Extract hotel search parameters from this query: "{query}"

CURRENT DATE CONTEXT:
- Today's date: {today.strftime('%Y-%m-%d')} ({today.strftime('%A, %B %d, %Y')})
- Tomorrow's date: {tomorrow.strftime('%Y-%m-%d')}
- Current year: {current_year}

Extract the following information:
- location: City or area for hotel search (e.g., "San Francisco, CA", "Paris, France", "New York")
- check_in: Check-in date in YYYY-MM-DD format
- check_out: Check-out date in YYYY-MM-DD format
- guests: Number of guests (default: 2 if not mentioned)

Examples:
- "find hotels in San Francisco from Dec 20 to Dec 25" → {{"location": "San Francisco, CA", "check_in": "{current_year}-12-20", "check_out": "{current_year}-12-25", "guests": 2}}
- "hotels in Paris tomorrow for 3 nights" → {{"location": "Paris, France", "check_in": "{tomorrow.strftime('%Y-%m-%d')}", "check_out": "calculate 3 nights from tomorrow", "guests": 2}}
- "book a hotel in Tokyo checking in Jan 10 checking out Jan 15 for 4 people" → {{"location": "Tokyo, Japan", "check_in": "{current_year}-01-10", "check_out": "{current_year}-01-15", "guests": 4}}
- "need accommodation in London next week for 2 nights" → Calculate check-in as next week Monday, check-out as 2 nights later
- "hotel near Times Square in 5 days, staying 1 week" → Calculate dates from today

IMPORTANT:
1. Always convert dates to YYYY-MM-DD format
2. For relative dates (tomorrow, next week, in 3 days), calculate from today's date: {today.strftime('%Y-%m-%d')}
3. If year is not mentioned, use current year: {current_year}
4. For month names without year (January, Feb, etc.), use {current_year}
5. If check-out is expressed as "X nights", calculate it as check_in + X days
6. Default to 2 guests if not mentioned
7. Include city and country/state when possible for location

Respond with JSON only:
{{"location": "city, country/state", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "guests": number}}"""

            response = self.llm.call(prompt).strip()
            logger.info(f"🏨 Parameter extraction response: {response}")

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
                logger.info(f"🏨 Successfully parsed parameters: {params}")

                # Validate required parameters
                if not params.get("location"):
                    return {
                        "tool": "hotel_search",
                        "success": False,
                        "error": "Could not extract location from query",
                        "timestamp": datetime.now().isoformat()
                    }

                if not params.get("check_in"):
                    return {
                        "tool": "hotel_search",
                        "success": False,
                        "error": "Could not extract check-in date from query",
                        "timestamp": datetime.now().isoformat()
                    }

                if not params.get("check_out"):
                    return {
                        "tool": "hotel_search",
                        "success": False,
                        "error": "Could not extract check-out date from query",
                        "timestamp": datetime.now().isoformat()
                    }

                # Ensure guests defaults to 2
                if not params.get("guests"):
                    params["guests"] = 2

            except json.JSONDecodeError as e:
                logger.error(f"🏨 JSON parsing error: {e}")
                return {
                    "tool": "hotel_search",
                    "success": False,
                    "error": f"Failed to extract hotel parameters from query. Please specify location, check-in, and check-out dates clearly.",
                    "timestamp": datetime.now().isoformat()
                }

            logger.info(f"🏨 Executing hotel tool with: {params}")

            # Execute hotel search tool
            tool_result = self._execute_tool_script("hotel_search", params)
            logger.info(f"🏨 Tool execution completed: {len(str(tool_result))} characters")

            if not tool_result.get("success", False):
                return {
                    "tool": "hotel_search",
                    "success": False,
                    "error": f"Hotel search tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Use LLM to analyze the search results
            llm_analysis = self._analyze_hotel_results_with_llm(tool_result, query)

            # Get tool_data from the tool result
            formatted_tool_data = tool_result.get("tool_data", self._format_tool_data(tool_result))

            # Get hotel data
            hotels_data = tool_result.get("hotels", [])

            result = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis,      # LLM insights
                "results": hotels_data,            # All hotels in standard results format
                "query": tool_result.get("query", {}),      # Query info for HotelCard
                "results_count": len(hotels_data),  # Count for HotelCard
                "hotels": hotels_data  # Hotel data for HotelCard
            }

            return {
                "tool": "hotel_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🏨 HotelSearchAgent error: {str(e)}")
            return {
                "tool": "hotel_search",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_hotel_results_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze hotel search results and provide insights"""
        try:
            # Extract relevant information from tool result
            hotels_data = tool_result.get("hotels", [])
            query_info = tool_result.get("query", {})
            location = query_info.get("location", "Unknown")
            check_in = query_info.get("check_in", "Unknown")
            check_out = query_info.get("check_out", "Unknown")
            nights = query_info.get("nights", 1)
            guests = query_info.get("guests", 2)

            analysis_prompt = f"""Analyze these hotel search results and provide helpful insights for the user query: "{original_query}"

Hotel Search Details:
- Location: {location}
- Check-in: {check_in}
- Check-out: {check_out}
- Nights: {nights}
- Guests: {guests}
- Hotels Found: {len(hotels_data)}

Hotel Options:
{self._format_hotel_results_for_analysis(hotels_data)}

Please provide:
1. **Summary** - Brief overview of available hotel options and price ranges
2. **Key Findings** - Highlight the best value options, highly-rated hotels, and category distribution
3. **Recommendations** - Suggest the best hotels based on rating, price, location, and amenities
4. **Booking Tips** - Any relevant advice about pricing, location, amenities, or cancellation policies

Format your response with:
- Clear section headers using **bold** markdown
- Well-organized information
- Bullet points for easy reading
- Highlight important details like prices, ratings, and amenities

Focus on providing actionable information to help the user choose and book the best hotel."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"🏨 Generated LLM analysis for hotel search results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"🏨 Error in LLM analysis: {str(e)}")
            return f"Hotel search completed but LLM analysis failed: {str(e)}"

    def _format_hotel_results_for_analysis(self, results: list) -> str:
        """Format hotel results for LLM analysis"""
        try:
            if not results:
                return "No hotel options available"

            formatted = ""
            for i, hotel in enumerate(results, 1):
                name = hotel.get("name", "Unknown")
                category = hotel.get("category", "Unknown")
                rating = hotel.get("rating", "N/A")
                reviews = hotel.get("reviews", 0)
                price_per_night = hotel.get("price_per_night", "N/A")
                total_price = hotel.get("total_price", "N/A")
                distance = hotel.get("distance_from_center", "N/A")
                amenities = hotel.get("amenities", [])
                cancellation = hotel.get("cancellation_policy", "N/A")

                formatted += f"{i}. {name}\n"
                formatted += f"   Category: {category} | Rating: {rating}/5.0 ({reviews} reviews)\n"
                formatted += f"   Price: {price_per_night}/night (Total: {total_price})\n"
                formatted += f"   Distance: {distance} from center\n"
                formatted += f"   Amenities: {', '.join(amenities[:5])}\n"
                formatted += f"   {cancellation}\n\n"

            return formatted
        except Exception as e:
            logger.error(f"🏨 Error formatting hotel results: {str(e)}")
            return "Error formatting hotel results"

    def _format_tool_data(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as text for downstream agents"""
        try:
            hotel_results = tool_result.get("hotels", [])
            query_info = tool_result.get("query", {})
            location = query_info.get("location", "Unknown")
            check_in = query_info.get("check_in", "Unknown")
            check_out = query_info.get("check_out", "Unknown")
            nights = query_info.get("nights", 1)

            if not hotel_results:
                return f"Hotel Search: {location} from {check_in} to {check_out}\nNo results found"

            formatted_text = f"Hotel Search Results:\n"
            formatted_text += f"Location: {location}\n"
            formatted_text += f"Check-in: {check_in}\n"
            formatted_text += f"Check-out: {check_out}\n"
            formatted_text += f"Nights: {nights}\n"
            formatted_text += f"Results Found: {len(hotel_results)}\n\n"

            # Show first few results
            for i, result in enumerate(hotel_results[:5], 1):
                name = result.get("name", "No name")
                rating = result.get("rating", "N/A")
                price = result.get("price_per_night", "N/A")
                formatted_text += f"{i}. {name}\n"
                formatted_text += f"   Rating: {rating}/5.0\n"
                formatted_text += f"   Price: {price}/night\n\n"

            if len(hotel_results) > 5:
                formatted_text += f"... and {len(hotel_results) - 5} more results\n"

            return formatted_text
        except Exception as e:
            logger.error(f"🏨 Error formatting tool data: {str(e)}")
            return f"Error formatting hotel search results: {str(e)}"
