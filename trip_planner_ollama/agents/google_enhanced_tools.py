"""
Google Search Enhanced Travel Tools

Enhanced travel tools that integrate with Google Search for real-time data
in comprehensive mode collaboration.
"""

import asyncio
import json
import logging
import ast
from typing import Dict, Any, List, Union
from datetime import datetime, timedelta

from langchain.tools import BaseTool, tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Type

from services.google_travel_search import GoogleTravelSearch
from config import get_config

logger = logging.getLogger(__name__)

config = get_config()

def _parse_json_or_literal(s: str) -> Dict[str, Any]:
    logger.debug(f"🔍 Parsing input: '{s}' (type: {type(s)}, length: {len(s)})")
    logger.debug(f"🔍 Input repr: {repr(s)}")
    
    # Clean up common malformations from LangChain retries
    cleaned_s = s.strip()
    
    # Extract just the JSON dictionary from contaminated LangChain logs
    # Look for the pattern {'key': 'value', ...} and extract it
    import re
    json_pattern = r"(\{[^{}]*(?:'[^']*'[^{}]*)*\})"
    json_matches = re.findall(json_pattern, cleaned_s)
    
    if json_matches:
        # Use the first (likely only) JSON-like structure found
        cleaned_s = json_matches[0]
        logger.debug(f"🔍 Extracted JSON from contaminated string: {cleaned_s}")
    else:
        # If no JSON pattern found, try to clean whitespace normally
        cleaned_s = ' '.join(cleaned_s.split())
    
    # Handle case where there's an extra quote at the end (LangChain retry bug)
    if cleaned_s.endswith("'") and cleaned_s.count("'") % 2 != 0:
        logger.debug(f"Detected extra quote, removing: {cleaned_s}")
        cleaned_s = cleaned_s[:-1]
    
    logger.debug(f"🔍 Final cleaned string: '{cleaned_s}'")
    logger.debug(f"🔍 Final repr: {repr(cleaned_s)}")
    
    try:
        # First, try to evaluate as a Python literal
        logger.debug("🔍 Attempting ast.literal_eval...")
        data = ast.literal_eval(cleaned_s)
        logger.debug(f"✅ ast.literal_eval succeeded: {data}")
        if isinstance(data, str):
            # If the result is a string, it might be JSON inside a string
            logger.debug("🔍 Result is string, trying json.loads...")
            data = json.loads(data)
        return data
    except (ValueError, SyntaxError) as e:
        logger.debug(f"❌ ast.literal_eval failed: {e}")
        # If literal_eval fails, try to parse as JSON
        try:
            logger.debug("🔍 Attempting json.loads...")
            result = json.loads(cleaned_s)
            logger.debug(f"✅ json.loads succeeded: {result}")
            return result
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse string as JSON or Python literal: {s}")
            logger.error(f"Final cleaned string was: {cleaned_s}")
            logger.error(f"ast.literal_eval error: {e}")
            logger.error(f"json.loads error: {e2}")
            raise ValueError(f"Invalid input format: {s}") from e2

class FlightSearchInput(BaseModel):
    query: str = Field(description="JSON string with flight search parameters")

class HotelSearchInput(BaseModel):
    query: str = Field(description="JSON string with hotel search parameters")

class ActivitySearchInput(BaseModel):
    query: str = Field(description="JSON string with activity search parameters")

class GoogleFlightSearchTool(BaseTool):
    """Enhanced flight search tool using Google Search integration."""
    
    name = "google_flight_search"
    description = """Search for flights using Google Search with real-time data.
    Input should be a JSON string with keys: origin, destination, departure_date.
    Example: '{\"origin\": \"Seattle\", \"destination\": \"Tokyo\", \"departure_date\": \"2025-09-02\"}'
    Returns comprehensive flight information with airlines, times, and prices."""
    
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, query: str) -> str:
        """Execute Google-enhanced flight search synchronously."""
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute Google-enhanced flight search asynchronously."""
        try:
            params = _parse_json_or_literal(query)
            origin = params.get("origin", "")
            destination = params.get("destination", "") 
            departure_date = params.get("departure_date", "")
            
            if not all([origin, destination, departure_date]):
                return "❌ Missing required parameters. Need origin, destination, and departure_date."
            
            logger.debug(f"🔍 Google flight search: {origin} → {destination} on {departure_date}")
            
            async with GoogleTravelSearch(api_key=config.google_search_api_key, search_engine_id=config.google_search_engine_id) as search:
                flights = await search.search_flights(origin, destination, departure_date)
            
            if not flights:
                return f"❌ No flights found from {origin} to {destination} on {departure_date}"
            
            # Format results for LLM consumption with data source indicators
            results = []
            for i, flight in enumerate(flights, 1):
                # Determine data source based on confidence and fallback status
                data_source = "google"
                result = f"Flight {i}: {flight.airline} - Depart: {flight.departure_time}, Arrive: {flight.arrival_time}, Price: {flight.price}"
                if hasattr(flight, 'duration') and flight.duration:
                    result += f", Duration: {flight.duration}"
                result += f" [Source: {data_source}]"
                results.append(result)
            
            formatted_results = "\n".join(results)
            logger.info(f"✅ Found {len(flights)} Google flights from {origin} to {destination}")
            
            return f"Found {len(flights)} flights from {origin} to {destination} on {departure_date}:\n{formatted_results}"
            
        except ValueError as e:
            logger.error(f"❌ JSON decode error in google_flight_search: {e}")
            logger.error(f"❌ Query was: '{query}'")
            return f"❌ Invalid JSON format. Please provide: {{'origin': 'City', 'destination': 'City', 'departure_date': 'YYYY-MM-DD'}}"
        except Exception as e:
            logger.error(f"❌ Google flight search error: {e}")
            return f"❌ Flight search failed: {str(e)}"

class GoogleHotelSearchTool(BaseTool):
    """Enhanced hotel search tool using Google Search integration."""
    
    name = "google_hotel_search"
    description = """Search for hotels using Google Search with real-time data.
    Input should be a JSON string with keys: city, check_in, check_out.
    Example: '{\"city\": \"Tokyo\", \"check_in\": \"2025-09-02\", \"check_out\": \"2025-09-07\"}'
    Returns comprehensive hotel information with names, ratings, prices, and amenities."""
    
    args_schema: Type[BaseModel] = HotelSearchInput

    def _run(self, query: str) -> str:
        """Execute Google-enhanced hotel search synchronously."""
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute Google-enhanced hotel search asynchronously."""
        try:
            params = _parse_json_or_literal(query)
            city = params.get("city", "")
            check_in = params.get("check_in", "")
            check_out = params.get("check_out", "")
            
            if not all([city, check_in, check_out]):
                return "❌ Missing required parameters. Need city, check_in, and check_out dates."
            
            logger.debug(f"🔍 Google hotel search: {city} from {check_in} to {check_out}")
            
            async with GoogleTravelSearch(api_key=config.google_search_api_key, search_engine_id=config.google_search_engine_id) as search:
                hotels = await search.search_hotels(city, check_in, check_out)
            
            if not hotels:
                return f"❌ No hotels found in {city} for {check_in} to {check_out}"
            
            # Format results for LLM consumption with data source indicators
            results = []
            for i, hotel in enumerate(hotels, 1):
                amenities_str = ", ".join(hotel.amenities[:4])  # Show top 4 amenities
                # Determine data source based on confidence and fallback status
                data_source = "google"
                result = f"Hotel {i}: {hotel.name} - {hotel.price_per_night}/night, Rating: {hotel.rating}"
                if amenities_str:
                    result += f", Amenities: {amenities_str}"
                if hasattr(hotel, 'address') and hotel.address:
                    result += f", Location: {hotel.address}"
                result += f" [Source: {data_source}]"
                results.append(result)
            
            formatted_results = "\n".join(results)
            logger.info(f"✅ Found {len(hotels)} Google hotels in {city}")
            
            return f"Found {len(hotels)} hotels in {city} for {check_in} to {check_out}:\n{formatted_results}"
            
        except ValueError as e:
            logger.error(f"❌ JSON decode error in google_hotel_search: {e}")
            logger.error(f"❌ Query was: '{query}'")
            return f"❌ Invalid JSON format. Please provide: {{'city': 'City', 'check_in': 'YYYY-MM-DD', 'check_out': 'YYYY-MM-DD'}}"
        except Exception as e:
            logger.error(f"❌ Google hotel search error: {e}")
            return f"❌ Hotel search failed: {str(e)}"

class GoogleActivitySearchTool(BaseTool):
    """Enhanced activity search tool using Google Search integration."""
    
    name = "google_activity_search"
    description = """Search for activities and attractions using Google Search.
    Input should be a JSON string with keys: city, interests (optional).
    Example: '{\"city\": \"Tokyo\", \"interests\": [\"food\", \"culture\"]}'
    Returns activity recommendations with descriptions and locations."""
    
    args_schema: Type[BaseModel] = ActivitySearchInput

    def _run(self, query: str) -> str:
        """Execute Google-enhanced activity search synchronously."""
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute Google-enhanced activity search asynchronously."""
        try:
            params = _parse_json_or_literal(query)
            city = params.get("city", "")
            interests = params.get("interests", [])
            
            if not city:
                return "❌ Missing required parameter: city"
            
            # Build search query based on interests
            if interests:
                interests_str = " ".join(interests)
                search_query = f"things to do in {city} {interests_str} attractions activities"
            else:
                search_query = f"top attractions things to do in {city} activities sightseeing"
            
            logger.debug(f"🔍 Google activity search: {search_query}")
            
            async with GoogleTravelSearch(api_key=config.google_search_api_key, search_engine_id=config.google_search_engine_id) as search:
                results = await search.search_web(search_query, num_results=6)
            
            if not results:
                return f"❌ No activities found in {city}"
            
            # Format results for LLM consumption with data source indicators
            activities = []
            for i, result in enumerate(results, 1):
                title = result.get('title', f'Activity {i}')
                description = result.get('snippet', 'No description available')[:150]
                # Activities from Google Search have high confidence when real API results, lower for fallback
                data_source = "google"
                activity = f"Activity {i}: {title} - {description} [Source: {data_source}]"
                activities.append(activity)
            
            formatted_results = "\n".join(activities)
            logger.info(f"✅ Found {len(results)} Google activities in {city}")
            
            return f"Found {len(results)} activities in {city}:\n{formatted_results}"
            
        except ValueError as e:
            logger.error(f"❌ JSON decode error in google_activity_search: {e}")
            logger.error(f"❌ Query was: '{query}'")
            return f"❌ Invalid JSON format. Please provide: {{'city': 'City', 'interests': ['interest1', 'interest2']}}"
        except Exception as e:
            logger.error(f"❌ Google activity search error: {e}")
            return f"❌ Activity search failed: {str(e)}"

class GoogleEnhancedTravelTools:
    """Collection of Google Search enhanced travel tools for comprehensive mode."""
    
    def __init__(self):
        self.google_flight_search = GoogleFlightSearchTool()
        self.google_hotel_search = GoogleHotelSearchTool()
        self.google_activity_search = GoogleActivitySearchTool()
        
        logger.debug("🚀 Google Enhanced Travel Tools initialized")
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all Google enhanced tools as a list."""
        return [
            self.google_flight_search,
            self.google_hotel_search,
            self.google_activity_search
        ]
    
    def get_flight_tools(self) -> List[BaseTool]:
        """Get flight-specific Google enhanced tools."""
        return [self.google_flight_search]
    
    def get_hotel_tools(self) -> List[BaseTool]:
        """Get hotel-specific Google enhanced tools."""
        return [self.google_hotel_search]
    
    def get_activity_tools(self) -> List[BaseTool]:
        """Get activity-specific Google enhanced tools."""
        return [self.google_activity_search]
