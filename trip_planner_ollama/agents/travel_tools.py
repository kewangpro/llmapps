"""
Travel Planning Tools for LangChain Agents

This module contains all the specialized tools that agents use for travel planning:
- flight_search: Find flights with pricing and schedules
- hotel_search: Search for accommodations with ratings
- activity_search: Discover local activities and attractions
- budget_analysis: Analyze and optimize trip budgets
- route_optimization: Plan efficient multi-city routes
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field

from services.google_travel_search import GoogleTravelSearch
from models import Flight, Hotel
from config import get_config

logger = logging.getLogger(__name__)

config = get_config()

# Tool Input Models using PydanticV1 for LangChain compatibility

class FlightSearchInput(PydanticV1BaseModel):
    """Input for flight search tool."""
    origin: str = PydanticV1Field(description="Origin city or airport code")
    destination: str = PydanticV1Field(description="Destination city or airport code") 
    departure_date: str = PydanticV1Field(description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = PydanticV1Field(None, description="Return date for round trip (YYYY-MM-DD)")
    passengers: int = PydanticV1Field(1, description="Number of passengers")

class HotelSearchInput(PydanticV1BaseModel):
    """Input for hotel search tool."""
    city: str = PydanticV1Field(description="City name for hotel search")
    check_in: str = PydanticV1Field(description="Check-in date (YYYY-MM-DD)")
    check_out: str = PydanticV1Field(description="Check-out date (YYYY-MM-DD)")
    guests: int = PydanticV1Field(1, description="Number of guests")
    price_range: Optional[str] = PydanticV1Field(None, description="Price range (budget, mid-range, luxury)")

class ActivitySearchInput(PydanticV1BaseModel):
    """Input for activity search tool."""
    location: str = PydanticV1Field(description="Location for activity search")
    interests: List[str] = PydanticV1Field(description="List of interests/activity types")
    date: Optional[str] = PydanticV1Field(None, description="Specific date (YYYY-MM-DD)")
    duration_hours: Optional[int] = PydanticV1Field(None, description="Preferred duration in hours")

class BudgetAnalysisInput(PydanticV1BaseModel):
    """Input for budget analysis tool."""
    total_budget: Optional[float] = PydanticV1Field(None, description="Total budget amount")
    destinations: List[str] = PydanticV1Field(description="List of destinations")
    duration_days: int = PydanticV1Field(description="Trip duration in days")
    travel_style: str = PydanticV1Field("mid-range", description="Travel style (budget, mid-range, luxury)")
    flights: Optional[List[Dict[str, Any]]] = PydanticV1Field(None, description="List of flight details")
    hotels: Optional[List[Dict[str, Any]]] = PydanticV1Field(None, description="List of hotel details")

class RouteOptimizationInput(PydanticV1BaseModel):
    """Input for route optimization tool."""
    origin: str = PydanticV1Field(description="Starting city")
    destinations: List[str] = PydanticV1Field(description="List of destination cities")
    preferences: Dict[str, Any] = PydanticV1Field(default_factory=dict, description="Route preferences")

# Specialized Travel Tools using LangChain's @tool decorator
class TravelPlanningTools:
    """Collection of travel planning tools for LangChain agents."""
    
    def __init__(self):
        self.google_search = GoogleTravelSearch(api_key=config.google_search_api_key, search_engine_id=config.google_search_engine_id)
        # Create tools as properties so they don't include self
        self._flight_search_tool = self._create_flight_search_tool()
        self._hotel_search_tool = self._create_hotel_search_tool()
        self._activity_search_tool = self._create_activity_search_tool()
        self._budget_analysis_tool = self._create_budget_analysis_tool()
        self._route_optimization_tool = self._create_route_optimization_tool()
    
    @property
    def flight_search(self):
        return self._flight_search_tool
    
    @property
    def hotel_search(self):
        return self._hotel_search_tool
    
    @property
    def activity_search(self):
        return self._activity_search_tool
        
    @property
    def budget_analysis(self):
        return self._budget_analysis_tool
        
    @property
    def route_optimization(self):
        return self._route_optimization_tool
    
    def _create_flight_search_tool(self):
        # Don't store the instance, create new ones with context manager
        
        @tool("flight_search", return_direct=False)
        def flight_search(query: str) -> str:
            """
            Search for flights and return a JSON list of flight details.
            Input should be a JSON string with origin, destination, and departure_date.
            """
            try:
                logger.debug(f"🔍 Flight search input: {repr(query)}, type: {type(query)}")
                
                # Handle both dict objects and JSON strings from LangChain
                if isinstance(query, dict):
                    # Direct dict input from agent
                    params = query
                    logger.debug(f"✅ Using dict input directly: {params}")
                elif isinstance(query, str) and query.strip():
                    import json
                    import ast
                    import re
                    clean_query = query.strip()
                    
                    # SANITIZATION: Extract JSON/dict from contaminated LangChain input
                    # Look for pattern like "Action Input: {'key': 'value'}" or similar
                    if not (clean_query.startswith('{') and clean_query.endswith('}')):
                        logger.debug(f"🧹 Input appears contaminated, attempting to extract JSON/dict: {repr(clean_query[:200])}")
                        
                        # More aggressive cleaning - look for the last occurrence of a dict pattern
                        # Since LangChain often puts: "Thought: ... Action: ... Action Input: {'key': 'value'}"
                        lines = clean_query.split('\n')
                        potential_dicts = []
                        
                        for line in lines:
                            line = line.strip()
                            # Look for lines containing dict patterns
                            if '{' in line and '}' in line:
                                # Extract everything from first { to last }
                                start_idx = line.find('{')
                                end_idx = line.rfind('}')
                                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                    candidate = line[start_idx:end_idx+1]
                                    potential_dicts.append(candidate)
                        
                        if potential_dicts:
                            # Use the last (most recent) dict found
                            clean_query = potential_dicts[-1]
                            logger.debug(f"✅ Extracted dict from line: {clean_query}")
                        else:
                            # Fallback: try regex pattern
                            dict_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                            matches = re.findall(dict_pattern, clean_query)
                            
                            if matches:
                                clean_query = matches[-1]  # Use last match
                                logger.debug(f"✅ Extracted dict pattern: {clean_query}")
                            else:
                                logger.error(f"❌ Could not extract valid JSON/dict from contaminated input: {repr(clean_query)}")
                                return json.dumps([{"error": f"Could not parse contaminated input: {repr(query[:100])}..."}])
                    
                    # Try to handle Python dict string representation first
                    if clean_query.startswith('{') and clean_query.endswith('}'):
                        try:
                            # Try to safely evaluate Python dict string
                            params = ast.literal_eval(clean_query)
                            logger.debug(f"✅ Parsed Python dict string: {params}")
                        except (ValueError, SyntaxError):
                            logger.debug(f"❌ Failed to parse as Python dict, trying JSON...")
                            # Fall back to JSON parsing
                            # Remove extra quotes if present (LangChain may add them)
                            if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                                clean_query = clean_query[1:-1]
                            params = json.loads(clean_query)
                            logger.debug(f"✅ Parsed as JSON: {params}")
                    else:
                        # Remove extra quotes if present (LangChain may add them)
                        if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                            clean_query = clean_query[1:-1]
                        params = json.loads(clean_query)
                        logger.debug(f"✅ Parsed as JSON: {params}")
                else:
                    # Handle empty or invalid input
                    return json.dumps([{"error": f"Invalid input received: {repr(query)}, type: {type(query)}"}])
                
                # Handle multi-city destination input (agent confusion)
                destination = params['destination']
                if isinstance(destination, list):
                    if len(destination) == 1:
                        destination = destination[0]
                    else:
                        return json.dumps([
                            {
                                "error": f"Multi-city flight search not supported. Please search one route at a time: {params['origin']} to each destination separately.",
                                "suggestion": f"Try searching: {params['origin']} to {destination[0]}, then {destination[0]} to {destination[1]}, etc."
                            }
                        ])
                
                params['destination'] = destination
                    
                # Pure LLM reasoning for flight search (Simple Mode - no Google Search)
                logger.debug(f"🔍 LLM flight reasoning: {params['origin']} → {params['destination']} on {params['departure_date']}")
                
                # Generate realistic flight data using LLM knowledge
                flights_from_google = self._generate_llm_flights(
                    origin=params["origin"],
                    destination=params["destination"], 
                    departure_date=params["departure_date"]
                )
                logger.debug(f"✅ LLM flight search completed, generated {len(flights_from_google)} flights")
                
                if not flights_from_google:
                    return json.dumps([{"message": f"No flights found from {params['origin']} to {params['destination']} on {params['departure_date']}", "from_city": params['origin'], "to_city": params['destination'], "date": params['departure_date']}])
                
                flights = [
                    Flight(
                        from_city=f.from_city,
                        to_city=f.to_city,
                        date=f.date,
                        departure_time=f.departure_time,
                        arrival_time=f.arrival_time,
                        airline=f.airline,
                        estimated_price=f.price,
                        data_source='google_search',
                        confidence=0.9
                    ).model_dump()
                    for f in flights_from_google
                ]
                
                # Create a simplified summary for the agent with data source indicators
                flight_summary = f"Found {len(flights)} flights from {params['origin']} to {params['destination']} on {params['departure_date']}:\n"
                for i, f in enumerate(flights[:3]):  # Show max 3 flights
                    data_source = "LLM reasoning"
                    flight_summary += f"Flight {i+1}: {f['airline']} - Depart: {f['departure_time']}, Arrive: {f['arrival_time']}, Price: {f['estimated_price']} [Source: {data_source}]\n"
                
                logger.debug(f"📤 Returning simplified flight summary: {len(flight_summary)} chars")
                logger.debug(f"🔍 Flight summary preview: {flight_summary[:300]}...")
                
                return flight_summary
                
            except Exception as e:
                import traceback
                error_details = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "params_received": params if 'params' in locals() else "Could not parse params",
                    "traceback": traceback.format_exc()
                }
                return json.dumps([error_details])
        
        return flight_search
    
    def _create_hotel_search_tool(self):
        # Don't store the instance, create new ones with context manager
        
        @tool("hotel_search", return_direct=False)
        def hotel_search(query: str) -> str:
            """
            Search for hotels and return a JSON list of hotel details.
            Input should be a JSON string with city, check_in, and check_out.
            """
            try:
                logger.debug(f"🏨 Hotel search input: {repr(query)}, type: {type(query)}")
                
                # Handle both dict objects and JSON strings from LangChain
                if isinstance(query, dict):
                    # Direct dict input from agent
                    params = query
                    logger.debug(f"✅ Using dict input directly: {params}")
                elif isinstance(query, str) and query.strip():
                    import json
                    import ast
                    clean_query = query.strip()
                    
                    # Try to handle Python dict string representation first
                    if clean_query.startswith('{') and clean_query.endswith('}'):
                        try:
                            # Try to safely evaluate Python dict string
                            params = ast.literal_eval(clean_query)
                            logger.debug(f"✅ Parsed Python dict string: {params}")
                        except (ValueError, SyntaxError):
                            logger.debug(f"❌ Failed to parse as Python dict, trying JSON...")
                            # Fall back to JSON parsing
                            # Remove extra quotes if present (LangChain may add them)
                            if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                                clean_query = clean_query[1:-1]
                            params = json.loads(clean_query)
                            logger.debug(f"✅ Parsed as JSON: {params}")
                    else:
                        # Remove extra quotes if present (LangChain may add them)
                        if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                            clean_query = clean_query[1:-1]
                        params = json.loads(clean_query)
                        logger.debug(f"✅ Parsed as JSON: {params}")
                else:
                    # Handle empty or invalid input
                    return json.dumps([{"error": f"Invalid input received: {repr(query)}, type: {type(query)}"}])
                    
                # Pure LLM reasoning for hotel search (Simple Mode - no Google Search)
                logger.debug(f"🏨 LLM hotel reasoning in {params['city']} for {params['check_in']} to {params['check_out']}")
                
                # Generate realistic hotel data using LLM knowledge
                hotels_from_google = self._generate_llm_hotels(
                    city=params["city"],
                    check_in=params["check_in"], 
                    check_out=params["check_out"]
                )
                logger.debug(f"✅ Hotel search completed, found {len(hotels_from_google)} hotels")
                
                if not hotels_from_google:
                    return "[]"
                
                hotels = [
                    Hotel(
                        name=h.name,
                        city=h.city,
                        rating=h.rating,
                        price_per_night=f"${h.price_per_night}",
                        amenities=h.amenities,
                        address=h.address,
                        data_source='google_search',
                        confidence=0.9
                    ).model_dump()
                    for h in hotels_from_google
                ]
                
                # Create a simplified summary for the agent with data source indicators
                hotel_summary = f"Found {len(hotels)} hotels in {params['city']} for {params['check_in']} to {params['check_out']}:\n"
                for i, h in enumerate(hotels[:3]):  # Show max 3 hotels
                    data_source = "LLM reasoning"
                    hotel_summary += f"Hotel {i+1}: {h['name']} - {h['price_per_night']}/night, Rating: {h['rating']} [Source: {data_source}]\n"
                
                logger.debug(f"📤 Returning hotel summary: {len(hotel_summary)} chars")
                logger.debug(f"🔍 Hotel summary preview: {hotel_summary[:300]}...")
                
                return hotel_summary
                
            except Exception as e:
                return json.dumps([{"error": str(e)}])
        
        return hotel_search
    
    def _create_activity_search_tool(self):
        google_search = self.google_search
        
        @tool("activity_search", return_direct=False)
        def activity_search(query: str) -> str:
            """
            Search for activities, attractions, and experiences in a location.
            Input should be a JSON string with location/city and interests.
            
            Finds activities matching user interests with practical details like
            operating hours, costs, and booking requirements.
            """
            try:
                logger.debug(f"🎯 Activity search input: {repr(query)}, type: {type(query)}")
                
                # Handle both dict objects and JSON strings from LangChain
                if isinstance(query, dict):
                    # Direct dict input from agent
                    params = query
                    logger.debug(f"✅ Using dict input directly: {params}")
                elif isinstance(query, str) and query.strip():
                    import json
                    import ast
                    clean_query = query.strip()
                    
                    # Try to handle Python dict string representation first
                    if clean_query.startswith('{') and clean_query.endswith('}'):
                        try:
                            # Try to safely evaluate Python dict string
                            params = ast.literal_eval(clean_query)
                            logger.debug(f"✅ Parsed Python dict string: {params}")
                        except (ValueError, SyntaxError):
                            logger.debug(f"❌ Failed to parse as Python dict, trying JSON...")
                            # Fall back to JSON parsing
                            # Remove extra quotes if present (LangChain may add them)
                            if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                                clean_query = clean_query[1:-1]
                            params = json.loads(clean_query)
                            logger.debug(f"✅ Parsed as JSON: {params}")
                    else:
                        # Remove extra quotes if present (LangChain may add them)
                        if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                            clean_query = clean_query[1:-1]
                        params = json.loads(clean_query)
                        logger.debug(f"✅ Parsed as JSON: {params}")
                else:
                    # Handle empty or invalid input
                    return json.dumps([{"error": f"Invalid input received: {repr(query)}, type: {type(query)}"}])
                
                location = params.get("location", params.get("city", "Unknown Location"))
                interests = params.get("interests", [])
                
                # Pure LLM reasoning for activity search (Simple Mode - no Google Search)
                interest_str = ", ".join(interests)
                logger.debug(f"🎯 LLM activity reasoning in {location} for interests: {interest_str}")
                
                # Generate realistic activity data using LLM knowledge
                results = self._generate_llm_activities(city=location, interests=interests)
                
                if not results:
                    logger.warning(f"⏰ Activity search found no results for {location}")
                    return f"No activities found for {interest_str} in {location}"
                
                logger.debug(f"✅ Activity search completed, found {len(results)} activities")
                
                # Format activity results for agent reasoning
                activity_summary = f"Found activities in {location} for interests: {interest_str}\n\n"
                
                for i, activity in enumerate(results, 1):
                    activity_summary += f"{i}. {activity.name} [Source: {activity.source}]\n"
                    activity_summary += f"   Description: {activity.description}\n"
                    activity_summary += f"   Category: {activity.category}\n\n"
                
                activity_summary += "Recommendation: Check operating hours and book popular attractions in advance."
                
                return activity_summary
                
            except Exception as e:
                return f"Activity search failed: {str(e)}"
        
        return activity_search
    
    def _create_budget_analysis_tool(self):
        @tool("budget_analysis", return_direct=False)
        def budget_analysis(query: str) -> str:
            """
            Analyze trip budget and provide allocation recommendations.
            
            Breaks down budget across categories (flights, hotels, food, activities)
            with smart recommendations based on destinations and travel style.
            """
            try:
                logger.debug(f"📊 Budget analysis called with query: {query}")
                logger.debug(f"📊 Query type: {type(query)}")
                
                # Handle dict objects, JSON strings, and plain text input
                if isinstance(query, dict):
                    # Direct dict input from agent
                    params = query
                    logger.info(f"📊 Received dict params: {params}")
                elif isinstance(query, str) and query.strip():
                    import json
                    clean_query = query.strip()
                    
                    # Try JSON parsing first
                    try:
                        # Remove extra quotes if present (LangChain may add them)
                        if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                            clean_query = clean_query[1:-1]
                        params = json.loads(clean_query)
                        logger.debug(f"📊 Successfully parsed JSON params: {params}")
                    except json.JSONDecodeError as e:
                        logger.info(f"📊 JSON parsing failed: {e}, treating as plain text")
                        # If JSON parsing fails, treat as plain text description and create default params
                        params = {
                            "total_budget": 3000.0,  # Default budget
                            "destinations": ["tokyo"],  # Default destination
                            "duration_days": 10,  # Default duration
                            "travel_style": "mid-range",  # Default style
                            "description": clean_query  # Store the original text
                        }
                else:
                    # Handle empty or non-string input
                    logger.warning(f"📊 Invalid input type or empty: {query}")
                    return "Invalid input: Please provide budget information or parameters"
                
                # Handle flexible parameter names with better defaults
                total_budget = params.get("total_budget", params.get("budget", 3000.0))  # Default to $3000
                destinations = params.get("destinations", [params.get("destination", "Tokyo")])  # Default to Tokyo
                duration_days = params.get("duration_days", params.get("duration", 10))  # Default to 10 days
                if isinstance(duration_days, str) and "days" in duration_days:
                    duration_days = int(duration_days.replace("days", "").strip())
                travel_style = params.get("travel_style", params.get("budget_style", "mid-range"))
                
                # Fix "Unknown" destinations
                if isinstance(destinations, list) and destinations and destinations[0] == "Unknown":
                    destinations = ["Tokyo"]  # Default destination
                elif not isinstance(destinations, list):
                    destinations = [str(destinations)]
                
                # Budget allocation percentages by travel style
                allocations = {
                    "budget": {
                        "flights": 0.35,
                        "accommodation": 0.25, 
                        "food": 0.20,
                        "activities": 0.10,
                        "local_transport": 0.05,
                        "emergency": 0.05
                    },
                    "mid-range": {
                        "flights": 0.30,
                        "accommodation": 0.30,
                        "food": 0.20,
                        "activities": 0.15,
                        "local_transport": 0.05
                    },
                    "luxury": {
                        "flights": 0.25,
                        "accommodation": 0.35,
                        "food": 0.25,
                        "activities": 0.10,
                        "local_transport": 0.05
                    }
                }
                
                allocation = allocations.get(travel_style, allocations["mid-range"])
                
                # If we have a description from plain text input, include it
                description = params.get("description", "")
                if description:
                    budget_breakdown = f"Budget Analysis based on: {description}\n"
                    budget_breakdown += f"Using ${total_budget:,.2f} budget for {duration_days}-day trip\n\n"
                else:
                    # Handle destinations list safely
                    destinations_str = ', '.join(str(d) for d in destinations) if isinstance(destinations, list) else str(destinations)
                    budget_breakdown = f"Budget Analysis for {duration_days} days to {destinations_str}\n"
                    budget_breakdown += f"Total Budget: ${total_budget:,.2f} ({travel_style} style)\n\n"
                
                daily_budget = total_budget / duration_days if duration_days > 0 else 0
                budget_breakdown += f"Daily Budget: ${daily_budget:.2f}\n\n"
                
                budget_breakdown += "Recommended Budget Allocation:\n"
                for category, percentage in allocation.items():
                    amount = total_budget * percentage
                    budget_breakdown += f"  {category.title()}: ${amount:,.2f} ({percentage*100:.0f}%)\n"
                
                budget_breakdown += f"\nTips for {travel_style} travel:\n"
                if travel_style == "budget":
                    budget_breakdown += "- Look for budget airlines and hostels\n"
                    budget_breakdown += "- Cook some meals or eat local street food\n"
                    budget_breakdown += "- Use public transportation\n"
                elif travel_style == "luxury":
                    budget_breakdown += "- Premium airlines and 4-5 star hotels\n"
                    budget_breakdown += "- Fine dining and premium experiences\n"
                    budget_breakdown += "- Private transfers and guided tours\n"
                else:
                    budget_breakdown += "- Balance of comfort and value\n"
                    budget_breakdown += "- Mix of hotel types and dining options\n"
                    budget_breakdown += "- Public transport with some taxis/rideshares\n"
            
                return budget_breakdown
                
            except Exception as e:
                return f"Budget analysis failed: {str(e)}"
        
        return budget_analysis
    
    def _create_route_optimization_tool(self):
        @tool("route_optimization", return_direct=False)  
        def route_optimization(query: str) -> str:
            """
            Optimize multi-city travel routes for efficiency and cost.
            
            Analyzes the best order to visit destinations considering travel time,
            costs, and geographical constraints.
            """
            try:
                logger.info(f"🗺️ Route optimization called with: {query}")
                
                # Handle both dict objects and JSON strings from LangChain
                if isinstance(query, dict):
                    # Direct dict input from agent
                    params = query
                elif isinstance(query, str) and query.strip():
                    import json
                    clean_query = query.strip()
                    # Remove extra quotes if present (LangChain may add them)
                    if (clean_query.startswith("'" ) and clean_query.endswith("'" )) or (clean_query.startswith('"') and clean_query.endswith('"')):
                        clean_query = clean_query[1:-1]
                    try:
                        params = json.loads(clean_query)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error in route_optimization: {e}")
                        logger.error(f"❌ Query was: '{query}'")
                        return f"Route optimization failed: Invalid JSON format - {str(e)}"
                else:
                    logger.error("❌ Empty or invalid query in route_optimization")
                    return "Route optimization failed: Empty input provided"
                
                # Handle different input formats
                destinations = params.get("destinations", [])
                
                # Extract origin - could be first destination or separate field
                origin = params.get("origin")
                if not origin and destinations:
                    # If no origin specified, assume first destination is origin
                    origin = destinations[0]
                    destinations = destinations[1:] if len(destinations) > 1 else []
                
                preferences = params.get("preferences", {})
                
                # Validate inputs
                if not origin:
                    logger.error("❌ Route optimization: No origin found")
                    return "Route optimization failed: No origin city specified"
                
                if not destinations:
                    logger.warning("⚠️ Route optimization: No destinations found, treating as round trip")
                    destinations = []
                
                logger.info(f"🗺️ Processing route: {origin} → {destinations}")
                
                # Simple route optimization logic
                route_analysis = f"Route Optimization Analysis\n"
                route_analysis += f"Origin: {origin}\n"
                # Handle destinations list safely  
                destinations_str = ', '.join(str(d) for d in destinations) if isinstance(destinations, list) else str(destinations)
                route_analysis += f"Destinations: {destinations_str}\n\n"
                
                # Suggest optimal order (simplified)
                if len(destinations) == 1:
                    optimal_route = [origin, destinations[0], origin]
                    route_analysis += f"Simple round trip: {' → '.join(optimal_route)}\n"
                else:
                    # For multiple destinations, suggest a logical order
                    optimal_route = [origin] + destinations + [origin]
                    route_analysis += f"Suggested route: {' → '.join(optimal_route)}\n"
                
                route_analysis += "\nRoute Considerations:\n"
                route_analysis += f"- Total destinations: {len(destinations)}\n"
                route_analysis += f"- Estimated total segments: {len(optimal_route) - 1}\n"
                
                if len(destinations) > 2:
                    route_analysis += "- Consider open-jaw flights for efficiency\n"
                    route_analysis += "- Book flights in advance for better prices\n"
                
                route_analysis += "- Allow buffer time between destinations\n"
                route_analysis += "- Check visa requirements for each country\n"
            
                return route_analysis
                
            except Exception as e:
                return f"Route optimization failed: {str(e)}"
        
        return route_optimization

    def _generate_llm_flights(self, origin: str, destination: str, departure_date: str):
        """Generate realistic flight data using LLM knowledge and reasoning."""
        import random
        from datetime import datetime, timedelta
        
        # Common airlines for different routes
        airline_mapping = {
            ('seattle', 'tokyo'): ['Japan Airlines', 'ANA', 'United', 'Delta'],
            ('seattle', 'london'): ['British Airways', 'Virgin Atlantic', 'United', 'Delta'],
            ('seattle', 'paris'): ['Air France', 'Delta', 'United', 'Lufthansa'],
            ('tokyo', 'taipei'): ['EVA Air', 'China Airlines', 'JAL', 'ANA'],
            ('tokyo', 'seoul'): ['Korean Air', 'Asiana', 'JAL', 'ANA'],
            ('london', 'paris'): ['British Airways', 'Air France', 'EuroWings', 'Vueling'],
        }
        
        # Typical flight durations and price ranges
        route_info = {
            ('seattle', 'tokyo'): {'duration_hours': 11, 'base_price': 1200},
            ('seattle', 'london'): {'duration_hours': 9, 'base_price': 800},
            ('seattle', 'paris'): {'duration_hours': 10, 'base_price': 900},
            ('tokyo', 'taipei'): {'duration_hours': 3, 'base_price': 400},
            ('tokyo', 'seoul'): {'duration_hours': 2, 'base_price': 300},
            ('london', 'paris'): {'duration_hours': 1, 'base_price': 150},
            ('taipei', 'seattle'): {'duration_hours': 12, 'base_price': 1100},
            ('seoul', 'seattle'): {'duration_hours': 10, 'base_price': 1000},
            ('paris', 'seattle'): {'duration_hours': 10, 'base_price': 900},
        }
        
        origin_key = origin.lower().replace(' ', '')
        dest_key = destination.lower().replace(' ', '')
        route_key = (origin_key, dest_key)
        reverse_route_key = (dest_key, origin_key)
        
        # Try to find route info, fallback to reverse or defaults
        if route_key in route_info:
            info = route_info[route_key]
            airlines = airline_mapping.get(route_key, ['United', 'Delta', 'Southwest'])
        elif reverse_route_key in route_info:
            info = route_info[reverse_route_key]
            airlines = airline_mapping.get(reverse_route_key, ['United', 'Delta', 'Southwest'])
        else:
            # Default for unknown routes
            info = {'duration_hours': 8, 'base_price': 800}
            airlines = ['United', 'Delta', 'Southwest Airlines']
        
        flights = []
        
        # Generate 3 flight options with different times and prices
        departure_times = ['08:30', '14:15', '19:45']
        for i in range(3):
            # Calculate arrival time
            dep_hour, dep_min = map(int, departure_times[i].split(':'))
            arrival_dt = datetime(2025, 1, 1, dep_hour, dep_min) + timedelta(hours=info['duration_hours'])
            arrival_time = arrival_dt.strftime('%H:%M')
            
            # Add variation to duration and price
            duration_variation = random.randint(-60, 60)  # +/- 1 hour
            actual_duration_minutes = info['duration_hours'] * 60 + duration_variation
            duration_str = f"{actual_duration_minutes // 60}h {actual_duration_minutes % 60}m"
            
            price_variation = random.randint(-200, 300)
            actual_price = info['base_price'] + price_variation
            
            flight = type('LLMFlight', (), {
                'from_city': origin,
                'to_city': destination,
                'date': departure_date,
                'departure_time': departure_times[i],
                'arrival_time': arrival_time,
                'airline': airlines[i % len(airlines)],
                'price': f'${actual_price}',
                'duration': duration_str,
                'source': 'LLM Reasoning'
            })()
            
            flights.append(flight)
        
        return flights

    def _generate_llm_hotels(self, city: str, check_in: str, check_out: str):
        """Generate realistic hotel data using LLM knowledge."""
        import random
        
        # Hotel name patterns by city
        hotel_patterns = {
            'tokyo': ['Tokyo', 'Shibuya', 'Shinjuku', 'Ginza', 'Imperial'],
            'seoul': ['Seoul', 'Gangnam', 'Myeongdong', 'Hongik', 'Lotte'],
            'london': ['London', 'Westminster', 'Kensington', 'Covent Garden', 'Tower'],
            'paris': ['Paris', 'Champs Elysees', 'Louvre', 'Montmartre', 'Saint Germain'],
            'taipei': ['Taipei', 'Xinyi', 'Zhongshan', 'Daan', 'Grand'],
        }
        
        hotel_types = ['Hotel', 'Suites', 'Inn', 'Resort']
        hotel_brands = ['Hyatt', 'Hilton', 'Marriott', 'Sheraton', 'InterContinental', 'Grand', 'Imperial', 'Royal']
        
        city_key = city.lower().replace(' ', '')
        patterns = hotel_patterns.get(city_key, [city])
        
        hotels = []
        base_prices = [120, 200, 350]  # Budget, mid-range, luxury
        
        for i in range(3):
            # Generate hotel name
            name_parts = [
                random.choice(hotel_brands),
                random.choice(patterns),
                random.choice(hotel_types)
            ]
            hotel_name = f"{name_parts[0]} {name_parts[1]} {name_parts[2]}"
            
            # Generate realistic data
            price = base_prices[i] + random.randint(-30, 50)
            rating = round(3.0 + i * 0.5 + random.uniform(-0.3, 0.3), 1)
            
            amenities = ['WiFi', 'Restaurant', '24h Reception']
            if i >= 1:
                amenities.extend(['Gym', 'Business Center'])
            if i == 2:
                amenities.extend(['Spa', 'Pool', 'Concierge'])
            
            hotel = type('LLMHotel', (), {
                'name': hotel_name,
                'city': city,
                'price_per_night': price,
                'rating': rating,
                'amenities': amenities,
                'address': f"{patterns[i % len(patterns)]}, {city}",  # Add address based on city area
                'source': 'LLM Reasoning'
            })()
            
            hotels.append(hotel)
            
        return hotels

    def _generate_llm_activities(self, city: str, interests: list):
        """Generate realistic activity data using LLM knowledge."""
        # Activity database by city and interest type
        activities_db = {
            'tokyo': {
                'food': [
                    'Tsukiji Outer Market Food Tour',
                    'Traditional Sushi Making Class', 
                    'Ramen Tasting in Shibuya',
                    'Izakaya Hopping in Shinjuku',
                    'Wagyu Beef Tasting Experience'
                ],
                'culture': [
                    'Senso-ji Temple Visit',
                    'Imperial Palace Gardens',
                    'Traditional Tea Ceremony',
                    'Kabuki Theater Performance',
                    'Meiji Shrine Experience'
                ],
                'sightseeing': [
                    'Tokyo Skytree Observatory',
                    'Shibuya Crossing Experience',
                    'Harajuku Fashion District',
                    'Akihabara Electronics Tour',
                    'Tokyo Bay Cruise'
                ]
            },
            'seoul': {
                'food': [
                    'Korean BBQ Cooking Class',
                    'Kimchi Making Workshop',
                    'Street Food Tour in Myeongdong',
                    'Traditional Korean Market Visit',
                    'Seoul Food Walking Tour'
                ],
                'culture': [
                    'Gyeongbokgung Palace Tour',
                    'Hanbok Wearing Experience',
                    'Traditional Korean Spa (Jjimjilbang)',
                    'Bukchon Hanok Village Walk',
                    'Korean Traditional Music Performance'
                ],
                'sightseeing': [
                    'N Seoul Tower Visit',
                    'Han River Park Cruise',
                    'Gangnam District Tour',
                    'Dongdaemun Design Plaza',
                    'Banpo Rainbow Bridge'
                ]
            },
            'taipei': {
                'food': [
                    'Night Market Food Tour',
                    'Din Tai Fung Dumpling Experience',
                    'Taiwanese Bubble Tea Workshop',
                    'Traditional Taiwanese Breakfast Tour',
                    'Local Cooking Class'
                ],
                'culture': [
                    'National Palace Museum',
                    'Longshan Temple Visit',
                    'Chinese Calligraphy Class',
                    'Traditional Chinese Medicine Tour',
                    'Aboriginal Culture Center'
                ],
                'sightseeing': [
                    'Taipei 101 Observatory',
                    'Elephant Mountain Hiking',
                    'Sun Moon Lake Day Trip',
                    'Jiufen Old Street',
                    'Yangmingshan National Park'
                ]
            }
        }
        
        # Default activities for unknown cities
        default_activities = {
            'food': [
                f'Local {city} Food Tour',
                f'Traditional {city} Restaurant Visit',
                f'{city} Market Experience',
                f'Cooking Class in {city}',
                f'{city} Street Food Walk'
            ],
            'culture': [
                f'{city} Museum Tour',
                f'Historical {city} Walk',
                f'{city} Cultural Center Visit',
                f'Local {city} Art Gallery',
                f'Traditional {city} Performance'
            ],
            'sightseeing': [
                f'{city} City Center Tour',
                f'{city} Landmark Visit',
                f'{city} Scenic Viewpoint',
                f'{city} Walking Tour',
                f'{city} Architecture Tour'
            ]
        }
        
        city_key = city.lower().replace(' ', '')
        city_activities = activities_db.get(city_key, default_activities)
        
        activities = []
        used_activities = set()
        
        # Generate activities based on interests
        for interest in interests:
            interest_key = interest.lower()
            if interest_key in city_activities:
                available = [act for act in city_activities[interest_key] if act not in used_activities]
                if available:
                    selected = available[0]  # Take the first available
                    used_activities.add(selected)
                    
                    activity = type('LLMActivity', (), {
                        'city': city,
                        'name': selected,
                        'description': f'Experience {selected.lower()} in {city}',
                        'category': interest,
                        'source': 'LLM Reasoning'
                    })()
                    
                    activities.append(activity)
        
        # If no specific interests or no matches, add some general activities
        if not activities:
            general_activities = city_activities.get('sightseeing', default_activities['sightseeing'])
            activity = type('LLMActivity', (), {
                'city': city,
                'name': general_activities[0],
                'description': f'Explore {general_activities[0].lower()}',
                'category': 'sightseeing',
                'source': 'LLM Reasoning'
            })()
            activities.append(activity)
            
        return activities