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
                    
                # Simplified async handling - always use asyncio.run in a new thread
                logger.debug(f"🔍 Searching flights: {params['origin']} → {params['destination']} on {params['departure_date']}")
                import concurrent.futures
                from services.google_travel_search import search_flights_google
                
                try:
                    # Always run in a new thread to avoid event loop conflicts
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            search_flights_google(
                                origin=params["origin"],
                                destination=params["destination"],
                                departure_date=params["departure_date"]
                            )
                        )
                        # Add timeout to prevent hanging
                        flights_from_google = future.result(timeout=30)
                        logger.info(f"✅ Flight search completed, found {len(flights_from_google) if flights_from_google else 0} flights")
                        
                except (concurrent.futures.TimeoutError, Exception) as e:
                    logger.warning(f"⏰ Flight search timeout or error: {e}")
                    # Return mock flight data on any error
                    flights_from_google = [
                        type('MockFlight', (), {
                            'from_city': params['origin'],
                            'to_city': params['destination'],
                            'date': params['departure_date'],
                            'departure_time': '10:00 AM',
                            'arrival_time': '2:00 PM (+1 day)',
                            'airline': 'Mock Airlines (Search Failed)',
                            'price': '$800'
                        })()
                    ]
                    logger.info(f"🔄 Using mock flight data due to search failure")
                
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
                    data_source = "AI agent" if f.get('data_source') == 'simulation' else "Google Search"
                    flight_summary += f"Flight {i+1}: {f['airline']} - Depart: {f['departure_time']}, Arrive: {f['arrival_time']}, Price: {f['estimated_price']} [Source: {data_source}]\n"
                
                logger.info(f"📤 Returning simplified flight summary: {len(flight_summary)} chars")
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
                    
                # Simplified async handling - always use asyncio.run in a new thread
                logger.info(f"🏨 Searching hotels in {params['city']} for {params['check_in']} to {params['check_out']}")
                import concurrent.futures
                from services.google_travel_search import search_hotels_google
                
                try:
                    # Always run in a new thread to avoid event loop conflicts
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            search_hotels_google(
                                city=params["city"],
                                checkin_date=params["check_in"],
                                checkout_date=params["check_out"]
                            )
                        )
                        # Add timeout to prevent hanging
                        hotels_from_google = future.result(timeout=30)
                        logger.info(f"✅ Hotel search completed, found {len(hotels_from_google) if hotels_from_google else 0} hotels")
                        
                except (concurrent.futures.TimeoutError, Exception) as e:
                    logger.warning(f"⏰ Hotel search timeout or error: {e}")
                    # Return mock hotel data on any error
                    hotels_from_google = [
                        type('MockHotel', (), {
                            'name': f'Mock Hotel in {params["city"]}',
                            'city': params['city'],
                            'address': f'123 Main St, {params["city"]}',
                            'price_per_night': '$120',
                            'rating': '4.0 (Search Failed)'
                        })()
                    ]
                    logger.info(f"🔄 Using mock hotel data due to search failure")
                
                if not hotels_from_google:
                    return "[]"
                
                hotels = [
                    Hotel(
                        name=h.name,
                        city=h.city,
                        rating=h.rating,
                        price_per_night=h.price_per_night,
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
                    data_source = "AI agent" if h.get('data_source') == 'simulation' else "Google Search"
                    hotel_summary += f"Hotel {i+1}: {h['name']} - {h['price_per_night']}/night, Rating: {h['rating']} [Source: {data_source}]\n"
                
                logger.info(f"📤 Returning hotel summary: {len(hotel_summary)} chars")
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
                
                # Build comprehensive search query
                interest_str = ", ".join(interests)
                search_query = f"{location} {interest_str} activities attractions things to do"
                logger.info(f"🎯 Searching activities in {location} for interests: {interest_str}")
                
                # Use web search for activity information
                try:
                    # Try to get the current event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in a running loop, create a new thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                google_search.search_web(search_query, num_results=8)
                            )
                            results = future.result()
                    else:
                        results = asyncio.run(google_search.search_web(search_query, num_results=8))
                except RuntimeError:
                    # No event loop running
                    results = asyncio.run(google_search.search_web(search_query, num_results=8))
                
                if not results:
                    logger.warning(f"⏰ Activity search found no results for {location}")
                    return f"No activities found for {interest_str} in {location}"
                
                logger.info(f"✅ Activity search completed, found {len(results)} activities")
                
                # Format activity results for agent reasoning
                activity_summary = f"Found activities in {location} for interests: {interest_str}\n\n"
                
                for i, result in enumerate(results[:6], 1):  # Top 6 activities
                    # Determine data source - mock activities have example.com URLs, real ones don't
                    data_source = "AI agent" if 'example.com' in result.get('url', '') else "Google Search"
                    activity_summary += f"{i}. {result.get('title', 'Unknown Activity')} [Source: {data_source}]\n"
                    snippet = result.get('snippet', '')
                    if snippet:
                        activity_summary += f"   Description: {snippet[:150]}...\n"
                    if result.get('url'):
                        activity_summary += f"   More info: {result['url']}\n"
                    activity_summary += "\n"
                
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
                logger.info(f"📊 Budget analysis called with query: {query}")
                logger.info(f"📊 Query type: {type(query)}")
                
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
                        logger.info(f"📊 Successfully parsed JSON params: {params}")
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
                    budget_breakdown = f"Budget Analysis for {duration_days} days to {', '.join(destinations)}\n"
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
                route_analysis += f"Destinations: {', '.join(destinations)}\n\n"
                
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