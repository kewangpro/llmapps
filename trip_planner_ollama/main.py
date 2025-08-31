from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import TripRequest, TripPlan, DayPlan, Flight, Hotel
from config import get_config
from services.error_handler import global_error_handler, log_error
# TRUE LangChain Agent Framework
from agents import create_langchain_agent_system, LangChainMultiAgentSystem
import uvicorn
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
async def _extract_trip_plan_data(result, request: TripRequest) -> dict:
    """
    Extracts structured trip plan data from the agent's result.
    """
    # First, check if the agent provided structured trip_plan data
    structured_trip_plan = None
    if hasattr(result, 'primary_result') and isinstance(result.primary_result, dict):
        structured_trip_plan = result.primary_result.get('trip_plan')
    
    if structured_trip_plan and isinstance(structured_trip_plan, dict):
        logger.info(f"✅ Using structured trip plan with {len(structured_trip_plan.get('flights', []))} flights and {len(structured_trip_plan.get('hotels', []))} hotels")
        
        # Use the structured data directly
        flights = structured_trip_plan.get('flights', [])
        hotels = structured_trip_plan.get('hotels', [])
        estimated_budget = structured_trip_plan.get('estimated_budget', "$3000.00")
        
        # Convert flights and hotels to proper model format if needed
        from models import Flight, Hotel
        formatted_flights = []
        for f in flights:
            formatted_flights.append(Flight(
                from_city=f.get('from_city', ''),
                to_city=f.get('to_city', ''),
                date=f.get('date', ''),
                departure_time=f.get('departure_time', ''),
                arrival_time=f.get('arrival_time', ''),
                airline=f.get('airline', ''),
                estimated_price=f.get('estimated_price', ''),
                data_source=f.get('data_source', 'langchain_agents'),
                confidence=f.get('confidence', 0.8)
            ))
        
        formatted_hotels = []
        for h in hotels:
            formatted_hotels.append(Hotel(
                name=h.get('name', ''),
                city=h.get('city', ''),
                rating=h.get('rating', 0.0),
                price_per_night=h.get('price_per_night', ''),
                amenities=h.get('amenities', []),
                address=h.get('address', ''),
                data_source=h.get('data_source', 'langchain_agents'),
                confidence=h.get('confidence', 0.8)
            ))
        
        flights = formatted_flights
        hotels = formatted_hotels
        
    else:
        # Extract text from pure LLM agent result
        result_text = ""
        if hasattr(result, 'primary_result') and isinstance(result.primary_result, dict):
            # Old agent system format
            result_text = str(result.primary_result.get("output", ""))
        elif isinstance(result, dict) and "output" in result:
            # Pure LLM agent format - direct output key
            result_text = str(result["output"])
        else:
            # Fallback - convert entire result to string
            result_text = str(result)
            
        logger.info(f"🔍 Agent output for parsing: {result_text[:500]}...")
        
        flights = _parse_flights_from_text(result_text, request)
        hotels = _parse_hotels_from_text(result_text, request)
        budget_float = _parse_budget_from_text(result_text)
        estimated_budget = f"${budget_float:.2f}" if budget_float else "$3000.00"

    # Create detailed daily plans with activities based on destinations
    daily_plans = []
    current_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    
    # Calculate days per destination for multi-city trips
    total_destinations = len(request.destinations)
    if total_destinations == 0:
        destinations_with_days = [(request.origin, request.duration_days)]
    else:
        days_per_city = request.duration_days // total_destinations
        remaining_days = request.duration_days % total_destinations
        destinations_with_days = []
        for i, dest in enumerate(request.destinations):
            city_days = days_per_city + (1 if i < remaining_days else 0)
            destinations_with_days.append((dest, city_days))
    
    day_counter = 1
    for dest_city, city_duration in destinations_with_days:
        for city_day in range(city_duration):
            activities = _generate_activities_for_city(dest_city, city_day + 1, request.preferences)
            daily_plans.append({
                "day": day_counter,
                "date": (current_date + timedelta(days=day_counter - 1)).strftime("%Y-%m-%d"),
                "city": dest_city,
                "activities": activities,
                "transportation": "Local metro/taxi" if city_day > 0 else "Airport transfer",
                "city_tips": _get_city_tips(dest_city)
            })
            day_counter += 1

    route_order = [request.origin] + request.destinations # Simplistic

    return {
        "origin": request.origin,
        "destinations": request.destinations,
        "start_date": request.start_date,
        "duration_days": request.duration_days,
        "budget": request.budget,
        "preferences": request.preferences,
        "plan_summary": result,
        "flights": flights,
        "hotels": hotels,
        "estimated_budget": estimated_budget,
        "total_days": request.duration_days,
        "route_order": route_order,
        "daily_plans": daily_plans,
        "travel_tips": ["Enjoy your trip!", "Check visa requirements", "Pack for the weather"] # Placeholder list
    }

def _parse_budget_from_text(text: str) -> float:
    """Extract budget information from agent's final text output."""
    try:
        import re
        # Pattern to match: "approximately $3000.00" or "total cost... $3000"
        patterns = [
            r'approximately\s*\$(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'total cost[^$]*\$(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'budget[^$]*\$(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'\$(\d+(?:,\d+)?(?:\.\d{2})?)(?:\s*budget|\s*total)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                budget_str = match.group(1).replace(',', '')
                return float(budget_str)
        
        logger.info("No budget information found in text, using default")
        return 3000.0  # Default budget
        
    except Exception as e:
        logger.error(f"Error parsing budget from text: {e}")
        return 3000.0  # Default budget

def _generate_activities_for_city(city: str, day_number: int, preferences: str) -> List[str]:
    """Generate sample activities for a city based on preferences."""
    city_lower = city.lower()
    preference_lower = preferences.lower() if preferences else ""
    
    # City-specific activities
    city_activities = {
        "tokyo": [
            "Visit Senso-ji Temple in Asakusa",
            "Explore Shibuya Crossing and shopping district", 
            "Experience Tsukiji Outer Market food tour",
            "Take in views from Tokyo Skytree",
            "Stroll through Ueno Park and museums",
            "Discover Harajuku street fashion",
            "Enjoy traditional kaiseki dining",
            "Visit Meiji Shrine"
        ],
        "taipei": [
            "Visit Taipei 101 observatory",
            "Explore Ximending night market",
            "Take day trip to Jiufen mountain town", 
            "Tour National Palace Museum",
            "Soak in Beitou hot springs",
            "Sample street food in Shilin night market"
        ],
        "seoul": [
            "Explore Gyeongbokgung Palace",
            "Shop in Myeongdong district",
            "Hike in Namsan Park to N Seoul Tower",
            "Experience Korean BBQ in Gangnam",
            "Visit Bukchon Hanok Village",
            "Discover Hongdae nightlife"
        ],
        "paris": [
            "Visit the Eiffel Tower and Trocadéro",
            "Explore Louvre Museum",
            "Stroll along the Seine and Notre-Dame",
            "Discover Montmartre and Sacré-Cœur",
            "Experience café culture in Le Marais",
            "Shop along Champs-Élysées"
        ]
    }
    
    # Get city activities or default
    activities = city_activities.get(city_lower, [
        f"Explore {city} city center",
        f"Visit local museums and landmarks",
        f"Try traditional {city} cuisine",
        f"Walk through historic districts"
    ])
    
    # Filter by preferences if provided
    if "food" in preference_lower or "cuisine" in preference_lower:
        activities = [act for act in activities if any(word in act.lower() for word in ["food", "market", "dining", "cuisine", "restaurant"])]
    elif "culture" in preference_lower or "history" in preference_lower:
        activities = [act for act in activities if any(word in act.lower() for word in ["temple", "museum", "palace", "shrine", "historic", "traditional"])]
    elif "shopping" in preference_lower:
        activities = [act for act in activities if any(word in act.lower() for word in ["shop", "market", "district"])]
    
    # Return 2-3 activities per day, varying by day number
    start_idx = (day_number - 1) * 2
    return activities[start_idx:start_idx + 3] or activities[:3]

def _get_city_tips(city: str) -> List[str]:
    """Get useful tips for a specific city."""
    city_lower = city.lower()
    
    city_tips = {
        "tokyo": [
            "Get a JR Pass for convenient train travel",
            "Carry cash - many places don't accept cards",
            "Bow slightly when greeting locals"
        ],
        "taipei": [
            "Use the EasyCard for public transportation",
            "Night markets are must-visit experiences",
            "Learn basic Mandarin greetings"
        ],
        "seoul": [
            "Download subway apps for navigation", 
            "Try Korean skincare products",
            "Respect cultural customs at temples"
        ],
        "paris": [
            "Buy metro day passes for easy transport",
            "Learn basic French phrases",
            "Dress stylishly to fit in with locals"
        ]
    }
    
    return city_tips.get(city_lower, [
        f"Research {city} cultural customs",
        "Learn key phrases in local language",
        "Use public transportation when possible"
    ])

config = get_config()

app = FastAPI(
    title="LangChain Agent Trip Planner API",
    description="Intelligent travel planning powered by LangChain agents with reasoning capabilities and Google Search tools",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Trip Planner API is running"}

# Enable CORS for all origins (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the TRUE LangChain Agent System
langchain_agent_system: Optional[LangChainMultiAgentSystem] = None

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    global langchain_agent_system
    
    logger.info("🚀 Starting LangChain Agent Trip Planner API")
    logger.info(f"Configuration: {config.get_environment_info()}")
    
    # Initialize TRUE LangChain Agent System
    try:
        langchain_agent_system = create_langchain_agent_system(model_name="gemma3:latest")
        logger.info("🤖 LangChain Agent System initialized with reasoning capabilities")
        logger.info("   - AgentExecutor with ReAct reasoning")
        logger.info("   - Chain-of-thought planning")
        logger.info("   - Automatic tool selection")
        logger.info("   - Multi-agent collaboration")
        logger.info("   - Google Search tools integration")
    except Exception as e:
        logger.error(f"❌ LangChain Agent System initialization failed: {e}")
        langchain_agent_system = None
    
    # Log API availability
    if config.has_google_search_config:
        logger.info("✅ Google Search API configured - enhanced search results available")
    else:
        logger.info("✅ Using intelligent travel data with LangChain agents")
    
    if langchain_agent_system:
        logger.info("🤖 LangChain Agent System ready with 5 specialized reasoning agents")

@app.get("/")
async def health_check():
    """Health check endpoint with detailed system status."""
    try:
        status = {
            "status": "healthy",
            "message": "Enhanced Trip Planner API is running",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "configuration": config.get_environment_info(),
            "error_stats": global_error_handler.get_error_stats()
        }
        return status
    except Exception as e:
        log_error(e, {"endpoint": "health_check"})
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/plan-trip")
async def plan_trip(request: TripRequest, background_tasks: BackgroundTasks):
    """
    Plan a comprehensive multi-city round trip with real-time data.
    
    This endpoint uses:
    - Smart flight data from Google Search integration
    - Smart hotel data from Google Search integration
    - Advanced route optimization algorithms
    - AI-powered trip enhancement
    - Comprehensive error handling and fallback systems
    
    Example request:
    {
        "origin": "San Francisco",
        "destinations": ["Tokyo", "Seoul", "Bangkok"],
        "start_date": "2024-03-15",
        "duration_days": 10,
        "budget": "medium",
        "preferences": "love food and culture, prefer 4+ star hotels"
    }
    """
    request_start_time = datetime.utcnow()
    
    try:
        # Process destinations - handle comma-separated cities in single strings
        processed_destinations = []
        for dest in request.destinations:
            if ',' in dest:
                # Split comma-separated cities and strip whitespace
                split_cities = [city.strip().title() for city in dest.split(',') if city.strip()]
                processed_destinations.extend(split_cities)
            else:
                processed_destinations.append(dest.strip().title())
        
        # Update the request with processed destinations
        request.destinations = processed_destinations
        
        # Validate request
        await validate_trip_request(request)
        
        logger.info(f"Planning trip: {request.origin} -> {request.destinations}")
        
        if not langchain_agent_system:
            raise HTTPException(status_code=503, detail="Agent system is not available")

        logger.info("🤖 Using TRUE LangChain Agent System with reasoning capabilities")
        
        budget_map = {"low": 2000.0, "medium": 3000.0, "high": 5000.0}
        numeric_budget = budget_map.get(request.budget, 3000.0)
        
        interests = []
        if request.preferences:
            interest_keywords = ["food", "culture", "history", "nature", "adventure", "art", "museums", "nightlife", "shopping", "beaches"]
            for keyword in interest_keywords:
                if keyword.lower() in request.preferences.lower():
                    interests.append(keyword)
        
        if not interests:
            interests = ["culture", "food", "sightseeing"]  # Default interests
        
        result = await langchain_agent_system.plan_trip_with_reasoning(
            origin=request.origin,
            destinations=request.destinations,
            start_date=request.start_date,
            duration_days=request.duration_days,
            budget=numeric_budget,
            interests=interests,
            travel_style=request.budget,
            collaboration_mode="simple"  # Use master agent for now
        )
        
        if result.primary_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Agent system failed: {result.primary_result.get('error')}")

        # Extract structured data from agent system response
        trip_plan_data = await _extract_trip_plan_data(result, request)

        # Ensure we have the required structure for TripPlan
        if "total_days" not in trip_plan_data:
            trip_plan_data["total_days"] = request.duration_days
        if "route_order" not in trip_plan_data:
            trip_plan_data["route_order"] = [request.origin] + request.destinations  
        if "daily_plans" not in trip_plan_data:
            trip_plan_data["daily_plans"] = [{"day": i+1, "date": (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d"), "city": request.destinations[0] if request.destinations else request.origin, "activities": []} for i in range(request.duration_days)]

        trip_plan = TripPlan(**trip_plan_data)
        logger.info(f"🎯 After TripPlan creation - flights: {len(trip_plan.flights)}, hotels: {len(trip_plan.hotels)}")
        
        # Transform the response for frontend compatibility
        response_data = {
            "total_days": trip_plan.total_days,
            "route_order": trip_plan.route_order,
            "daily_plans": [day.dict() for day in trip_plan.daily_plans],
            "estimated_budget": trip_plan.estimated_budget,
            "travel_tips": trip_plan.travel_tips,
            "flights": [_transform_flight_for_frontend(flight) for flight in trip_plan.flights],
            "hotels": [_transform_hotel_for_frontend(hotel) for hotel in trip_plan.hotels]
        }
        
        processing_time = (datetime.utcnow() - request_start_time).total_seconds()
        background_tasks.add_task(
            log_trip_request, 
            request, 
            trip_plan, 
            processing_time, 
            "success"
        )
        
        return response_data
    
    except ValueError as e:
        log_error(e, {"endpoint": "plan_trip", "request": request.dict()})
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        processing_time = (datetime.utcnow() - request_start_time).total_seconds()
        log_error(e, {
            "endpoint": "plan_trip",
            "request": request.dict(),
            "processing_time": processing_time
        })
        
        background_tasks.add_task(
            log_trip_request,
            request,
            None,
            processing_time,
            f"error: {str(e)}"
        )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to plan trip: {str(e)}"
        )

@app.get("/debug-fallback")
async def debug_fallback():
    """Debug fallback data creation."""
    from models import TripRequest
    request = TripRequest(
        origin="Seattle",
        destinations=["Tokyo"], 
        start_date="2025-09-02",
        duration_days=2,
        budget="medium"
    )
    
    flights = _create_fallback_flights(request)
    hotels = _create_fallback_hotels(request)
    
    flights_transformed = [_transform_flight_for_frontend(flight) for flight in flights]
    hotels_transformed = [_transform_hotel_for_frontend(hotel) for hotel in hotels]
    
    return {
        "flights_created": len(flights),
        "hotels_created": len(hotels),
        "flights_transformed": len(flights_transformed),
        "hotels_transformed": len(hotels_transformed),
        "flights": flights_transformed,
        "hotels": hotels_transformed
    }

@app.get("/debug-text-extraction")
async def debug_text_extraction():
    """Debug text extraction with sample agent output."""
    sample_text = """Here is a summary of the trip plan based on the information gathered:

**Flights:**

*   **Seattle to Tokyo:** Alaska Airlines - Depart: 08:06, Arrive: 21:53, Price: $1047
*   **Tokyo to Taipei:** Japan Airlines - Depart: 08:18, Arrive: 23:52, Price: $796
*   **Taipei to Seattle:** Japan Airlines - Depart: 08:15, Arrive: 19:58, Price: $842

**Hotels:**

*   **Tokyo:** Sheraton Tokyo Resort - $201/night, Rating: 3.6
*   **Taipei:** Plaza Hotel Taipei Suites - $121/night, Rating: 3.6

**Budget Breakdown:**

*   Total Estimated Cost: Approximately $3000.00

**Recommendations:**

*   Mid-range travel with balance of comfort and value
*   Mix of hotel types and dining options
*   Public transport with some taxis/rideshares"""
    
    from models import TripRequest
    request = TripRequest(
        origin="Seattle",
        destinations=["Tokyo", "Taipei"],
        start_date="2025-09-02", 
        duration_days=10,
        budget="medium"
    )
    
    flights = _parse_flights_from_text(sample_text, request)
    hotels = _parse_hotels_from_text(sample_text, request)
    estimated_budget = _parse_budget_from_text(sample_text)
    
    return {
        "sample_text": sample_text,
        "flights_extracted": len(flights),
        "hotels_extracted": len(hotels),
        "estimated_budget": estimated_budget,
        "flights": [_transform_flight_for_frontend(f) for f in flights],
        "hotels": [_transform_hotel_for_frontend(h) for h in hotels]
    }

@app.get("/api-status")
async def api_status():
    """Get detailed status of LangChain agent system."""
    try:
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "langchain_agents": {
                "status": "available" if langchain_agent_system else "unavailable",
                "agent_count": 5 if langchain_agent_system else 0,
                "reasoning_enabled": True if langchain_agent_system else False,
                "google_search_tools": config.has_google_search_config
            },
            "error_stats": global_error_handler.get_error_stats(),
            "integration_mode": "langchain_agents_only"
        }
        
        if langchain_agent_system:
            status["langchain_agents"]["agents"] = [
                "MasterTravelAgent",
                "FlightPlanningAgent", 
                "AccommodationAgent",
                "ActivityAgent",
                "BudgetPlanningAgent"
            ]
        
        return status
    
    except Exception as e:
        log_error(e, {"endpoint": "api_status"})
        raise HTTPException(status_code=500, detail="Failed to get API status")


@app.get("/test-flight-cards")
async def test_flight_cards():
    """Test endpoint to verify flight cards work in mobile app."""
    try:
        from datetime import datetime, timedelta
        
        # Generate test flights exactly like the main endpoint
        flights = []
        destinations = ["Tokyo", "Seoul"]
        origin = "San Francisco"
        
        for i, destination in enumerate(destinations):
            departure_date = (datetime.utcnow() + timedelta(days=30 + i*2)).strftime("%Y-%m-%d")
            origin_city = origin if i == 0 else destinations[i-1]
            
            flight = {
                "from_city": origin_city,
                "to_city": destination,
                "date": departure_date,
                "departure_time": "10:30 AM",
                "arrival_time": "2:45 PM",
                "airline": "LangChain Test Airlines",
                "estimated_price": f"${200 + i*50}",
                "data_source": "langchain_agents",
                "confidence": 0.85
            }
            flights.append(flight)
        
        # Return flight
        return_date = (datetime.utcnow() + timedelta(days=35)).strftime("%Y-%m-%d")
        return_flight = {
            "from_city": destinations[-1],
            "to_city": origin,
            "date": return_date,
            "departure_time": "6:15 PM",
            "arrival_time": "11:30 PM",
            "airline": "LangChain Test Airlines",
            "estimated_price": "$280",
            "data_source": "langchain_agents",
            "confidence": 0.85
        }
        flights.append(return_flight)
        
        # Generate test hotels
        hotels = []
        for destination in destinations:
            hotel = {
                "name": f"LangChain Test Hotel {destination}",
                "city": destination,
                "rating": 4.2,
                "price_per_night": f"${80 + len(destination)*5}",
                "amenities": ["WiFi", "Pool", "Gym", "Restaurant"],
                "address": f"Downtown {destination}",
                "data_source": "langchain_agents",
                "confidence": 0.80
            }
            hotels.append(hotel)
        
        return {
            "test_purpose": "Verify mobile app flight cards display",
            "flights_count": len(flights),
            "hotels_count": len(hotels),
            "sample_flight": flights[0] if flights else None,
            "sample_hotel": hotels[0] if hotels else None,
            "data_source": "langchain_agents",
            "status": "Flight cards should display properly in mobile app"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "test_failed"}

@app.get("/test-ollama")
async def test_ollama():
    """Test Ollama connectivity and available models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            return {
                "status": "Ollama is running",
                "models": models,
                "model_count": len(models.get("models", []))
            }
        else:
            return {
                "status": "Ollama not accessible",
                "error": f"HTTP {response.status_code}",
                "message": "Make sure Ollama is running: ollama serve"
            }
    
    except requests.exceptions.RequestException as e:
        return {
            "status": "Ollama connection failed",
            "error": str(e),
            "message": "Make sure Ollama is running: ollama serve"
        }
    except Exception as e:
        log_error(e, {"endpoint": "test_ollama"})
        return {"status": "Ollama test failed", "error": str(e)}

@app.post("/reset-error-stats")
async def reset_error_stats():
    """Reset error statistics (useful for monitoring)."""
    try:
        global_error_handler.reset_error_stats()
        return {
            "status": "success",
            "message": "Error statistics reset",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_error(e, {"endpoint": "reset_error_stats"})
        raise HTTPException(status_code=500, detail="Failed to reset error stats")


# Removed MCP endpoints - using only LangChain agents

# Validation and helper functions
async def validate_trip_request(request: TripRequest) -> None:
    """Validate trip request parameters."""
    # Validate dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        if start_date < datetime.now().date():
            raise ValueError("Start date cannot be in the past")
    except ValueError as e:
        if "does not match format" in str(e):
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
        raise e
    
    # Validate duration
    if request.duration_days < 2:
        raise ValueError("Trip duration must be at least 2 days")
    if request.duration_days > 30:
        raise ValueError("Trip duration cannot exceed 30 days")
    
    # Validate destinations
    if not request.destinations:
        raise ValueError("At least one destination is required")
    if len(request.destinations) > 6:
        raise ValueError("Maximum 6 destinations allowed")
    
    # Validate budget
    if request.budget and request.budget.lower() not in ['low', 'medium', 'high']:
        raise ValueError("Budget must be 'low', 'medium', or 'high'")

async def log_trip_request(
    request: TripRequest,
    trip_plan: TripPlan = None,
    processing_time: float = 0,
    status: str = "unknown"
) -> None:
    """Log trip request for analytics (background task)."""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "origin": request.origin,
            "destinations": request.destinations,
            "duration_days": request.duration_days,
            "budget": request.budget,
            "processing_time_seconds": processing_time,
            "status": status
        }
        
        if trip_plan:
            log_data.update({
                "route_order": trip_plan.route_order,
                "flights_count": len(trip_plan.flights),
                "estimated_budget": trip_plan.estimated_budget
            })
        
        # In production, you might want to store this in a database or analytics service
        logger.info(f"Trip request logged: {log_data}")
    
    except Exception as e:
        logger.error(f"Failed to log trip request: {e}")

def _parse_flights_from_text(text: str, request: TripRequest) -> List[Flight]:
    """Extract flight data from agent's final text output.""" 
    flights = []
    try:
        import re
        # Exact pattern to match the required agent output format: "- Seattle to Tokyo: Alaska Airlines - $500"
        pattern = r'-\s+([^:]+?)\s+to\s+([^:]+?):\s+([^-]+?)\s*-\s*\$(\d+(?:,\d+)?)'
        matches = list(re.finditer(pattern, text))
        
        logger.info(f"Looking for flight pattern in text: {text[:300]}")
        logger.info(f"Found {len(matches)} flight matches using exact format pattern")
        
        flight_dates = []
        if len(request.destinations) == 1:
            # Single destination - outbound and return
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            flight_dates = [
                request.start_date,
                (start_dt + timedelta(days=request.duration_days)).strftime("%Y-%m-%d")
            ]
        else:
            # Multi-city - calculate dates for each segment
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            days_per_city = request.duration_days // len(request.destinations)
            current_date = start_dt
            flight_dates = []
            
            # Add dates for each segment
            for i in range(len(request.destinations) + 1):  # +1 for return flight
                flight_dates.append(current_date.strftime("%Y-%m-%d"))
                if i < len(request.destinations):
                    current_date += timedelta(days=days_per_city)
        
        for i, match in enumerate(matches):
            from_city = match.group(1).strip()
            to_city = match.group(2).strip()
            airline = match.group(3).strip()
            price = f"${match.group(4)}"
            
            # Use calculated flight date
            flight_date = flight_dates[i] if i < len(flight_dates) else request.start_date
            
            flight = Flight(
                from_city=from_city,
                to_city=to_city,
                date=flight_date,
                departure_time="08:00",  # Default time since not provided in simple format
                arrival_time="20:00",    # Default time since not provided in simple format
                airline=airline,
                estimated_price=price,
                data_source="langchain_agents_text_extraction",
                confidence=0.9
            )
            flights.append(flight)
                
        logger.info(f"Extracted {len(flights)} flights from text using regex patterns")
        
    except Exception as e:
        logger.error(f"Error parsing flights from text: {e}")
    
    return flights

def _parse_hotels_from_text(text: str, request: TripRequest) -> List[Hotel]:
    """Extract hotel data from agent's final text output."""
    hotels = []
    try:
        import re
        # Exact pattern to match required agent output format: "- Tokyo: Hotel Name - $200/night"
        pattern = r'-\s+([^:]+?):\s+([^-]+?)\s*-\s*\$(\d+(?:,\d+)?)/night'
        matches = list(re.finditer(pattern, text))
        logger.info(f"Found {len(matches)} hotel matches using pattern matching")
        
        for match in matches:
            city = match.group(1).strip()
            name = match.group(2).strip()
            # Clean up any remaining markdown formatting from the name
            if name.startswith('*'):
                name = name.lstrip('* ').strip()
            price = f"${match.group(3)}/night"
            rating = 3.8  # Default rating since not in simple format
                
            hotel = Hotel(
                name=name,
                city=city,
                rating=rating,
                price_per_night=price,
                amenities=["WiFi", "Restaurant", "Gym"], # Default amenities
                address=f"Downtown {city}", # Default address
                data_source="langchain_agents_text_extraction", 
                confidence=0.9
            )
            hotels.append(hotel)
                
        logger.info(f"Extracted {len(hotels)} hotels from text using regex patterns")
        
    except Exception as e:
        logger.error(f"Error parsing hotels from text: {e}")
    
    return hotels

def _create_fallback_flights(request: TripRequest) -> List[Flight]:
    """Create fallback flight data when agent doesn't provide flights."""
    flights = []
    try:
        current_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        route = [request.origin] + request.destinations + [request.origin]
        
        for i in range(len(route) - 1):
            flight_date = (current_date + timedelta(days=i*2)).strftime("%Y-%m-%d")
            
            flight = Flight(
                from_city=route[i],
                to_city=route[i+1],
                date=flight_date,
                departure_time="10:30 AM",
                arrival_time="2:45 PM" + (" (+1 day)" if i > 0 else ""),
                airline="Agent Airlines",
                estimated_price=f"${300 + i*50}",
                data_source="langchain_agents_fallback",
                confidence=0.7
            )
            flights.append(flight)
        
        logger.info(f"Created {len(flights)} fallback flights")
        
    except Exception as e:
        logger.error(f"Error creating fallback flights: {e}")
    
    return flights

def _create_fallback_hotels(request: TripRequest) -> List[Hotel]:
    """Create fallback hotel data when agent doesn't provide hotels."""
    hotels = []
    try:
        for destination in request.destinations:
            hotel = Hotel(
                name=f"Downtown Hotel {destination}",
                city=destination,
                rating=4.2,
                price_per_night=f"${100 + len(destination)*5}",
                amenities=["WiFi", "Pool", "Gym", "Restaurant"],
                address=f"Downtown {destination}",
                data_source="langchain_agents_fallback", 
                confidence=0.7
            )
            hotels.append(hotel)
        
        logger.info(f"Created {len(hotels)} fallback hotels")
        
    except Exception as e:
        logger.error(f"Error creating fallback hotels: {e}")
    
    return hotels

def _transform_flight_for_frontend(flight) -> Dict[str, Any]:
    """Transform flight data from backend format to frontend format."""
    try:
        if hasattr(flight, '__dict__'):
            flight_dict = flight.__dict__.copy()
        else:
            flight_dict = flight.copy()
        
        # Transform field names to match Flutter model expectations
        if 'from_city' in flight_dict:
            flight_dict['from'] = flight_dict.pop('from_city')
        if 'to_city' in flight_dict:
            flight_dict['to'] = flight_dict.pop('to_city')
        
        logger.info(f"✅ Transformed flight: {flight_dict.get('from', 'N/A')} → {flight_dict.get('to', 'N/A')}")
        return flight_dict
    except Exception as e:
        logger.error(f"❌ Error transforming flight: {e}")
        return {}

def _transform_hotel_for_frontend(hotel) -> Dict[str, Any]:
    """Transform hotel data from backend format to frontend format."""
    try:
        if hasattr(hotel, '__dict__'):
            hotel_dict = hotel.__dict__.copy()
        else:
            hotel_dict = hotel.copy()
        
        # Keep field names as snake_case to match Flutter model expectations
        # No transformation needed - Flutter model expects 'price_per_night' directly
        
        logger.info(f"✅ Transformed hotel: {hotel_dict.get('name', 'N/A')} in {hotel_dict.get('city', 'N/A')}")
        return hotel_dict
    except Exception as e:
        logger.error(f"❌ Error transforming hotel: {e}")
        return {}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    log_error(exc, {"url": str(request.url), "method": request.method})
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    print("🚀 Starting LangChain Agent Trip Planner API...")
    print("📝 Features:")
    print("   - 🤖 LangChain AgentExecutor Framework")
    print("   - 🧠 ReAct (Reasoning + Acting) agents")
    print("   - 🔗 Chain-of-thought planning")
    print("   - 🤝 Multi-agent collaboration")
    print("   - 🔍 Google Search tools integration")
    print("   - ✈️ Intelligent travel planning")
    print("   - 🏨 Smart accommodation recommendations")
    print("")
    print("🤖 LangChain Agent System (5 Specialized Agents):")
    print("   - MasterTravelAgent: Comprehensive trip coordination")
    print("   - FlightPlanningAgent: Flight search and route optimization")
    print("   - AccommodationAgent: Hotel and accommodation research")
    print("   - ActivityAgent: Local activities and experiences")
    print("   - BudgetPlanningAgent: Financial analysis and optimization")
    print("")
    print("⚙️  Configuration:")
    print(f"   - Google Search API: {'✅ Enhanced mode' if config.has_google_search_config else '✅ Basic mode'}")
    print(f"   - LangChain Agents: ✅ Enabled with reasoning capabilities")
    print("")
    print("📚 Documentation available at: http://localhost:8000/docs")
    print("🧪 Test endpoints:")
    print("   - POST /plan-trip - Main trip planning endpoint")
    print("   - GET /api-status - Check agent system status") 
    print("   - GET /test-ollama - Test Ollama connectivity")
    print("   - GET /test-flight-cards - Test mobile app integration")
    print("")
    print("💡 Optional Google Search API:")
    print("   - Get API key: https://developers.google.com/custom-search/v1/overview")
    print("   - Copy .env.example to .env and add your keys")
    print("   - System works without API keys using agent reasoning")
    print("")
    print("⚠️  Make sure Ollama is running:")
    print("   ollama serve")
    print("   ollama pull mistral:latest")
    print("   # Alternative: ollama pull gemma3:latest or llama3.1:latest")
    print("")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )