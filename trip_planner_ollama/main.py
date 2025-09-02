
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import TripRequest, TripPlan, DayPlan, Flight, Hotel
from config import get_config
from services.error_handler import global_error_handler, log_error
from agents import create_langchain_agent_system, LangChainMultiAgentSystem
import uvicorn
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_standardized_json_output(result_text: str, request: TripRequest, agent_result: dict = None) -> tuple:
    """
    Parse standardized JSON output from agent instead of using fragile regex patterns.
    
    Returns:
        tuple: (flights, hotels, activities, budget_float) where:
        - flights: List[Flight] - parsed flight data
        - hotels: List[Hotel] - parsed hotel data  
        - activities: Dict[str, List[str]] - activities by city
        - budget_float: float - extracted budget amount
    """
    from models import Flight, Hotel
    from schemas.agent_output_schema import StandardizedAgentOutput, validate_agent_output
    
    flights = []
    hotels = []
    activities = {}
    budget_float = 3000.0  # Default budget
    
    # Look for JSON in agent output - handle multiple formats
    import re
    json_str = None
    
    # Try direct JSON parsing first
    if result_text.strip().startswith('{'):
        json_str = result_text.strip()
        logger.debug(f"📝 Parsing result text as direct JSON: {json_str[:200]}...")
    else:
        # Try "Final Answer:" format
        final_answer_pattern = r'Final Answer:\s*({.*?})'
        final_answer_match = re.search(final_answer_pattern, result_text, re.DOTALL)
        
        # Try markdown code block format
        markdown_pattern = r'```json\s*({.*?})\s*```'
        markdown_match = re.search(markdown_pattern, result_text, re.DOTALL)
        
        # Try any JSON block pattern
        json_block_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
        json_block_match = re.search(json_block_pattern, result_text, re.DOTALL)
        
        if final_answer_match:
            json_str = final_answer_match.group(1).strip()
            logger.info(f"📝 Found Final Answer JSON: {json_str[:200]}...")
        elif markdown_match:
            json_str = markdown_match.group(1).strip()
            logger.info(f"📝 Found markdown JSON: {json_str[:200]}...")
        elif json_block_match:
            json_str = json_block_match.group(0).strip()
            logger.info(f"📝 Found JSON block: {json_str[:200]}...")
        else:
            logger.error(f"❌ No JSON pattern matched in output: {result_text[:300]}...")

    if json_str:
        try:
            # Parse the JSON
            parsed_data = json.loads(json_str)
            logger.info(f"✅ Successfully parsed JSON with {len(parsed_data.get('flights', []))} flights, {len(parsed_data.get('hotels', []))} hotels, {len(parsed_data.get('activities', []))} activities")
            
            # Patch: supply default date for flights before schema validation
            if 'flights' in parsed_data and isinstance(parsed_data['flights'], list):
                # Calculate return date
                return_date = (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=request.duration_days)).strftime('%Y-%m-%d')
                
                for flight in parsed_data['flights']:
                    if 'date' not in flight or not str(flight.get('date', '')).strip():
                        # Determine if this is outbound or return flight
                        from_city = flight.get('from_city', '').lower()
                        to_city = flight.get('to_city', '').lower()
                        origin = request.origin.lower()
                        
                        # If any destination city appears in from_city and origin appears in to_city, it's a return flight
                        is_return_flight = False
                        for dest in request.destinations:
                            if dest.lower() in from_city and origin in to_city:
                                is_return_flight = True
                                break
                        
                        # Assign appropriate date
                        if is_return_flight:
                            flight['date'] = return_date
                        else:
                            flight['date'] = request.start_date
            
            # Patch: fix budget structure if needed
            if 'budget' in parsed_data and isinstance(parsed_data['budget'], dict):
                budget = parsed_data['budget']
                # If budget doesn't have breakdown, create one
                if 'breakdown' not in budget:
                    total_budget = budget.get('total', 3000.0)
                    flight_cost = budget.get('flights', total_budget * 0.4)
                    hotel_cost = budget.get('hotel', total_budget * 0.3)
                    activity_cost = budget.get('activities', total_budget * 0.2)
                    food_cost = total_budget * 0.1
                    
                    budget['breakdown'] = {
                        'flights': flight_cost,
                        'hotels': hotel_cost,
                        'activities': activity_cost,
                        'food': food_cost,
                        'transport': 0.0
                    }
                    logger.info(f"🔧 Created budget breakdown: flights=${flight_cost}, hotels=${hotel_cost}, activities=${activity_cost}, food=${food_cost}")
                # If budget has breakdown but missing fields, patch them
                elif 'breakdown' in budget:
                    breakdown = budget['breakdown']
                    total_budget = budget.get('total', 3000.0)
                    
                    # Add missing fields with reasonable defaults
                    if 'food' not in breakdown:
                        breakdown['food'] = total_budget * 0.1
                        logger.info(f"🔧 Added missing 'food' field to budget breakdown: ${breakdown['food']}")
                    if 'transport' not in breakdown:
                        breakdown['transport'] = total_budget * 0.05
                        logger.info(f"🔧 Added missing 'transport' field to budget breakdown: ${breakdown['transport']}")
                    if 'activities' not in breakdown:
                        breakdown['activities'] = total_budget * 0.15
                        logger.info(f"🔧 Added missing 'activities' field to budget breakdown: ${breakdown['activities']}")
                    if 'hotels' not in breakdown:
                        breakdown['hotels'] = total_budget * 0.3
                        logger.info(f"🔧 Added missing 'hotels' field to budget breakdown: ${breakdown['hotels']}")
                    if 'flights' not in breakdown:
                        breakdown['flights'] = total_budget * 0.4
                        logger.info(f"🔧 Added missing 'flights' field to budget breakdown: ${breakdown['flights']}")
            
            # Patch: add summary if missing
            if 'summary' not in parsed_data:
                destinations_str = ', '.join(request.destinations) if request.destinations else 'destinations'
                parsed_data['summary'] = f"Trip plan from {request.origin} to {destinations_str} for {request.duration_days} days with {request.budget} budget"
                logger.info(f"🔧 Added default summary: {parsed_data['summary']}")
            
            # Validate against schema
            validated_output = validate_agent_output(parsed_data)
            # Convert flights to Flight models
            for flight_data in validated_output.flights:
                # Patch: supply default date if missing or empty
                flight_date = getattr(flight_data, 'date', None)
                if not flight_date or not str(flight_date).strip():
                    # Use request.start_date as fallback
                    flight_date = request.start_date
                flight = Flight(
                    from_city=flight_data.from_city,
                    to_city=flight_data.to_city,
                    date=flight_date,
                    departure_time=flight_data.departure_time,
                    arrival_time=flight_data.arrival_time,
                    airline=flight_data.airline,
                    estimated_price=f"${flight_data.price:.2f}",
                    data_source=flight_data.source
                )
                flights.append(flight)
                logger.debug(f"✈️  {flight.from_city} → {flight.to_city}: {flight.airline} - {flight.estimated_price} [{flight.data_source}]")
            # Convert hotels to Hotel models
            for hotel_data in validated_output.hotels:
                hotel = Hotel(
                    city=hotel_data.city,
                    name=hotel_data.name,
                    price_per_night=f"${hotel_data.price_per_night:.2f}",
                    rating=float(hotel_data.rating),
                    amenities=hotel_data.amenities,
                    data_source=hotel_data.source
                )
                hotels.append(hotel)
                logger.debug(f"🏨 {hotel.city}: {hotel.name} - {hotel.price_per_night}/night [{hotel.data_source}]")
            # Activities - store as objects instead of just strings
            activities = {}
            for activity_data in validated_output.activities:
                city = getattr(activity_data, 'city', None)
                if city:
                    if city not in activities:
                        activities[city] = []
                    # Store the complete activity object data for transformation
                    activity_obj = {
                        'city': city,
                        'name': getattr(activity_data, 'name', ''),
                        'description': getattr(activity_data, 'description', ''),
                        'category': getattr(activity_data, 'category', ''),
                        'source': getattr(activity_data, 'source', 'ai_agent')
                    }
                    activities[city].append(activity_obj)
            # Budget
            if validated_output.budget:
                budget_float = validated_output.budget.total
        except Exception as e:
            logger.error(f"Error parsing agent output: {e}")
            logger.error(f"Raw agent output for debugging:\n{json_str}")
            flights = []
            hotels = []
            activities = {}
            budget_float = 3000.0
    # Always return a tuple, even if nothing was parsed
    return flights, hotels, activities, budget_float


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
    config = get_config()
    logger.debug(f"Configuration: {config.get_environment_info()}")
    
    # Initialize TRUE LangChain Agent System
    try:
        langchain_agent_system = create_langchain_agent_system(model_name=config.ollama_model)
        logger.info("🤖 LangChain Agent System initialized")
        logger.debug("   - AgentExecutor with ReAct reasoning")
        logger.debug("   - Chain-of-thought planning")
        logger.debug("   - Automatic tool selection")
        logger.debug("   - Multi-agent collaboration")
        logger.debug("   - Google Search tools integration")
    except Exception as e:
        logger.error(f"❌ LangChain Agent System initialization failed: {e}")
        langchain_agent_system = None
    
    # Log API availability
    if config.has_google_search_config:
        logger.info("✅ Google Search API configured")
    else:
        logger.info("✅ Using intelligent travel data with LangChain agents")
    
    if langchain_agent_system:
        logger.debug("🤖 LangChain Agent System ready with 5 specialized reasoning agents")

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

        logger.debug("🤖 Using TRUE LangChain Agent System with reasoning capabilities")
        
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
            collaboration_mode=request.collaboration_mode
        )
        
        if result.primary_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Agent system failed: {result.primary_result.get('error')}")

        # Parse agent output directly
        result_text = str(result.primary_result.get("output", "")) if hasattr(result, 'primary_result') and isinstance(result.primary_result, dict) else str(result)
        agent_result_dict = result.primary_result if hasattr(result, 'primary_result') and isinstance(result.primary_result, dict) else result if isinstance(result, dict) else {}
        flights, hotels, activities, budget_float = _parse_standardized_json_output(result_text, request, agent_result_dict)
        estimated_budget = f"${budget_float:.2f}" if budget_float else "$3000.00"
        # Build TripPlan data
        trip_plan_data = {
            "flights": flights,
            "hotels": hotels,
            "activities": activities,
            "estimated_budget": estimated_budget,
            "total_days": request.duration_days,
            "route_order": [request.origin] + request.destinations,
            "daily_plans": [{"day": i+1, "date": (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d"), "city": request.destinations[0] if request.destinations else request.origin, "activities": []} for i in range(request.duration_days)],
            "travel_tips": []
        }
        trip_plan = TripPlan(**trip_plan_data)
        activities_data = trip_plan_data.get('activities', {})
        logger.debug(f"🔍 Activities data type: {type(activities_data)}, content: {activities_data}")
        if isinstance(activities_data, dict):
            total_activities = sum(len(acts) for acts in activities_data.values())
        elif isinstance(activities_data, list):
            total_activities = len(activities_data)
        else:
            total_activities = 0
        logger.info(f"🎯 After TripPlan creation - flights: {len(trip_plan.flights)}, hotels: {len(trip_plan.hotels)}, activities: {total_activities}")
        # Patch: Remove top-level activities, include activities in daily_plans
        # Assign activities to daily_plans by city and date
        # Rebuild daily_plans: randomly pick one transformed activity per day
        import random
        daily_plans_with_activities = []
        # Flatten activities to a list of dicts
        flat_activities = []
        logger.debug(f"🔍 Flattening activities - type: {type(activities_data)}, content: {activities_data}")
        if isinstance(activities_data, dict):
            for city, acts in activities_data.items():
                logger.debug(f"🔍 Processing city {city} with {len(acts) if isinstance(acts, list) else 'unknown'} activities")
                for act in acts:
                    if isinstance(act, dict):
                        flat_activities.append(act)
                        logger.debug(f"🔍 Added activity: {act.get('name', 'Unknown')}")
        elif isinstance(activities_data, list):
            for act in activities_data:
                if isinstance(act, dict):
                    flat_activities.append(act)
                    logger.debug(f"🔍 Added activity: {act.get('name', 'Unknown')}")
        
        logger.info(f"🎯 Flattened {len(flat_activities)} activities for daily plan assignment")
        # Assign activities to days based on proper city distribution
        for i in range(request.duration_days):
            day_date = (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Calculate which city this day belongs to for multi-city trips
            if len(request.destinations) > 1:
                days_per_city = request.duration_days // len(request.destinations)
                city_index = min(i // days_per_city, len(request.destinations) - 1)
                city = request.destinations[city_index]
            else:
                city = request.destinations[0] if request.destinations else request.origin
            
            activities_list = []
            if flat_activities:
                # Filter activities for the current city
                city_activities = [act for act in flat_activities if act.get('city', '').lower() == city.lower()]
                if city_activities:
                    picked = random.choice(city_activities)
                    logger.debug(f"🎯 Day {i+1} ({city}): Selected activity '{picked.get('name', 'Unknown')}' from {len(city_activities)} available")
                else:
                    # Fallback to any activity if no city-specific activities found
                    picked = random.choice(flat_activities)
                    logger.warning(f"⚠️ Day {i+1} ({city}): No city-specific activities found, using fallback activity")
                
                transformed = _transform_activity_for_frontend(picked)
                logger.debug(f"✅ Transformed activity: {transformed.get('name', 'N/A')} in {transformed.get('city', 'N/A')}")
                # For compatibility: activities field expects simple strings (as per DayPlan model)
                # Use description for richer content, fallback to name if description is empty
                activity_text = transformed.get('description', '').strip() or transformed.get('name', 'Activity')
                activities_list.append(activity_text)
            daily_plans_with_activities.append({
                "day": i+1,
                "date": day_date,
                "city": city,
                "activities": activities_list
            })
        response_data = {
            "total_days": trip_plan.total_days,
            "route_order": trip_plan.route_order,
            "daily_plans": daily_plans_with_activities,
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

    except Exception as e:
        background_tasks.add_task(
            log_trip_request,
            request,
            None,
            (datetime.utcnow() - request_start_time).total_seconds(),
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to plan trip: {str(e)}"
        )

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

def _transform_activity_for_frontend(activity) -> dict:
    """Transform activity data from backend format to frontend format."""
    try:
        if hasattr(activity, '__dict__'):
            activity_dict = activity.__dict__.copy()
        else:
            activity_dict = activity.copy() if isinstance(activity, dict) else {}
        # Only keep relevant fields for frontend
        transformed = {
            "city": activity_dict.get("city", ""),
            "date": activity_dict.get("date", ""),
            "name": activity_dict.get("name", ""),
            "description": activity_dict.get("description", ""),
            "category": activity_dict.get("category", ""),
            "source": activity_dict.get("source", "")
        }
        logger.debug(f"✅ Transformed activity: {transformed.get('name', 'N/A')} in {transformed.get('city', 'N/A')}")
        return transformed
    except Exception as e:
        logger.error(f"❌ Error transforming activity: {e}")
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
