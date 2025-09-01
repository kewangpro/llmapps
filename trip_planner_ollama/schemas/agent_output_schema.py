"""
Standardized Agent Output Schema

This defines the required JSON format for all agent outputs to eliminate
parsing issues and ensure consistent, reliable data extraction.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator

class FlightData(BaseModel):
    """Standardized flight data structure."""
    from_city: str = Field(..., description="Origin city")
    to_city: str = Field(..., description="Destination city") 
    date: str = Field(..., description="Flight date in YYYY-MM-DD format")
    airline: str = Field(..., description="Airline name")
    price: float = Field(..., description="Price in USD")
    departure_time: str = Field(..., description="Departure time in HH:MM format")
    arrival_time: str = Field(..., description="Arrival time in HH:MM format")
    duration: str = Field(..., description="Flight duration (e.g. '12h 39m')")
    source: str = Field(..., description="Data source: 'google_search' or 'ai_agent'")

class HotelData(BaseModel):
    """Standardized hotel data structure."""
    city: str = Field(..., description="City where hotel is located")
    name: str = Field(..., description="Hotel name")
    price_per_night: float = Field(..., description="Price per night in USD")
    rating: float = Field(..., description="Hotel rating (0-5 scale)")
    amenities: List[str] = Field(default_factory=list, description="List of hotel amenities")
    address: Optional[str] = Field(None, description="Hotel address")
    source: str = Field(..., description="Data source: 'google_search' or 'ai_agent'")

class ActivityData(BaseModel):
    """Standardized activity data structure."""
    city: str = Field(..., description="City where activity is located")
    name: str = Field(..., description="Activity name")
    description: str = Field(..., description="Activity description")
    category: Optional[str] = Field(None, description="Activity category (e.g. 'culture', 'food')")
    source: str = Field(..., description="Data source: 'google_search' or 'ai_agent'")

class BudgetBreakdown(BaseModel):
    """Budget allocation breakdown."""
    flights: float = Field(..., description="Allocated budget for flights")
    hotels: float = Field(..., description="Allocated budget for hotels")
    activities: float = Field(..., description="Allocated budget for activities")
    food: float = Field(..., description="Allocated budget for food")
    transport: float = Field(default=0, description="Allocated budget for local transport")

class BudgetData(BaseModel):
    """Standardized budget data structure."""
    total: float = Field(..., description="Total trip budget in USD")
    currency: str = Field(default="USD", description="Budget currency")
    breakdown: BudgetBreakdown = Field(..., description="Budget breakdown by category")

class StandardizedAgentOutput(BaseModel):
    """
    Standardized output format for all travel planning agents.
    
    This schema ensures consistent, parseable output and eliminates
    the need for complex regex parsing.
    """
    flights: List[FlightData] = Field(default_factory=list, description="List of flight options")
    hotels: List[HotelData] = Field(default_factory=list, description="List of hotel options") 
    activities: List[ActivityData] = Field(default_factory=list, description="List of activity recommendations")
    budget: Optional[BudgetData] = Field(None, description="Budget analysis and breakdown")
    summary: str = Field(..., description="Brief summary of the trip plan")
    
    @validator('flights')
    def validate_flights(cls, v):
        if len(v) == 0:
            raise ValueError("At least one flight must be provided")
        return v
    
    @validator('summary')
    def validate_summary(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Summary cannot be empty")
        return v.strip()

# JSON Schema for validation
AGENT_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "required": ["flights", "hotels", "activities", "summary"],
    "properties": {
        "flights": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from_city", "to_city", "date", "airline", "price", "departure_time", "arrival_time", "duration", "source"],
                "properties": {
                    "from_city": {"type": "string"},
                    "to_city": {"type": "string"},
                    "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                    "airline": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                    "departure_time": {"type": "string"},
                    "arrival_time": {"type": "string"},
                    "duration": {"type": "string"},
                    "source": {"type": "string", "enum": ["google_search", "ai_agent"]}
                }
            }
        },
        "hotels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["city", "name", "price_per_night", "rating", "source"],
                "properties": {
                    "city": {"type": "string"},
                    "name": {"type": "string"},
                    "price_per_night": {"type": "number", "minimum": 0},
                    "rating": {"type": "number", "minimum": 0, "maximum": 5},
                    "amenities": {"type": "array", "items": {"type": "string"}},
                    "address": {"type": "string"},
                    "source": {"type": "string", "enum": ["google_search", "ai_agent"]}
                }
            }
        },
        "activities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["city", "name", "description", "source"],
                "properties": {
                    "city": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                    "source": {"type": "string", "enum": ["google_search", "ai_agent"]}
                }
            }
        },
        "budget": {
            "type": "object",
            "required": ["total", "breakdown"],
            "properties": {
                "total": {"type": "number", "minimum": 0},
                "currency": {"type": "string"},
                "breakdown": {
                    "type": "object",
                    "required": ["flights", "hotels", "activities", "food"],
                    "properties": {
                        "flights": {"type": "number", "minimum": 0},
                        "hotels": {"type": "number", "minimum": 0},
                        "activities": {"type": "number", "minimum": 0},
                        "food": {"type": "number", "minimum": 0},
                        "transport": {"type": "number", "minimum": 0}
                    }
                }
            }
        },
        "summary": {"type": "string", "minLength": 1}
    }
}

def validate_agent_output(data: dict) -> StandardizedAgentOutput:
    """
    Validate agent output against the standardized schema.
    
    Args:
        data: Raw agent output as dictionary
        
    Returns:
        StandardizedAgentOutput: Validated and typed agent output
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    return StandardizedAgentOutput.parse_obj(data)
