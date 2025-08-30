from pydantic import BaseModel
from typing import List, Optional

class TripRequest(BaseModel):
    origin: str
    destinations: List[str]
    start_date: str
    duration_days: int
    budget: Optional[str] = "medium"
    preferences: Optional[str] = ""

class Flight(BaseModel):
    from_city: str = ""  # Internal field name
    to_city: str = ""    # Internal field name
    date: str = ""
    departure_time: str = ""
    arrival_time: str = ""
    airline: Optional[str] = None
    estimated_price: Optional[str] = None
    # NEW: Data source tracking for MCP/API/fallback indicators
    data_source: Optional[str] = None  # "mcp", "api", "fallback", "mcp_fallback", etc.
    confidence: Optional[float] = None  # 0.0-1.0 confidence score

class DayPlan(BaseModel):
    day: int
    date: str
    city: str
    activities: List[str]
    accommodation: Optional[str] = None
    transportation: Optional[str] = None
    city_tips: List[str] = []

class Hotel(BaseModel):
    name: str = ""
    city: str = ""
    rating: float = 0.0
    price_per_night: str = ""
    amenities: List[str] = []
    address: Optional[str] = None
    # NEW: Data source tracking for MCP/API/fallback indicators
    data_source: Optional[str] = None  # "mcp", "api", "fallback", "mcp_fallback", etc.
    confidence: Optional[float] = None  # 0.0-1.0 confidence score

class TripPlan(BaseModel):
    total_days: int
    route_order: List[str]
    daily_plans: List[DayPlan]
    estimated_budget: Optional[str] = None
    travel_tips: List[str] = []
    flights: List[Flight] = []
    hotels: List[Hotel] = []