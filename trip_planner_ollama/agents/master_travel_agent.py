"""
Master Travel Agent - Simplified Comprehensive Trip Planning Coordinator

The MasterTravelAgent now uses a clean, unified approach for both single and multi-city trips,
eliminating the over-engineered complexity that was causing the agent to get stuck.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class MasterTravelAgent(BaseLangChainAgent):
    """
    Master travel planning agent that coordinates with specialized agents.
    
    This agent now uses a simplified, unified approach that:
    - Handles both single and multi-city trips with the same code path
    - Converts any trip into a route array: [origin, dest1, dest2, ..., origin]
    - Makes focused, clean tool calls
    - Assembles results consistently
    """
    
    def __init__(self, model_name: str = None, use_google_search: bool = False):
        from config import get_config
        config = get_config()
        
        # Simplified tool setup - no complex branching
        if use_google_search:
            from .google_enhanced_tools import GoogleEnhancedTravelTools
            self.travel_tools = TravelPlanningTools()
            self.google_tools = GoogleEnhancedTravelTools()
        else:
            self.travel_tools = TravelPlanningTools()
            self.google_tools = None
        
        self.use_google_search = use_google_search
        
        super().__init__(
            agent_name="MasterTravelAgent",
            agent_description="Master AI travel planning agent that uses a unified approach for both single and multi-city trips.",
            model_name=model_name or config.ollama_model,
            temperature=0.1,
            max_iterations=12,  # Increased from 10 for multi-city trips
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up travel planning tools - unified approach."""
        if self.use_google_search and self.google_tools:
            return [
                self.google_tools.google_flight_search,
                self.google_tools.google_hotel_search,
                self.google_tools.google_activity_search,
                self.travel_tools.budget_analysis,
                self.travel_tools.route_optimization
            ]
        else:
            return [
                self.travel_tools.flight_search,
                self.travel_tools.hotel_search, 
                self.travel_tools.activity_search,
                self.travel_tools.budget_analysis,
                self.travel_tools.route_optimization
            ]
    
    async def plan_complete_trip(
        self,
        origin: str,
        destinations: List[str], 
        start_date: str,
        duration_days: int,
        budget: float,
        interests: List[str],
        travel_style: str = "mid-range"
    ) -> Dict[str, Any]:
        """
        Plan a complete trip using simplified unified approach for both single and multi-city trips.
        
        This uses a clean, single code path that:
        - Converts any trip into a route array
        - Makes focused tool calls
        - Assembles results consistently
        """
        # Build unified route: [origin, dest1, dest2, ..., origin] - works for both single and multi-city
        route = [origin] + destinations + [origin]
        logger.info(f"🗺️  Planning route: {' → '.join(route)}")
        
        # Calculate timing uniformly
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        days_per_destination = duration_days // len(destinations) if len(destinations) > 1 else duration_days
        
        # Build search steps dynamically
        search_steps = []
        current_date = start_dt
        
        # Add flight searches
        for i in range(len(route) - 1):
            flight_date = current_date.strftime('%Y-%m-%d')
            search_steps.append(f"Search flights: {{'origin': '{route[i]}', 'destination': '{route[i+1]}', 'departure_date': '{flight_date}'}}")
            
            # If this is a destination (not return leg), add hotel and activities
            if i < len(destinations):
                checkout_date = (current_date + timedelta(days=days_per_destination)).strftime('%Y-%m-%d')
                search_steps.append(f"Search hotels: {{'city': '{route[i+1]}', 'check_in': '{flight_date}', 'check_out': '{checkout_date}'}}")
                search_steps.append(f"Search activities: {{'location': '{route[i+1]}', 'interests': {interests}}}")
                current_date += timedelta(days=days_per_destination)
        
        # Add budget analysis
        search_steps.append(f"Analyze budget for ${budget:,.2f} trip over {duration_days} days")
        
        # Log the search steps for debugging
        logger.info(f"🔍 Generated {len(search_steps)} search steps for route: {' → '.join(route)}")
        for i, step in enumerate(search_steps):
            logger.info(f"   {i+1}. {step}")
        
        # Create clean, focused planning query with explicit stopping
        planning_query = f"""Plan trip: {' → '.join(route)}

You must execute ALL {len(search_steps)} searches in this EXACT order. Do NOT skip any searches:

{chr(10).join([f"□ Step {i+1}: {step}" for i, step in enumerate(search_steps)])}

MANDATORY: Execute each step above in order. After completing all {len(search_steps)} steps, provide your Final Answer JSON using the actual data from your searches.

Final Answer: {{
  "flights": [
    {{"from_city": "actual_city", "to_city": "actual_city", "date": "actual_date", "airline": "actual airline from search", "price": actual_price_number, "departure_time": "actual time", "arrival_time": "actual time", "duration": "actual duration", "source": "search"}}
  ],
  "hotels": [
    {{"city": "actual_city", "name": "actual hotel name from search", "price_per_night": actual_price_number, "rating": actual_rating_number, "amenities": ["actual", "amenities"], "source": "search"}}
  ],
  "activities": [
    {{"city": "actual_city", "name": "actual activity name from search", "description": "actual description from search", "category": "actual category", "source": "search"}}
  ],
  "budget": {{"total": {budget}, "breakdown": {{"flights": 0, "hotels": 0, "activities": 0, "food": 0, "transport": 0}}}},
  "summary": "Brief trip summary based on your search results"
}}

Do NOT continue thinking after providing the Final Answer JSON."""
        
        logger.debug("🤖 Starting simplified trip planning...")
        result = await self.process_query(planning_query)
        
        logger.info("🎯 Trip planning completed!")
        return result