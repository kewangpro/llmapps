"""
Travel Agent - Simple Mode Trip Planning

The TravelAgent is dedicated to Simple Mode operation using pure LLM reasoning.
Clean, fast trip planning without external API dependencies.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class TravelAgent(BaseLangChainAgent):
    """
    Simple Mode travel planning agent using pure LLM reasoning.
    
    This agent:
    - Uses only travel_tools.py (pure LLM reasoning)
    - Handles single and multi-city trips efficiently
    - Provides fast execution (~30-60 seconds)
    - No external API dependencies
    """
    
    def __init__(self, model_name: str = None):
        from config import get_config
        config = get_config()
        
        # Simple Mode: Only LLM reasoning tools
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="TravelAgent",
            agent_description="AI travel planning agent for Simple Mode using pure LLM reasoning for fast, comprehensive trip planning.",
            model_name=model_name or config.ollama_model,
            temperature=0.1,
            max_iterations=9,  # 8 tool calls + 1 final answer
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up pure LLM reasoning tools for Simple Mode."""
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
        Plan a complete trip using pure LLM reasoning (Simple Mode).
        
        Fast execution with comprehensive results using LLM knowledge.
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
                search_steps.append(f"Search activities: {{'city': '{route[i+1]}', 'interests': {interests}}}")
                current_date += timedelta(days=days_per_destination)
        
        # Add budget analysis
        search_steps.append(f"Analyze budget for ${budget:,.2f} trip over {duration_days} days")
        
        # Log the search steps for debugging
        logger.info(f"🔍 Generated {len(search_steps)} search steps for route: {' → '.join(route)}")
        for i, step in enumerate(search_steps):
            logger.info(f"   {i+1}. {step}")
        
        # Create clean, focused planning query with explicit stopping
        planning_query = f"""Plan trip: {' → '.join(route)}

INSTRUCTIONS:
1. Execute these {len(search_steps)} searches in order
2. After completing ALL searches, immediately output "Final Answer:" followed by JSON

Search Steps:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(search_steps)])}

After completing search step {len(search_steps)} (budget analysis), you MUST immediately respond with "Final Answer:" followed by this exact JSON structure:

Final Answer: {{
  "flights": [
    // Include ALL flights from your searches - use the actual data from each flight_search call
    {{"from_city": "actual_origin", "to_city": "actual_destination", "date": "actual_date", "airline": "actual airline from search", "price": actual_price_number, "departure_time": "actual time", "arrival_time": "actual time", "duration": "actual duration", "source": "search"}}
    // Repeat for EACH flight search you performed - include all flight segments
  ],
  "hotels": [
    // Include ALL hotels from your searches - use the actual data from each hotel_search call  
    {{"city": "actual_city", "name": "actual hotel name from search", "price_per_night": actual_price_number, "rating": actual_rating_number, "amenities": ["actual", "amenities"], "source": "search"}}
    // Repeat for EACH destination city
  ],
  "activities": [
    // Include ALL activities from your searches - use the actual data from each activity_search call
    {{"city": "actual_city", "name": "actual activity name from search", "description": "actual description from search", "category": "actual category", "source": "search"}}
    // Repeat for EACH destination city  
  ],
  "budget": {{"total": {budget}, "breakdown": {{"flights": flight_cost_number, "hotels": hotel_cost_number, "activities": activity_cost_number, "food": food_cost_number, "transport": transport_cost_number}}}},
  "summary": "Brief trip summary based on your search results"
}}

IMPORTANT BUDGET FORMAT RULES:
- "budget" must have BOTH "total" and "breakdown" fields
- Use "hotels" not "accommodation" 
- Use "transport" not "local_transport"
- All breakdown values must be numbers, not zero

IMMEDIATELY after completing all {len(search_steps)} searches, provide your Final Answer JSON. Do NOT continue thinking, reasoning, or performing additional actions after providing the Final Answer JSON."""
        
        logger.debug("🤖 Starting Simple Mode trip planning...")
        result = await self.process_query(planning_query)
        
        logger.info("🎯 Simple Mode trip planning completed!")
        return result