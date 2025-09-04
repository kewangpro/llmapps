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
        search_steps.append(f"Analyze budget for ${int(budget):,} trip over {duration_days} days")
        
        # Log the search steps for debugging
        logger.info(f"🔍 Generated {len(search_steps)} search steps for route: {' → '.join(route)}")
        for i, step in enumerate(search_steps):
            logger.info(f"   {i+1}. {step}")
        
        # Build the search steps list
        search_steps_text = chr(10).join([f"{i+1}. {step}" for i, step in enumerate(search_steps)])
        
        # Build JSON template with budget value
        budget_value = int(budget)
        json_template = """{
  "flights": [
    {"from_city": "San Francisco", "to_city": "Tokyo", "date": "2024-04-01", "airline": "Japan Airlines", "price": 850, "departure_time": "10:30", "arrival_time": "14:20+1", "duration": "11h 50m", "source": "llm"},
    {"from_city": "Tokyo", "to_city": "Seoul", "date": "2024-04-05", "airline": "Korean Air", "price": 420, "departure_time": "09:15", "arrival_time": "11:30", "duration": "2h 15m", "source": "llm"}
    // Include ALL flights from your searches - replace examples with actual search results
  ],
  "hotels": [
    {"city": "Tokyo", "name": "Park Hyatt Tokyo", "price_per_night": 450, "rating": 4.8, "amenities": ["Wi-Fi", "Fitness Center", "Spa"], "source": "llm"},
    {"city": "Seoul", "name": "Lotte Hotel Seoul", "price_per_night": 280, "rating": 4.6, "amenities": ["Wi-Fi", "Business Center", "Pool"], "source": "llm"}
    // Include ALL hotels from your searches - replace examples with actual search results
  ],
  "activities": [
    {"city": "Tokyo", "name": "Senso-ji Temple", "description": "Historic Buddhist temple in Asakusa", "category": "culture", "source": "llm"},
    {"city": "Seoul", "name": "Bukchon Hanok Village", "description": "Traditional Korean architecture district", "category": "culture", "source": "llm"}
    // Include ALL activities from your searches - replace examples with actual search results
  ],
  "budget": {"total": """ + str(budget_value) + """, "breakdown": {"flights": 1270, "hotels": 2920, "activities": 400, "food": 800, "transport": 200}},
  "summary": "Multi-city adventure through Tokyo and Seoul with cultural highlights and premium accommodations"
}"""

        # Create clean, focused planning query with explicit stopping
        planning_query = f"""Plan trip: {' → '.join(route)}

INSTRUCTIONS:
1. Execute these {len(search_steps)} searches in order
2. After completing ALL searches, immediately output "Final Answer:" followed by JSON

Search Steps:
{search_steps_text}

After completing search step {len(search_steps)} (budget analysis), you MUST immediately respond with "Final Answer:" followed by this exact JSON structure:

Final Answer: {json_template}

CRITICAL JSON FORMAT REQUIREMENTS:
- "budget" must contain EXACTLY these two fields: "total" and "breakdown"
- "budget.total" must be the overall budget number (e.g., 3000)
- "budget.breakdown" must contain: "flights", "hotels", "activities", "food", "transport" 
- Do NOT create "total_budget" as a separate field
- Do NOT put breakdown values directly in "budget"
- Use "hotels" not "accommodation", "transport" not "local_transport"
- All breakdown values must be numbers, not zero

CORRECT budget format example:
"budget": {{"total": 3000, "breakdown": {{"flights": 900, "hotels": 900, "activities": 450, "food": 600, "transport": 150}}}}

INCORRECT formats (DO NOT USE):
- "budget": {{"flights": 900, ...}}, "total_budget": 3000
- "total": 3000, "budget": {{"flights": 900, ...}}

IMMEDIATELY after completing all {len(search_steps)} searches, provide your Final Answer JSON. Do NOT continue thinking, reasoning, or performing additional actions after providing the Final Answer JSON."""
        
        logger.debug("🤖 Starting Simple Mode trip planning...")
        result = await self.process_query(planning_query)
        
        logger.info("🎯 Simple Mode trip planning completed!")
        return result