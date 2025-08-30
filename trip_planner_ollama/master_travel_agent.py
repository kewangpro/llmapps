"""
Master Travel Agent - Comprehensive Trip Planning Coordinator

The MasterTravelAgent is the primary agent that coordinates comprehensive trip planning
using all available tools and can orchestrate complex multi-step travel planning tasks.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class MasterTravelAgent(BaseLangChainAgent):
    """
    Master travel planning agent that coordinates with specialized agents.
    
    This agent can:
    - Handle complex multi-step travel planning
    - Coordinate between specialized agents
    - Synthesize information from multiple sources
    - Provide comprehensive travel recommendations
    """
    
    def __init__(self, model_name: str = None):
        from config import get_config
        config = get_config()
        
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="MasterTravelAgent",
            agent_description="Master AI travel planning agent that coordinates comprehensive trip planning using specialized tools and agents for flights, hotels, activities, and budget optimization.",
            model_name=model_name or config.ollama_model,
            temperature=config.ollama_temperature,
            max_iterations=15, # Increased for complex multi-leg trips
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up all travel planning tools for comprehensive planning."""
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
        Plan a complete trip using LangChain's reasoning and tool calling.
        
        This demonstrates true agentic behavior with:
        - Multi-step reasoning
        - Automatic tool selection
        - Chain-of-thought planning
        - Comprehensive synthesis
        """
        
        # Handle both single and multi-city trips properly
        if len(destinations) == 1:
            planning_query = f"""Plan a trip from {origin} to {destinations[0]} starting {start_date} for {duration_days} days.

REQUIRED ACTIONS:
1. Use flight_search tool to find flights from {origin} to {destinations[0]} on {start_date}
2. Use hotel_search tool to find hotels in {destinations[0]}
3. Use budget_analysis tool to analyze the costs
4. Provide comprehensive trip summary

OUTPUT FORMAT REQUIREMENTS:
For flights, use EXACTLY this format:
**Flights:**

*   **OriginCity to DestinationCity:** AirlineName - Depart: HH:MM, Arrive: HH:MM, Price: $Amount

For hotels, use EXACTLY this format:
**Hotels:**

*   **CityName:** HotelName - $Price/night

Examples:
*   **Seattle to Tokyo:** Alaska Airlines - Depart: 08:30, Arrive: 22:26, Price: $937
*   **Tokyo:** Marriott Tokyo Inn - $184/night

Budget: ${budget:,.2f}, Style: {travel_style}"""
        else:
            # Dynamically construct the multi-city planning query
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            days_per_city = duration_days // len(destinations)
            
            route = [origin] + destinations + [origin]
            
            steps = []
            current_date = start_dt
            
            # Create flight and hotel steps
            for i in range(len(route) - 1):
                # Add flight search step
                steps.append(f"STEP {len(steps) + 1}: Find a flight from {route[i]} to {route[i+1]} on or around {current_date.strftime('%Y-%m-%d')}.")
                
                # Add hotel search step (if it's a destination, not the final return to origin)
                if i < len(destinations):
                    hotel_start_date = current_date
                    hotel_end_date = current_date + timedelta(days=days_per_city)
                    steps.append(f"STEP {len(steps) + 1}: Find a hotel in {route[i+1]} from {hotel_start_date.strftime('%Y-%m-%d')} to {hotel_end_date.strftime('%Y-%m-%d')}.")
                    current_date = hotel_end_date # Next flight departs after the hotel stay
            
            # Add budget analysis step
            budget_step = f"STEP {len(steps) + 1}: Call the 'budget_analysis' tool. "
            budget_step += f"Input should be a JSON string with 'flights' (from previous flight_search observations), "
            budget_step += f"'hotels' (from previous hotel_search observations), "
            budget_step += f"'duration_days': {duration_days}, and 'budget_style': '{travel_style}'. "
            budget_step += "Ensure you collect all flight and hotel data from previous steps to pass to this tool."
            steps.append(budget_step)

            # Final summary step
            steps.append(f"STEP {len(steps) + 1}: Provide a comprehensive summary of the trip, including all flights and hotels found.")

            trip_description = " → ".join(route)
            planning_query = f"""Plan a multi-city trip: {trip_description}

BUDGET: ${budget:,.2f}
INTERESTS: {', '.join(interests)}
TRAVEL STYLE: {travel_style}

Follow these steps to complete the plan:
""" + "\n".join(steps) + """

OUTPUT FORMAT REQUIREMENTS:
For flights, use EXACTLY this format:
**Flights:**

*   **OriginCity to DestinationCity:** AirlineName - Depart: HH:MM, Arrive: HH:MM, Price: $Amount

For hotels, use EXACTLY this format:
**Hotels:**

*   **CityName:** HotelName - $Price/night

Examples:
*   **Seattle to Tokyo:** Alaska Airlines - Depart: 08:30, Arrive: 22:26, Price: $937
*   **Tokyo:** Marriott Tokyo Inn - $184/night

Ensure all steps are completed before providing the final summary."""
        
        # Use the agent's reasoning capabilities to plan the trip
        result = await self.process_query(planning_query)
        
        # The agent's response should be a JSON object with the trip plan.
        # We will parse it and return it as a dictionary.
        try:
            trip_plan = json.loads(result["output"])
        except (json.JSONDecodeError, TypeError):
            # If the agent did not return a valid JSON, we'll have to rely on the intermediate steps.
            # This is a fallback within the agent itself.
            trip_plan = {
                "flights": [],
                "hotels": [],
                "activities": [],
                "budget": None,
                "itinerary": None
            }
            for step in result.get("intermediate_steps", []):
                action = step[0]
                if action.tool == "flight_search":
                    try:
                        flights = json.loads(step[1])
                        trip_plan["flights"].extend(flights)
                    except json.JSONDecodeError:
                        # Handle cases where the output is not a valid JSON
                        pass
                elif action.tool == "hotel_search":
                    try:
                        hotels = json.loads(step[1])
                        trip_plan["hotels"].extend(hotels)
                    except json.JSONDecodeError:
                        # Handle cases where the output is not a valid JSON
                        pass
        
        result["trip_plan"] = trip_plan
        return result
