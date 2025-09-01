"""
Master Travel Agent - Comprehensive Trip Planning Coordinator

The MasterTravelAgent is the primary agent that coordinates comprehensive trip planning
using all available tools and can orchestrate complex multi-step travel planning tasks.
"""

import asyncio
import logging
from typing import List, Dict, Any

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
    
    def __init__(self, model_name: str = None, use_google_search: bool = False):
        from config import get_config
        config = get_config()
        
        # Choose tools based on mode
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
            agent_description="Master AI travel planning agent that coordinates comprehensive trip planning using specialized tools and agents for flights, hotels, activities, and budget optimization with Google Search integration when available.",
            model_name=model_name or config.ollama_model,
            temperature=0.1,  # Very low temperature for better format compliance
            max_iterations=8,  # Reduced iterations for simpler tasks
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up travel planning tools - Google enhanced for comprehensive mode."""
        if self.use_google_search and self.google_tools:
            # Use Google enhanced tools for comprehensive mode
            return [
                self.google_tools.google_flight_search,
                self.google_tools.google_hotel_search,
                self.google_tools.google_activity_search,
                self.travel_tools.budget_analysis,
                self.travel_tools.route_optimization
            ]
        else:
            # Use regular tools for simple mode
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
            from datetime import datetime, timedelta
            return_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=duration_days)).strftime('%Y-%m-%d')
            planning_query = f"""Plan a round-trip from {origin} to {destinations[0]}.

You MUST follow these steps and use the actual search tools - DO NOT provide fake or placeholder data:

Step 1: Search flights from {origin} to {destinations[0]} on {start_date}
Step 2: Search return flights from {destinations[0]} to {origin} on {return_date}
Step 3: Search hotels in {destinations[0]} from {start_date} to {return_date}
Step 4: For each day in {destinations[0]}, search for one unique activity based on interests: {interests}. You must return exactly one activity per day, each with its own description and category.
Step 5: Analyze budget allocation for the trip
Step 6: Compile results into the required JSON format

Budget: ${budget:,.2f} for {duration_days} days

CRITICAL: Your response MUST end with EXACTLY this format (no extra text, no markdown, no code blocks):

Final Answer: {{
  "flights": [
    {{
      "from_city": "{origin}",
      "to_city": "{destinations[0]}",
      "date": "{start_date}",
      "airline": "[Actual Airline from Search]",
      "price": [Actual Price Number],
      "departure_time": "[Actual Time]",
      "arrival_time": "[Actual Time]",
      "duration": "[Actual Duration]",
      "source": "google_search"
    }},
    {{
      "from_city": "{destinations[0]}",
      "to_city": "{origin}",
      "date": "{return_date}",
      "airline": "[Actual Airline from Search]",
      "price": [Actual Price Number],
      "departure_time": "[Actual Time]",
      "arrival_time": "[Actual Time]", 
      "duration": "[Actual Duration]",
      "source": "google_search"
    }}
  ],
  "hotels": [
    {{
      "city": "{destinations[0]}",
      "name": "[Actual Hotel Name from Search]",
      "price_per_night": [Actual Price Number],
      "rating": [Actual Rating Number],
      "amenities": ["[Actual Amenity]", "[Actual Amenity]"],
      "source": "google_search"
    }}
  ],
  "activities": [
    // For each day, one activity object:
    {{
      "city": "{destinations[0]}",
      "date": "[YYYY-MM-DD]", // The date for this activity
      "name": "[Actual Activity Name]",
      "description": "[Actual Description]",
      "category": "[Category like 'culture' or 'food']",
      "source": "google_search"
    }}
    // ...repeat for each day
  ],
  "budget": {{
    "total": {budget},
    "breakdown": {{
      "flights": [Estimated Flight Costs],
      "hotels": [Estimated Hotel Costs],
      "activities": [Estimated Activity Costs],
      "food": [Estimated Food Costs],
      "transport": [Estimated Transport Costs]
    }}
  }},
  "summary": "Brief summary of the planned trip"
}}

MANDATORY FORMATTING RULES:
1. Use ACTUAL data from your tool searches - NO placeholders
2. All prices must be numbers (not strings with $)
3. Must be valid JSON format
4. Must include "Final Answer: " prefix EXACTLY
5. NO markdown code blocks (```json)
6. NO extra text after the JSON
7. JSON must be on a single line after "Final Answer: "
8. You MUST call all required search tools

EXAMPLE OF CORRECT FORMAT:
Final Answer: {{"flights": [{{"from_city": "Seattle", "to_city": "Tokyo", "airline": "ANA", "price": 800}}], "hotels": [], "activities": [], "summary": "Trip planned"}}"""
        else:
            # Multi-city trip - simple step-by-step approach
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            days_per_city = duration_days // len(destinations)
            route = [origin] + destinations + [origin]
            
            # Create simple, clear steps
            planning_query = f"""Plan a multi-city trip: {' → '.join(route)}

Search flights, hotels, and activities for each segment:

"""
            current_date = start_dt
            step_num = 1
            
            # Add each flight, hotel, and activity step
            for i in range(len(route) - 1):
                flight_date = current_date.strftime('%Y-%m-%d')
                planning_query += f"Step {step_num}: Search flights from {route[i]} to {route[i+1]} on {flight_date}\n"
                step_num += 1
                
                # Add hotel and activity step if this is a destination (not return leg)
                if i < len(destinations):
                    hotel_end = (current_date + timedelta(days=days_per_city)).strftime('%Y-%m-%d')
                    planning_query += f"Step {step_num}: Search hotels in {route[i+1]} from {flight_date} to {hotel_end}\n"
                    step_num += 1
                    planning_query += f"Step {step_num}: Search for activities in {route[i+1]} based on interests: {interests}\n"
                    step_num += 1
                    current_date += timedelta(days=days_per_city)
            
            planning_query += f"Step {step_num}: Analyze budget allocation for the trip\n"
            step_num += 1
            planning_query += f"Step {step_num}: Compile results into required JSON format\n"
            planning_query += f"\nBudget: ${budget:,.2f} for {duration_days} days\n\n"
            
            # Build JSON template for multi-city
            flights_json = []
            hotels_json = []
            activities_json = []
            
            for i in range(len(route) - 1):
                flight_date = (start_dt + timedelta(days=i*days_per_city)).strftime('%Y-%m-%d')
                flights_json.append(f'''{{
      "from_city": "{route[i]}",
      "to_city": "{route[i+1]}",
      "date": "{flight_date}",
      "airline": "[Actual Airline from Search]",
      "price": [Actual Price Number],
      "departure_time": "[Actual Time]",
      "arrival_time": "[Actual Time]",
      "duration": "[Actual Duration]",
      "source": "google_search"
    }}''')
                
                if i < len(destinations):
                    hotels_json.append(f'''{{
      "city": "{route[i+1]}",
      "name": "[Actual Hotel Name from Search]",
      "price_per_night": [Actual Price Number],
      "rating": [Actual Rating Number],
      "amenities": ["[Actual Amenity]", "[Actual Amenity]"],
      "source": "google_search"
    }}''')
                    
                    activities_json.append(f'''{{
      "city": "{route[i+1]}",
      "name": "[Actual Activity Name]",
      "description": "[Actual Description]",
      "category": "[Category]",
      "source": "google_search"
    }}''')
            
            planning_query += f"""CRITICAL: Your response MUST end with EXACTLY this format (no extra text, no markdown, no code blocks):

Final Answer: {{
  "flights": [
    {','.join(flights_json)}
  ],
  "hotels": [
    {','.join(hotels_json)}
  ],
  "activities": [
    {','.join(activities_json)}
  ],
  "budget": {{
    "total": {budget},
    "breakdown": {{
      "flights": [Estimated Flight Costs],
      "hotels": [Estimated Hotel Costs],
      "activities": [Estimated Activity Costs],
      "food": [Estimated Food Costs],
      "transport": [Estimated Transport Costs]
    }}
  }},
  "summary": "Brief summary of the planned trip"
}}

MANDATORY FORMATTING RULES:
1. Use ACTUAL data from your tool searches - NO placeholders
2. All prices must be numbers (not strings with $)
3. Must be valid JSON format
4. Must include "Final Answer: " prefix EXACTLY
5. NO markdown code blocks (```json)
6. NO extra text after the JSON
7. JSON must be on a single line after "Final Answer: "
8. You MUST call all required search tools

EXAMPLE OF CORRECT FORMAT:
Final Answer: {{"flights": [{{"from_city": "Seattle", "to_city": "Tokyo", "airline": "ANA", "price": 800}}], "hotels": [], "activities": [], "summary": "Trip planned"}}"""
        
        # Use pure LLM agent reasoning - no fallback needed
        logger.debug("🤖 Starting LLM agent reasoning process...")
        result = await self.process_query(planning_query)
        
        logger.info("🎯 Agent completed successfully!")
        
        # Check if the agent produced intermediate steps with tool results
        if "reasoning_steps" in result and result["reasoning_steps"]:
            logger.info(f"✅ Agent used {len(result['reasoning_steps'])} tools and provided reasoning")
        else:
            logger.info("ℹ️ Agent completed without tool usage - may have used cached knowledge")
        
        return result

