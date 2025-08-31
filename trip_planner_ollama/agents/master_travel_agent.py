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
    
    def __init__(self, model_name: str = None):
        from config import get_config
        config = get_config()
        
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="MasterTravelAgent",
            agent_description="Master AI travel planning agent that coordinates comprehensive trip planning using specialized tools and agents for flights, hotels, activities, and budget optimization.",
            model_name=model_name or config.ollama_model,
            temperature=0.1,  # Very low temperature for better format compliance
            max_iterations=8,  # Reduced iterations for simpler tasks
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
            from datetime import datetime, timedelta
            return_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=duration_days)).strftime('%Y-%m-%d')
            planning_query = f"""Plan a round-trip from {origin} to {destinations[0]}.

Step 1: Search flights from {origin} to {destinations[0]} on {start_date}
Step 2: Search hotels in {destinations[0]} from {start_date} to {return_date}  
Step 3: Search return flights from {destinations[0]} to {origin} on {return_date}

Budget: ${budget:,.2f} for {duration_days} days

IMPORTANT: Your Final Answer must be in EXACTLY this format:

Final Answer: FLIGHTS:
- {origin} to {destinations[0]}: [Airline] - $[Price]
- {destinations[0]} to {origin}: [Airline] - $[Price]

HOTELS:  
- {destinations[0]}: [Hotel Name] - $[Price]/night

Use the actual data from your tool searches. Include "Final Answer: " prefix for LangChain parsing."""
        else:
            # Multi-city trip - simple step-by-step approach
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            days_per_city = duration_days // len(destinations)
            route = [origin] + destinations + [origin]
            
            # Create simple, clear steps
            planning_query = f"""Plan a multi-city trip: {' → '.join(route)}

Search flights and hotels for each segment:

"""
            current_date = start_dt
            step_num = 1
            
            # Add each flight and hotel step
            for i in range(len(route) - 1):
                flight_date = current_date.strftime('%Y-%m-%d')
                planning_query += f"Step {step_num}: Search flights from {route[i]} to {route[i+1]} on {flight_date}\n"
                step_num += 1
                
                # Add hotel step if this is a destination (not return leg)
                if i < len(destinations):
                    hotel_end = (current_date + timedelta(days=days_per_city)).strftime('%Y-%m-%d')
                    planning_query += f"Step {step_num}: Search hotels in {route[i+1]} from {flight_date} to {hotel_end}\n"
                    step_num += 1
                    current_date += timedelta(days=days_per_city)
            
            planning_query += f"\nBudget: ${budget:,.2f} for {duration_days} days\n\n"
            planning_query += """IMPORTANT: Your Final Answer must be in EXACTLY this format:

Final Answer: FLIGHTS:
- [Origin] to [Dest1]: [Airline] - $[Price]
- [Dest1] to [Dest2]: [Airline] - $[Price]  
- [Dest2] to [Origin]: [Airline] - $[Price]

HOTELS:
- [Dest1]: [Hotel Name] - $[Price]/night
- [Dest2]: [Hotel Name] - $[Price]/night

Use the actual data from your tool searches. Include "Final Answer: " prefix for LangChain parsing."""
        
        # Use pure LLM agent reasoning - no fallback needed
        logger.info("🤖 Starting LLM agent reasoning process...")
        result = await self.process_query(planning_query)
        
        logger.info("🎯 Agent completed successfully!")
        
        # Check if the agent produced intermediate steps with tool results
        if "reasoning_steps" in result and result["reasoning_steps"]:
            logger.info(f"✅ Agent used {len(result['reasoning_steps'])} tools and provided reasoning")
        else:
            logger.info("ℹ️ Agent completed without tool usage - may have used cached knowledge")
        
        return result

