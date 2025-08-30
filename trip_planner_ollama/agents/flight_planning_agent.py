"""
Flight Planning Agent - Specialized Flight Search and Routing

The FlightPlanningAgent is specialized in flight search, comparison, and booking 
optimization with comprehensive airline knowledge and pricing strategies.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class FlightPlanningAgent(BaseLangChainAgent):
    """LangChain agent specialized in flight search and booking with reasoning capabilities."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="FlightPlanningAgent",
            agent_description="AI agent specialized in flight search, comparison, and booking optimization with comprehensive airline knowledge and pricing strategies.",
            model_name=model_name,
            temperature=0.2,  # Lower temperature for more consistent flight searches
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up flight-specific tools."""
        return [
            self.travel_tools.flight_search,
            self.travel_tools.route_optimization
        ]