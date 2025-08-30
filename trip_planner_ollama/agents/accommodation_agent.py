"""
Accommodation Agent - Specialized Hotel and Lodging Search

The AccommodationAgent is specialized in finding and recommending hotels, vacation 
rentals, and accommodations with detailed amenity analysis and location optimization.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class AccommodationAgent(BaseLangChainAgent):
    """LangChain agent specialized in hotel and accommodation search with reasoning."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="AccommodationAgent", 
            agent_description="AI agent specialized in finding and recommending hotels, vacation rentals, and accommodations with detailed amenity analysis and location optimization.",
            model_name=model_name,
            temperature=0.3,
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up accommodation-specific tools."""
        return [
            self.travel_tools.hotel_search
        ]