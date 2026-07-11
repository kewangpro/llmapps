"""
Accommodation Agent - Specialized Hotel and Lodging Search

The AccommodationAgent is specialized in finding and recommending hotels, vacation 
rentals, and accommodations with detailed amenity analysis and location optimization.
Enhanced with Google Search integration for real-time hotel data.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .google_enhanced_tools import GoogleEnhancedTravelTools

logger = logging.getLogger(__name__)

class AccommodationAgent(BaseLangChainAgent):
    """LangChain agent specialized in hotel and accommodation search with Google Search integration."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.google_tools = GoogleEnhancedTravelTools()
        
        super().__init__(
            agent_name="AccommodationAgent", 
            agent_description="AI agent specialized in finding and recommending hotels, vacation rentals, and accommodations with Google Search integration for real-time hotel data, detailed amenity analysis, and location optimization.",
            model_name=model_name,
            temperature=0.3,
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up Google Search enhanced accommodation-specific tools only."""
        return [
            # Google Search enhanced tools only
            self.google_tools.google_hotel_search
        ]