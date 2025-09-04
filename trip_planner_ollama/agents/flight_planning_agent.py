"""
Flight Planning Agent - Specialized Flight Search and Routing

The FlightPlanningAgent is specialized in flight search, comparison, and booking 
optimization with comprehensive airline knowledge and pricing strategies.
Enhanced with Google Search integration for real-time flight data.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .google_enhanced_tools import GoogleEnhancedTravelTools

logger = logging.getLogger(__name__)

class FlightPlanningAgent(BaseLangChainAgent):
    """LangChain agent specialized in flight search and booking with Google Search integration."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.google_tools = GoogleEnhancedTravelTools()
        
        super().__init__(
            agent_name="FlightPlanningAgent",
            agent_description="AI agent specialized in flight search, comparison, and booking optimization with Google Search integration for real-time flight data, comprehensive airline knowledge, and pricing strategies.",
            model_name=model_name,
            temperature=0.2,  # Lower temperature for more consistent flight searches
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up Google Search enhanced flight-specific tools only."""
        return [
            # Google Search enhanced tools only
            self.google_tools.google_flight_search
        ]