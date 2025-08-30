"""
Activity Agent - Specialized Local Activities and Experiences

The ActivityAgent is specialized in discovering activities, attractions, cultural 
experiences, and local recommendations with deep local knowledge and personalization.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class ActivityAgent(BaseLangChainAgent):
    """LangChain agent specialized in activities and experiences with reasoning."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="ActivityAgent",
            agent_description="AI agent specialized in discovering activities, attractions, cultural experiences, and local recommendations with deep local knowledge and personalization.",
            model_name=model_name,
            temperature=0.4,  # Higher creativity for activity recommendations
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up activity-specific tools."""
        return [
            self.travel_tools.activity_search
        ]