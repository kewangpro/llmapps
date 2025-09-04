"""
Activity Agent - Specialized Local Activities and Experiences

The ActivityAgent is specialized in discovering activities, attractions, cultural 
experiences, and local recommendations with deep local knowledge and personalization.
Enhanced with Google Search integration for real-time activity data.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .google_enhanced_tools import GoogleEnhancedTravelTools

logger = logging.getLogger(__name__)

class ActivityAgent(BaseLangChainAgent):
    """LangChain agent specialized in activities and experiences with Google Search integration."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.google_tools = GoogleEnhancedTravelTools()
        
        super().__init__(
            agent_name="ActivityAgent",
            agent_description="AI agent specialized in discovering activities, attractions, cultural experiences, and local recommendations with Google Search integration for real-time activity data, deep local knowledge, and personalization.",
            model_name=model_name,
            temperature=0.4,  # Higher creativity for activity recommendations
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up Google Search enhanced activity-specific tools only."""
        return [
            # Google Search enhanced tools only
            self.google_tools.google_activity_search
        ]