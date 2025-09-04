"""
Master Synthesis Agent - Comprehensive Mode Final Synthesis

The MasterSynthesisAgent is used exclusively in Comprehensive Mode to synthesize
results from specialized agents (Budget, Flight, Accommodation, Activity) into
a cohesive final travel plan with reasoning and recommendations.
"""

import logging
from typing import List, Dict, Any

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent

logger = logging.getLogger(__name__)

class MasterSynthesisAgent(BaseLangChainAgent):
    """
    Synthesis agent for Comprehensive Mode multi-agent collaboration.
    
    This agent:
    - Receives results from 4 specialized agents
    - Synthesizes information into cohesive travel plan
    - Provides reasoning and recommendations
    - Creates structured final output
    """
    
    def __init__(self, model_name: str = None):
        from config import get_config
        config = get_config()
        
        # Synthesis agent doesn't need any tools - it just combines results
        
        super().__init__(
            agent_name="MasterSynthesisAgent",
            agent_description="AI synthesis agent that combines multi-agent results into cohesive travel plans with reasoning and recommendations.",
            model_name=model_name or config.ollama_model,
            temperature=0.2,  # Slightly higher for creative synthesis
            max_iterations=3,  # Just for synthesis, not extensive searching
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up tools for synthesis work."""
        return [
            # No tools needed - synthesis agent just combines existing results
        ]
    
    async def synthesize_trip_plan(
        self,
        budget_analysis: str,
        flight_results: str, 
        accommodation_results: List[str],
        activity_results: List[str],
        origin: str,
        destinations: List[str],
        duration_days: int,
        travel_style: str
    ) -> Dict[str, Any]:
        """
        Synthesize comprehensive trip plan from specialized agent results.
        
        This combines all the research from specialized agents into a 
        cohesive, structured travel plan with reasoning.
        """
        logger.info("🎯 Starting trip synthesis from multi-agent collaboration results")
        
        # Build synthesis query that produces the expected JSON format for main.py
        synthesis_query = f"""Synthesize the multi-agent travel planning results into a cohesive travel plan.

AVAILABLE DATA FROM SPECIALIZED AGENTS:
Budget Analysis: {budget_analysis}
Flight Results: {flight_results}
Accommodations: {chr(10).join([f"Location {i+1}: {result}" for i, result in enumerate(accommodation_results)])}
Activities: {chr(10).join([f"Destination {i+1}: {result}" for i, result in enumerate(activity_results)])}

Trip Context: {' → '.join([origin] + destinations + [origin])} ({duration_days} days, {travel_style} style)

IMPORTANT: Extract and structure the flight, hotel, and activity data from the above results into the exact JSON format below. Parse the agent results to extract specific flights, hotels, and activities mentioned.

Final Answer: {{
  "flights": [
    // Extract actual flight information from Flight Results above
    // Example: {{"from_city": "Seattle", "to_city": "Seoul", "date": "2025-10-02", "airline": "Partner Airlines", "price": 613, "departure_time": "09:50", "arrival_time": "00:02", "duration": "15h 45m", "source": "comprehensive"}}
  ],
  "hotels": [
    // Extract actual hotel information from Accommodations above  
    // Example: {{"city": "Seoul", "name": "Akira Back", "price_per_night": 142, "rating": 3.6, "amenities": ["Breakfast", "Bar", "WiFi"], "source": "comprehensive"}}
  ],
  "activities": [
    // Extract actual activity information from Activities above
    // Example: {{"city": "Seoul", "name": "Gyeongbokgung Palace", "description": "Historic Korean palace", "category": "culture", "source": "comprehensive"}}
  ],
  "budget": {{
    "total": 3000,
    "breakdown": {{
      "flights": 900,
      "hotels": 900, 
      "activities": 450,
      "food": 600,
      "transport": 150
    }}
  }},
  "summary": "Comprehensive travel plan synthesis combining specialized agent research with strategic recommendations for optimal {travel_style} travel experience"
}}

CRITICAL: Parse the actual flight, hotel, and activity details from the agent results above. Do not create fake data - use the specific information provided by each specialized agent."""
        
        logger.debug("🤖 Starting comprehensive synthesis...")
        
        # Direct LLM call without agent executor since this is pure synthesis
        from config import get_config
        config = get_config()
        
        from langchain_community.llms import Ollama
        llm = Ollama(model=config.ollama_model, temperature=0.2)
        
        # Get direct response from LLM
        response = await llm.ainvoke(synthesis_query)
        
        # Structure the result
        result = {
            "response": response,
            "reasoning_steps": ["Synthesis of multi-agent collaboration results"],
            "tools_used": [],
            "agent_name": self.agent_name
        }
        
        logger.info("🎯 Trip synthesis completed!")
        logger.info(f"🔍 DEBUG - Synthesis output length: {len(response)} chars")
        logger.info(f"🔍 DEBUG - Output starts with: {response[:100]}...")
        logger.info(f"🔍 DEBUG - Output ends with: ...{response[-100:]}")
        return result