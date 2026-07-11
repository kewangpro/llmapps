"""
Budget Planning Agent - Specialized Financial Analysis and Optimization

The BudgetPlanningAgent is specialized in travel budget analysis, cost optimization, 
and financial planning with expertise in different travel styles and cost-saving strategies.
"""

import logging
from typing import List

from langchain.tools import BaseTool

from .langchain_base_agent import BaseLangChainAgent
from .travel_tools import TravelPlanningTools

logger = logging.getLogger(__name__)

class BudgetPlanningAgent(BaseLangChainAgent):
    """LangChain agent specialized in budget planning and financial optimization."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.travel_tools = TravelPlanningTools()
        
        super().__init__(
            agent_name="BudgetPlanningAgent",
            agent_description="AI agent specialized in travel budget analysis, cost optimization, and financial planning with expertise in different travel styles and cost-saving strategies.",
            model_name=model_name,
            temperature=0.2,  # Lower temperature for consistent budget calculations
            verbose=True
        )
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up budget-specific tools."""
        return [
            self.travel_tools.budget_analysis
        ]