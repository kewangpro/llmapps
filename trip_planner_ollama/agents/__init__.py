"""
TRUE LangChain Multi-Agent Travel Planning System

This module implements LangChain's actual agentic framework with:
- AgentExecutor for reasoning and planning
- ReAct (Reasoning + Acting) framework  
- Chain-of-thought processing
- Multi-agent collaboration with specialized tools
- Autonomous decision making

TRUE LangChain Agents:
- MasterTravelAgent: Comprehensive trip coordination with all tools
- FlightPlanningAgent: Flight search and route optimization specialist
- AccommodationAgent: Hotel and accommodation research expert
- ActivityAgent: Local activities and experience recommendations
- BudgetPlanningAgent: Financial analysis and budget optimization
"""

# TRUE LangChain Agent Framework
from .langchain_base_agent import BaseLangChainAgent, AgentMessage, AgentTask
from .travel_tools import TravelPlanningTools
from .google_enhanced_tools import GoogleEnhancedTravelTools
from .master_travel_agent import MasterTravelAgent
from .flight_planning_agent import FlightPlanningAgent
from .accommodation_agent import AccommodationAgent
from .activity_agent import ActivityAgent
from .budget_planning_agent import BudgetPlanningAgent
from .langchain_multi_agent_system import (
    LangChainMultiAgentSystem, 
    AgentCollaborationResult,
    create_langchain_agent_system
)

__all__ = [
    # Base LangChain Framework
    'BaseLangChainAgent',
    'AgentMessage', 
    'AgentTask',
    
    # Specialized LangChain Agents
    'MasterTravelAgent',
    'FlightPlanningAgent',
    'AccommodationAgent',
    'ActivityAgent', 
    'BudgetPlanningAgent',
    'TravelPlanningTools',
    'GoogleEnhancedTravelTools',
    
    # Multi-Agent System
    'LangChainMultiAgentSystem',
    'AgentCollaborationResult', 
    'create_langchain_agent_system'
]