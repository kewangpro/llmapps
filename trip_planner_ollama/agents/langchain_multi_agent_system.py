"""
True LangChain Multi-Agent Travel Planning System

This implements a collaborative multi-agent system using LangChain's actual
agent framework with reasoning, planning, and inter-agent communication.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .master_travel_agent import MasterTravelAgent
from .flight_planning_agent import FlightPlanningAgent
from .accommodation_agent import AccommodationAgent
from .activity_agent import ActivityAgent
from .budget_planning_agent import BudgetPlanningAgent
from .langchain_base_agent import AgentMessage

logger = logging.getLogger(__name__)

@dataclass
class AgentCollaborationResult:
    """Result from multi-agent collaboration."""
    primary_result: Dict[str, Any]
    agent_contributions: Dict[str, Dict[str, Any]]
    reasoning_chain: List[str]
    tools_used: List[str]
    collaboration_summary: str

class LangChainMultiAgentSystem:
    """
    True LangChain multi-agent system with reasoning, collaboration, and planning.
    
    This system demonstrates real agentic behavior with:
    - Independent agent reasoning and planning
    - Tool-based problem solving
    - Inter-agent collaboration and information sharing
    - Chain-of-thought processing across agents
    - Autonomous decision making
    """
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.model_name = model_name
        
        # Initialize specialized LangChain agents
        self.master_agent = MasterTravelAgent(model_name)
        self.flight_agent = FlightPlanningAgent(model_name)
        self.accommodation_agent = AccommodationAgent(model_name)
        self.activity_agent = ActivityAgent(model_name)
        self.budget_agent = BudgetPlanningAgent(model_name)
        
        # Agent registry for coordination
        self.agents = {
            "master": self.master_agent,
            "flight": self.flight_agent,
            "accommodation": self.accommodation_agent,
            "activity": self.activity_agent,
            "budget": self.budget_agent
        }
        
        # System state
        self.active_collaborations = {}
        self.system_metrics = {
            "total_queries": 0,
            "successful_plans": 0,
            "agent_interactions": 0,
            "tools_used": {}
        }
        
        logger.info(f"LangChain Multi-Agent System initialized with {len(self.agents)} reasoning agents")
    
    async def plan_trip_with_reasoning(
        self,
        origin: str,
        destinations: List[str],
        start_date: str, 
        duration_days: int,
        budget: float,
        interests: List[str],
        travel_style: str = "mid-range",
        collaboration_mode: str = "simple"
    ) -> AgentCollaborationResult:
        """
        Plan a trip using collaborative LangChain agents with reasoning.
        
        This demonstrates true agentic behavior:
        - Master agent reasons about the planning approach
        - Specialized agents contribute their expertise
        - Agents make autonomous decisions about tool usage
        - Results are synthesized through reasoning
        
        Args:
            collaboration_mode: "simple" (master only) or "comprehensive" (all agents)
        """
        
        start_time = datetime.now()
        collaboration_id = f"collab_{{start_time.strftime('%Y%m%d_%H%M%S')}}"
        
        logger.info(f"🤖 Starting LangChain agent collaboration for trip planning")
        logger.info(f"Collaboration mode: {collaboration_mode}")
        
        try:
            if collaboration_mode == "simple":
                # Use master agent only with all tools
                result = await self._simple_agent_planning(
                    origin, destinations, start_date, duration_days, 
                    budget, interests, travel_style
                )
            else:
                # Use collaborative multi-agent approach
                result = await self._collaborative_agent_planning(
                    origin, destinations, start_date, duration_days,
                    budget, interests, travel_style, collaboration_id
                )
            
            # Update metrics
            self.system_metrics["total_queries"] += 1
            self.system_metrics["successful_plans"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent trip planning failed: {e}")
            self.system_metrics["total_queries"] += 1
            
            # Return error result in same format
            return AgentCollaborationResult(
                primary_result={
                    "status": "error",
                    "error": str(e),
                    "agent": "LangChainMultiAgentSystem"
                },
                agent_contributions={},
                reasoning_chain=[f"Error occurred: {str(e)}"],
                tools_used=[],
                collaboration_summary=f"Trip planning failed due to: {str(e)}"
            )
    
    async def _simple_agent_planning(
        self, 
        origin: str, destinations: List[str], start_date: str,
        duration_days: int, budget: float, interests: List[str], travel_style: str
    ) -> AgentCollaborationResult:
        """Use master agent only for trip planning with reasoning."""
        
        logger.info("🎯 Using master agent with comprehensive reasoning")
        
        # Let the master agent reason through the entire planning process
        result = await self.master_agent.plan_complete_trip(
            origin=origin,
            destinations=destinations,
            start_date=start_date,
            duration_days=duration_days,
            budget=budget,
            interests=interests,
            travel_style=travel_style
        )
        
        return AgentCollaborationResult(
            primary_result=result,
            agent_contributions={"master": result},
            reasoning_chain=result.get("reasoning_steps", []),
            tools_used=result.get("tools_used", []),
            collaboration_summary=f"Master agent planned trip using {len(result.get('tools_used', []))} tools with autonomous reasoning"
        )
    
    async def _collaborative_agent_planning(
        self,
        origin: str, destinations: List[str], start_date: str,
        duration_days: int, budget: float, interests: List[str], 
        travel_style: str, collaboration_id: str
    ) -> AgentCollaborationResult:
        """Use collaborative approach with multiple reasoning agents."""
        
        logger.info("🤝 Using collaborative multi-agent reasoning approach")
        
        # Track collaboration state
        self.active_collaborations[collaboration_id] = {
            "start_time": datetime.now(),
            "participants": list(self.agents.keys()),
            "status": "in_progress"
        }
        
        agent_results = {}
        all_reasoning_steps = []
        all_tools_used = []
        
        # Phase 1: Budget Planning (sets constraints for other agents)
        logger.info("💰 Phase 1: Budget agent reasoning about financial planning...")
        budget_query = f"""
        Analyze the budget for this trip and provide detailed allocation recommendations:
        - Total budget: ${budget:,.2f}
        - Destinations: {', '.join(destinations)}
        - Duration: {duration_days} days  
        - Travel style: {travel_style}
        
        Think through the optimal budget allocation and provide specific recommendations.
        """
        
        budget_result = await self.budget_agent.process_query(budget_query)
        agent_results["budget"] = budget_result
        all_reasoning_steps.extend(budget_result.get("reasoning_steps", []))
        all_tools_used.extend(budget_result.get("tools_used", []))
        
        # Phase 2: Flight Planning (uses budget constraints)
        logger.info("✈️ Phase 2: Flight agent reasoning about travel logistics...")
        flight_query = f"""
        Plan flights for this multi-city trip with budget consideration:
        - Route: {origin} → {' → '.join(destinations)} → {origin}
        - Start date: {start_date}
        - Flight budget: Consider the budget allocation from: {budget_result.get('response', 'Standard allocation')}
        
        Reason through the best flight options, routing strategy, and booking recommendations.
        """
        
        flight_result = await self.flight_agent.process_query(flight_query)
        agent_results["flights"] = flight_result
        all_reasoning_steps.extend(flight_result.get("reasoning_steps", []))
        all_tools_used.extend(flight_result.get("tools_used", []))
        
        # Phase 3: Accommodation Planning (uses location and budget info)
        logger.info("🏨 Phase 3: Accommodation agent reasoning about lodging options...")
        
        accommodation_tasks = []
        for i, destination in enumerate(destinations):
            # Calculate stay dates for each destination
            days_per_dest = duration_days // len(destinations)
            check_in = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i * days_per_dest)).strftime("%Y-%m-%d")
            check_out = (datetime.strptime(check_in, "%Y-%m-%d") + timedelta(days=days_per_dest)).strftime("%Y-%m-%d")
            
            accommodation_query = f"""
            Find accommodation in {destination}:
            - Check-in: {check_in}
            - Check-out: {check_out}  
            - Duration: {days_per_dest} nights
            - Style: {travel_style}
            - Budget context: {budget_result.get('response', 'Consider budget allocation')}
            
            Reason through the best accommodation options for this destination.
            """
            
            accommodation_tasks.append(
                self.accommodation_agent.process_query(accommodation_query)
            )
        
        accommodation_results = await asyncio.gather(*accommodation_tasks)
        agent_results["accommodation"] = accommodation_results
        
        for result in accommodation_results:
            all_reasoning_steps.extend(result.get("reasoning_steps", []))
            all_tools_used.extend(result.get("tools_used", []))
        
        # Phase 4: Activity Planning (uses interests and location info)
        logger.info("🎉 Phase 4: Activity agent reasoning about experiences...")
        
        activity_tasks = []
        for destination in destinations:
            activity_query = f"""
            Recommend activities and experiences for {destination}:
            - Interests: {', '.join(interests)}
            - Travel style: {travel_style}
            - Budget consideration: Factor in activity budget from budget analysis
            
            Think through the best activities that match the traveler's interests and style.
            """
            
            activity_tasks.append(
                self.activity_agent.process_query(activity_query)
            )
        
        activity_results = await asyncio.gather(*activity_tasks)
        agent_results["activities"] = activity_results
        
        for result in activity_results:
            all_reasoning_steps.extend(result.get("reasoning_steps", []))
            all_tools_used.extend(result.get("tools_used", []))
        
        # Phase 5: Master Synthesis (master agent reasons about the complete plan)
        logger.info("🎯 Phase 5: Master agent synthesizing comprehensive travel plan...")
        
        synthesis_query = f"""
        Synthesize all the planning information into a comprehensive trip plan:
        
        Budget Analysis: {budget_result.get('response', 'Not available')}
        
        Flight Planning: {flight_result.get('response', 'Not available')}
        
        Accommodation Results: {[r.get('response', 'Not available') for r in accommodation_results]}
        
        Activity Recommendations: {[r.get('response', 'Not available') for r in activity_results]}
        
        Create a final, cohesive travel plan with:
        - Day-by-day itinerary
        - Specific recommendations and bookings
        - Budget summary
        - Travel tips and considerations
        
        Reason through how all these elements work together for an optimal trip.
        """
        
        master_result = await self.master_agent.process_query(synthesis_query)
        agent_results["synthesis"] = master_result
        all_reasoning_steps.extend(master_result.get("reasoning_steps", []))
        all_tools_used.extend(master_result.get("tools_used", []))
        
        # Update collaboration state
        self.active_collaborations[collaboration_id]["status"] = "completed"
        self.active_collaborations[collaboration_id]["end_time"] = datetime.now()
        
        # Create collaboration summary
        collaboration_summary = f"""
        Multi-Agent Collaboration Completed:
        - {len(self.agents)} LangChain agents participated
        - {len(all_reasoning_steps)} reasoning steps executed
        - {len(set(all_tools_used))} unique tools used
        - Comprehensive trip plan synthesized with agent reasoning
        """
        
        self.system_metrics["agent_interactions"] += len(agent_results)
        for tool in all_tools_used:
            self.system_metrics["tools_used"][tool] = self.system_metrics["tools_used"].get(tool, 0) + 1
        
        logger.info("🎉 Multi-agent collaboration completed successfully")
        
        return AgentCollaborationResult(
            primary_result=master_result,
            agent_contributions=agent_results,
            reasoning_chain=all_reasoning_steps,
            tools_used=list(set(all_tools_used)),
            collaboration_summary=collaboration_summary
        )
    
    async def get_specialized_recommendation(
        self, 
        agent_type: str, 
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get specialized recommendation from a specific agent with reasoning.
        
        Demonstrates individual agent reasoning capabilities.
        """
        
        if agent_type not in self.agents:
            return {
                "error": f"Unknown agent type: {agent_type}",
                "available_agents": list(self.agents.keys())
            }
        
        agent = self.agents[agent_type]
        
        logger.info(f"🤖 {agent.agent_name} reasoning about specialized query")
        
        # Add context to query if provided
        if context:
            enhanced_query = f"{query}\n\nAdditional Context: {json.dumps(context, indent=2)}"
        else:
            enhanced_query = query
        
        result = await agent.process_query(enhanced_query)
        
        return {
            **result,
            "agent_type": agent_type,
            "specialization": agent.agent_description
        }
    
    async def demonstrate_agent_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Demonstrate how different agents reason about the same query.
        
        Shows how each agent's specialized perspective and tools lead to 
        different reasoning approaches.
        """
        
        logger.info("🧠 Demonstrating agent reasoning across specialists")
        
        reasoning_demo = {}
        
        # Let each agent reason about the query from their perspective
        for agent_type, agent in self.agents.items():
            if agent_type == "master":
                continue  # Skip master for this demo
                
            specialized_query = f"""
            From your perspective as a {agent.agent_name}, analyze this travel query:
            
            "{query}"
            
            Think about:
            1. What aspects of this query relate to your expertise
            2. What tools you would use to help
            3. What specific recommendations you would provide
            4. How your analysis would help other agents
            
            Show your reasoning process step by step.
            """
            
            result = await agent.process_query(specialized_query)
            reasoning_demo[agent_type] = {
                "agent_name": agent.agent_name,
                "reasoning": result.get("response", ""),
                "tools_considered": result.get("tools_used", []),
                "reasoning_steps": result.get("reasoning_steps", [])
            }
        
        return {
            "query": query,
            "agent_reasoning": reasoning_demo,
            "demonstration_summary": "Each agent analyzed the query from their specialized perspective using LangChain's reasoning framework",
            "framework": "LangChain AgentExecutor with ReAct reasoning"
        }
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities."""
        return {
            "system_name": "LangChain Multi-Agent Travel Planning System",
            "framework": "LangChain AgentExecutor with ReAct reasoning",
            "version": "2.0.0",
            "agents": {
                agent_type: {
                    "name": agent.agent_name,
                    "description": agent.agent_description,
                    "tools": [tool.name for tool in agent.tools],
                    "capabilities": agent.get_agent_info()["capabilities"]
                }
                for agent_type, agent in self.agents.items()
            },
            "collaboration_features": [
                "Multi-agent reasoning coordination",
                "Inter-agent information sharing",
                "Specialized tool usage with planning",
                "Chain-of-thought synthesis",
                "Autonomous decision making",
                "Context-aware recommendations"
            ],
            "reasoning_capabilities": [
                "ReAct (Reasoning + Acting) framework",
                "Chain-of-thought planning",
                "Automatic tool selection",
                "Multi-step problem decomposition", 
                "Evidence-based decision making",
                "Context retention and memory"
            ],
            "metrics": self.system_metrics
        }
    
    async def shutdown(self):
        """Gracefully shutdown the multi-agent system."""
        logger.info("🔄 Shutting down LangChain Multi-Agent System")
        
        # Complete any active collaborations
        for collab_id, collab in self.active_collaborations.items():
            if collab.get("status") == "in_progress":
                collab["status"] = "interrupted"
                collab["end_time"] = datetime.now()
        
        logger.info("✅ LangChain Multi-Agent System shutdown complete")

# Factory function for easy creation
def create_langchain_agent_system(model_name: str = "gemma3:latest") -> LangChainMultiAgentSystem:
    """Create a new LangChain multi-agent system."""
    return LangChainMultiAgentSystem(model_name)
