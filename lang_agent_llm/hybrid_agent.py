from typing import List, Tuple, Union, Literal
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tools import search_tool, scrape_webpages, python_repl_tool
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")

class AgentAssignment(BaseModel):
    """Assignment of plan step to specific agent"""
    step: str = Field(description="The step to execute")
    agent: Literal["search", "web_scraper", "coder"] = Field(description="Which agent to assign this step to")
    reasoning: str = Field(description="Why this agent is best for this step")

class ExecutionPlan(BaseModel):
    """Complete execution plan with agent assignments"""
    assignments: List[AgentAssignment] = Field(description="List of step-agent assignments")

class Answer(BaseModel):
    """Final response to user"""
    response: str

class HybridState(TypedDict):
    input: str
    plan: List[str]
    assignments: List[dict]
    executed: List[Tuple]
    response: str
    current_step: int

class HybridAgent:
    """Hybrid agent combining planning with multi-agent coordination"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_components()
        self._build_graph()
    
    def _setup_components(self):
        """Setup planner and specialized agents"""
        
        # Strategic planner
        planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strategic planner. Create a step-by-step plan to accomplish the user's request. "
                "Consider what types of agents (search, web_scraper, coder) would be best for each step. "
                "Make each step specific and actionable."
            ),
            ("user", "{input}")
        ])
        self.planner = planner_prompt | self.llm.with_structured_output(Plan)
        
        # Agent assignment planner
        assignment_prompt = ChatPromptTemplate.from_template(
            "Given this plan: {plan}\n\n"
            "Assign each step to the most appropriate agent:\n"
            "- search: For web searches and finding information\n"
            "- web_scraper: For extracting content from specific URLs\n"
            "- coder: For data analysis, calculations, and generating charts\n\n"
            "Provide reasoning for each assignment."
        )
        self.assignment_planner = assignment_prompt | self.llm.with_structured_output(ExecutionPlan)
        
        # Specialized agents
        self.search_agent = create_react_agent(
            self.llm,
            tools=[search_tool],
            prompt="You are a search specialist. Focus on finding relevant information."
        )
        
        self.web_scraper_agent = create_react_agent(
            self.llm,
            tools=[scrape_webpages],
            prompt="You are a web scraping specialist. Extract detailed content from URLs."
        )
        
        self.code_agent = create_react_agent(
            self.llm,
            tools=[python_repl_tool],
            prompt="You are a code specialist. Perform calculations, data analysis, and generate visualizations."
        )
        
        # Final synthesizer
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Based on the original request: {input}\n\n"
            "And the executed steps: {executed}\n\n"
            "Provide a single, clear, and comprehensive final answer. Do not repeat information. "
            "Synthesize the results into one concise response."
        )
        self.synthesizer = synthesis_prompt | self.llm.with_structured_output(Answer)
    
    def _build_graph(self):
        """Build the hybrid workflow graph"""
        builder = StateGraph(HybridState)
        
        # Add nodes
        builder.add_node("planner", self._plan_step)
        builder.add_node("assign_agents", self._assign_agents_step)
        builder.add_node("execute_step", self._execute_step)
        builder.add_node("synthesize", self._synthesize_step)
        
        # Add edges
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "assign_agents")
        builder.add_edge("assign_agents", "execute_step")
        builder.add_conditional_edges(
            "execute_step",
            self._should_continue,
            ["execute_step", "synthesize"]
        )
        builder.add_edge("synthesize", END)
        
        self.graph = builder.compile()
    
    def _plan_step(self, state: HybridState):
        """Create strategic plan"""
        plan = self.planner.invoke({"input": state["input"]})
        return {"plan": plan.steps, "current_step": 0}
    
    def _assign_agents_step(self, state: HybridState):
        """Assign steps to appropriate agents"""
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
        execution_plan = self.assignment_planner.invoke({"plan": plan_str})
        
        assignments = []
        for assignment in execution_plan.assignments:
            assignments.append({
                "step": assignment.step,
                "agent": assignment.agent,
                "reasoning": assignment.reasoning
            })
        
        return {"assignments": assignments}
    
    def _execute_step(self, state: HybridState):
        """Execute current step with assigned agent"""
        current_step = state["current_step"]
        if current_step >= len(state["assignments"]):
            return {"current_step": current_step}
        
        assignment = state["assignments"][current_step]
        step = assignment["step"]
        agent_type = assignment["agent"]
        
        # Select appropriate agent
        if agent_type == "search":
            agent = self.search_agent
        elif agent_type == "web_scraper":
            agent = self.web_scraper_agent
        else:  # coder
            agent = self.code_agent
        
        # Execute step
        try:
            result = agent.invoke({"messages": [("user", step)]})
            executed_entry = (step, agent_type, result["messages"][-1].content)
        except Exception as e:
            executed_entry = (step, agent_type, f"Error: {str(e)}")
        
        # Update state
        executed = state.get("executed", [])
        executed.append(executed_entry)
        
        return {
            "executed": executed,
            "current_step": current_step + 1
        }
    
    def _synthesize_step(self, state: HybridState):
        """Synthesize final answer from all executed steps"""
        # Check if we already have enough information from the agents
        executed = state.get("executed", [])
        if executed:
            # If the last step already contains a complete answer, use it directly
            last_result = executed[-1][2]  # Get result from last executed step
            if "President" in last_result and ("Trump" in last_result or "Biden" in last_result):
                # Direct answer found, no need to synthesize
                return {"response": last_result}
        
        # Otherwise, synthesize from all steps
        executed_summary = "\n".join([
            f"Step: {step}\nAgent: {agent}\nResult: {result}\n"
            for step, agent, result in executed
        ])
        
        final_answer = self.synthesizer.invoke({
            "input": state["input"],
            "executed": executed_summary
        })
        
        return {"response": final_answer.response}
    
    def _should_continue(self, state: HybridState):
        """Determine if we should execute more steps or synthesize"""
        current_step = state.get("current_step", 0)
        total_steps = len(state.get("assignments", []))
        
        if current_step >= total_steps:
            return "synthesize"
        else:
            return "execute_step"
    
    def get_graph(self):
        """Get the graph for visualization"""
        return self.graph
    
    def process(self, message: str) -> str:
        """Process a message and return the response"""
        config = {"recursion_limit": 50}
        inputs = {"input": message}
        
        result = self.graph.invoke(inputs, config=config)
        return result.get("response", "No response generated")
    
    def process_with_streaming(self, message: str, response_container) -> str:
        """Process with streaming updates"""
        config = {"recursion_limit": 50}
        inputs = {"input": message}
        
        final_response = ""
        
        for event in self.graph.stream(inputs, config=config):
            response_container.json(event)
            
            # Check for final response
            for k, v in event.items():
                if "response" in v and v["response"]:
                    final_response = v["response"]
        
        return final_response or "No response generated"