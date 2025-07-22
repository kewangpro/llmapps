from typing import List, Tuple, Union
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tools import search_tool
import json
# Removed custom logging - using LangChain debug only

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(description="different steps to follow, should be in sorted order as numbered list of strings")

class Answer(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Answer, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Answer. "
        "If you need to further use tools to get the answer, use Plan."
    )

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    executed: List[Tuple]
    response: str
    step_count: int

class PlanExecuteAgent:
    """Plan and Execute agent implementation"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_components()
        self._build_graph()
    
    def _setup_components(self):
        """Setup planner, replanner, and executor components"""
        
        # Create executor agent
        prompt = "You are a helpful assistant."
        self.agent_executor = create_react_agent(
            self.llm, 
            [search_tool], 
            prompt=prompt
        )
        
        # Setup planner
        planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "For the given objective, come up with a simple step by step plan. "
                "This plan should involve individual tasks, that if executed correctly will yield the correct answer. "
                "Do not add any superfluous steps. The result of the final step should be the final answer. "
                "Make sure that each step has all the information needed - do not skip steps. "
                "Return only simple action steps as strings, not JSON objects or complex structures. "
                "Each step should be a clear, actionable instruction."
            ),
            ("placeholder", "{messages}"),
        ])
        self.planner = planner_prompt | self.llm.with_structured_output(Plan)
        
        # Setup replanner
        replanner_prompt = ChatPromptTemplate.from_template(
            "For the given objective, come up with a simple step by step plan. "
            "This plan should involve individual tasks, that if executed correctly will yield the correct answer. "
            "Do not add any superfluous steps. The result of the final step should be the final answer. "
            "Make sure that each step has all the information needed - do not skip steps. "
            "Return only simple action steps as strings, not JSON objects or complex structures.\n\n"
            "Your objective was this:\n{input}\n\n"
            "Your original plan was this:\n{plan}\n\n"
            "You have currently done the follow steps:\n{executed}\n\n"
            "IMPORTANT: If you have already gathered enough information to answer the user's question, "
            "provide a final answer instead of continuing with more steps. "
            "Only create a new plan if you truly need more information.\n\n"
            "If no more steps are needed and you can return to the user, then respond with that.\n"
            "Otherwise, Update your plan accordingly. Do not return previously done steps as part of the plan."
        )
        self.replanner = replanner_prompt | self.llm.with_structured_output(Act)
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(PlanExecute)
        
        # Add nodes
        builder.add_node("planner", self._plan_step)
        builder.add_node("agent", self._agent_step)
        builder.add_node("replan", self._replan_step)
        
        # Add edges
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "agent")
        builder.add_edge("agent", "replan")
        builder.add_conditional_edges(
            "replan",
            self._should_end,
            ["agent", END],
        )
        
        self.graph = builder.compile()
    
    def _plan_step(self, state: PlanExecute):
        """Create initial plan"""
        plan = self.planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps, "step_count": 0}
    
    def _agent_step(self, state: PlanExecute):
        """Execute the next step in the plan"""
        plan = state["plan"]
        executed = state.get("executed", [])
        
        if not plan:
            return {"executed": executed, "step_count": state.get("step_count", 0)}
        
        # Get the next unexecuted step
        remaining_steps = plan[len(executed):] if len(executed) < len(plan) else plan[:1]
        if not remaining_steps:
            return {"executed": executed, "step_count": state.get("step_count", 0)}
        
        task = remaining_steps[0]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        step_number = len(executed) + 1
        
        task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing step {step_number}: {task}

Be specific and actionable in your response. If this involves searching, use the search tool to find current information."""
        
        agent_response = self.agent_executor.invoke({"messages": [("user", task_formatted)]})
        response_content = agent_response["messages"][-1].content
        
        # Accumulate executed steps with previous ones
        executed.append((task, response_content))
        
        # Increment step counter
        step_count = state.get("step_count", 0) + 1
        
        return {"executed": executed, "step_count": step_count}
    
    def _replan_step(self, state: PlanExecute):
        """Replan based on executed steps"""
        executed = state.get("executed", [])
        plan = state.get("plan", [])
        step_count = state.get("step_count", 0)
        
        # Check if we've done too many steps - force final answer
        if step_count >= 6:
            # Generate final answer based on executed steps
            final_answer = executed[-1][1] if executed else "Unable to complete task"
            return {"response": final_answer}
        
        # Check if we've completed all planned steps
        if len(executed) >= len(plan):
            # All steps completed - check if we have an answer
            if executed:
                last_result = executed[-1][1]
                # If the last result seems to contain an answer, return it
                if any(keyword in last_result.lower() for keyword in ['president', 'current', 'biden', 'trump']):
                    return {"response": last_result}
        
        try:
            output = self.replanner.invoke(state)
            
            if isinstance(output.action, Answer):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}
        except Exception as e:
            # Fallback if replanner fails
            if executed:
                return {"response": executed[-1][1]}
            else:
                return {"response": "Unable to complete task"}
    
    def _should_end(self, state: PlanExecute):
        """Determine if we should end or continue"""
        if "response" in state and state["response"]:
            return END
        elif "plan" in state and state["plan"] and len(state["plan"]) > 0:
            return "agent"
        else:
            return END
    
    def get_graph(self):
        """Get the graph for visualization"""
        return self.graph
    
    def process(self, message: str) -> str:
        """Process a message and return the response"""
        config = {"recursion_limit": 100}
        inputs = {"input": message}
        
        result = self.graph.invoke(inputs, config=config)
        
        # First check if we have an explicit response
        if result.get("response"):
            return result["response"]
        
        # If no explicit response, extract from the last executed step
        executed = result.get("executed", [])
        if executed:
            # Return the result of the last executed step
            return executed[-1][1]
        
        return "No response generated"
    
    def process_with_streaming(self, message: str, response_container) -> str:
        """Process with streaming updates"""
        config = {"recursion_limit": 100}
        inputs = {"input": message}
        
        steps = []
        final_response = ""
        last_executed = []
        
        for event in self.graph.stream(inputs, config=config):
            steps.append(event)
            response_container.json(event)
            
            # Check if we have a final response
            for k, v in event.items():
                if "response" in v and v["response"]:
                    final_response = v["response"]
                # Track executed steps
                if "executed" in v and v["executed"]:
                    last_executed = v["executed"]
        
        # If we have an explicit response, return it
        if final_response:
            return final_response
        
        # Otherwise, extract from the last executed step
        if last_executed:
            return last_executed[-1][1]
        
        return "No response generated"