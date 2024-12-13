from typing_extensions import TypedDict
from typing import Annotated, List, Tuple, Union, Literal
from executor import agent_executor
from planner import planner, replanner, Response
from langgraph.graph import END
import operator

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    executed: Annotated[List[Tuple], operator.add]
    response: str

def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})
    return {"executed": [(task, agent_response["messages"][-1].content)]}

def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
