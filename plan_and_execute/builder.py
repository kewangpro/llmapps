from langgraph.graph import StateGraph, START, END
from steps import PlanExecute, plan_step, agent_step, replan_step, should_end

builder = StateGraph(PlanExecute)

# Add the plan node
builder.add_node("planner", plan_step)
# Add the execution step
builder.add_node("agent", agent_step)
# Add a replan node
builder.add_node("replan", replan_step)

# From start we go to planner
builder.add_edge(START, "planner")
# From plan we go to agent
builder.add_edge("planner", "agent")
# From agent, we replan
builder.add_edge("agent", "replan")
# From replan, we decide to either go to agent, or end
builder.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
agents_graph = builder.compile()
