from langgraph.graph import StateGraph, START, END
from steps import PlanExecute, plan_step, execute_step, replan_step, should_end
import asyncio

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)
# Add the execution step
workflow.add_node("agent", execute_step)
# Add a replan node
workflow.add_node("replan", replan_step)

# From start we go to planner
workflow.add_edge(START, "planner")
# From plan we go to agent
workflow.add_edge("planner", "agent")
# From agent, we replan
workflow.add_edge("agent", "replan")
# From replan, we decide to either go to agent, or end
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 US president-elect?"}

async def run_app_workflow():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
#Output:
#{'plan': ['Identify the 2024 US presidential election winner.', 'Determine the hometown of the 2024 US president-elect by researching their biography or profile.']}
#{'executed': [('Identify the 2024 US presidential election winner.', 'The winner of the 2024 US presidential election is Donald J. Trump.')]}
#{'plan': ['Determine the hometown of Donald J. Trump by researching his biography or profile.']}
#{'executed': [('Determine the hometown of Donald J. Trump by researching his biography or profile.', 'Donald J. Trump was born in Queens, New York City, New York. Therefore, his hometown is Queens, New York.')]}
#{'response': 'The hometown of the 2024 US president-elect, Donald J. Trump, is Queens, New York, as determined by researching his biography.'}

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_app_workflow())
    loop.close()

if __name__ == '__main__':
    main()