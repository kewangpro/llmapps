from langgraph.graph import StateGraph, START, END
from steps import PlanExecute, plan_step, execute_step, replan_step, should_end
import streamlit as st

st.set_page_config(page_title="Chat with planner", page_icon="🦜")
st.title("🦜 Chat with planner")

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

with st.sidebar:
    st.image(app.get_graph().draw_mermaid_png())


if "messages" not in st.session_state or st.sidebar.button("Reset chat history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What's your question?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    config = {"recursion_limit": 50}
    inputs = {"input": f"{prompt}"}

    with st.spinner("Planning and executing..."):
        with st.expander("See plan details"):
            for event in app.stream(inputs, config=config):
                st.write(event)
                last_response = event

    with st.chat_message("assistant"):
        for k, v in last_response.items():
            st.write(v["response"])
            st.session_state.messages.append({"role": "assistant", "content": v["response"]})
            break
