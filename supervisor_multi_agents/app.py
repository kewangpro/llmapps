from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler
from langgraph.graph import StateGraph, MessagesState, START
from supervisor import make_supervisor_node
from agents import search_node, web_scraper_node, code_node, llm
import streamlit as st

st.set_page_config(page_title="Chat with agents", page_icon="🦜")
st.title("🦜 Chat with agents")

research_builder = StateGraph(MessagesState)

research_builder.add_node("supervisor", make_supervisor_node(llm, ["search", "web_scraper", "coder"]))
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)
research_builder.add_node("coder", code_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

st.image(research_graph.get_graph().draw_mermaid_png())


if "messages" not in st.session_state or st.sidebar.button("Reset chat history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What's your question?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = {"messages": [("user", f"{prompt}")]}

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        cfg["recursion_limit"] = 50

        for response in research_graph.stream(inputs, config=cfg):
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
