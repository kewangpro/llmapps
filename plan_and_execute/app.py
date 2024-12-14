from builder import agents_graph
import streamlit as st

st.set_page_config(page_title="Chat with planner", page_icon="🦜")
st.title("🦜 Chat with planner")

with st.sidebar:
    if "messages" not in st.session_state or st.button("Reset chat history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    st.image(agents_graph.get_graph().draw_mermaid_png())

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What's your question?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    config = {"recursion_limit": 50}
    inputs = {"input": f"{prompt}"}

    with st.spinner("Planning and executing..."):
        with st.expander("See plan details"):
            for event in agents_graph.stream(inputs, config=config):
                st.write(event)
                last_response = event

    with st.chat_message("assistant"):
        for k, v in last_response.items():
            st.write(v["response"])
            st.session_state.messages.append({"role": "assistant", "content": v["response"]})
            break
