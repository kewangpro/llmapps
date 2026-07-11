import streamlit as st
from agent_manager import AgentManager
import json
import logging
import io
import sys

# Setup custom log capture
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        message = record.getMessage()
        # Only capture LangChain debug output, not our custom chain of thought
        if "langchain" in record.name.lower():
            self.logs.append(message)
    
    def get_and_clear_logs(self):
        logs = self.logs.copy()
        self.logs.clear()
        return logs

# Initialize custom handler
if 'log_handler' not in st.session_state:
    st.session_state.log_handler = StreamlitLogHandler()
    
    # Add handler only to LangChain loggers
    for logger_name in ['langchain']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(st.session_state.log_handler)
        logger.setLevel(logging.DEBUG)

st.set_page_config(
    page_title="Multi-Agent LangChain Hub", 
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent LangChain Hub")
st.sidebar.title("Agent Selection")

# LLM Provider selection
llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider:",
    ["ollama", "openai"],
    index=0,  # Default to ollama (first option)
    format_func=lambda x: "OpenAI GPT-4o-mini" if x == "openai" else "Ollama (Auto-select best model)"
)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("🔍 Enable LangChain Debug Mode", value=True, help="Shows detailed agent reasoning and tool calls in console")

# Initialize agent manager with selected provider
if 'agent_manager' not in st.session_state or st.session_state.get('current_llm_provider') != llm_provider:
    try:
        st.session_state.agent_manager = AgentManager(llm_provider=llm_provider, debug_mode=debug_mode)
        st.session_state.current_llm_provider = llm_provider
        # Clear chat history when switching providers
        st.session_state.agent_manager.clear_messages()
        st.success(f"✅ Switched to {st.session_state.agent_manager.get_current_llm()}")
    except Exception as e:
        st.error(f"❌ Failed to initialize {llm_provider}: {str(e)}")
        # Fall back to previous provider if available
        if 'agent_manager' in st.session_state:
            st.warning(f"Using previous provider: {st.session_state.agent_manager.get_current_llm()}")
        else:
            st.stop()

# Agent mode selection
agent_mode = st.sidebar.selectbox(
    "Choose Agent Mode:",
    ["Plan & Execute", "Multi-Agent Supervisor", "Interactive Search", "Hybrid Mode"]
)

# Reset chat history button
if st.sidebar.button("Reset Chat History"):
    st.session_state.agent_manager.clear_messages()
    st.rerun()

# Display current mode info
mode_descriptions = {
    "Plan & Execute": "Creates strategic plans and executes them step by step with replanning",
    "Multi-Agent Supervisor": "Coordinates specialized agents (search, web scraper, coder)",
    "Interactive Search": "Conversational search agent with memory",
    "Hybrid Mode": "Combines planning with multi-agent coordination"
}

# Mode descriptions are removed from sidebar

# Show agent graph if applicable
if agent_mode in ["Plan & Execute", "Multi-Agent Supervisor", "Hybrid Mode"]:
    graph = st.session_state.agent_manager.get_graph(agent_mode)
    if graph:
        st.sidebar.image(graph.get_graph().draw_mermaid_png(), caption=f"{agent_mode} Graph")

# Display chat history (only show messages that aren't being processed in the current run)
messages = st.session_state.agent_manager.get_messages()
for msg in messages:
    role = "assistant" if msg["role"] == "ai" else msg["role"]
    with st.chat_message(role):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process the message (this adds both user and AI messages to history)
    with st.spinner(f"{agent_mode} is processing..."):
        # Clear previous logs
        st.session_state.log_handler.get_and_clear_logs()
        
        if agent_mode in ["Plan & Execute", "Multi-Agent Supervisor", "Hybrid Mode"]:
            # Show execution details in expander
            with st.expander("See execution details", expanded=False):
                response_container = st.empty()
                final_response = st.session_state.agent_manager.process_message(
                    prompt, agent_mode, response_container
                )
            
            # Show LangChain debug output
            langchain_logs = st.session_state.log_handler.get_and_clear_logs()
            if langchain_logs and debug_mode:
                with st.expander("🔍 LangChain Debug Output", expanded=False):
                    for log in langchain_logs:
                        st.code(log, language="text")
        else:
            # Interactive search mode
            final_response = st.session_state.agent_manager.process_message(
                prompt, agent_mode
            )
    
    # Rerun to show the updated chat history with the new messages
    st.rerun()