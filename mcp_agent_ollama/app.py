#!/usr/bin/env python3
"""
Streamlit web app for MCP Client
"""

import asyncio
import json
import streamlit as st
from mcp_ollama import MCPOllamaIntegration

# Page config
st.set_page_config(
    page_title="MCP Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        margin-left: 1rem;
        color: white;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None

# Sidebar
with st.sidebar:
    st.title("🤖 MCP Chat")
    st.markdown("---")
    
    # Server status
    st.subheader("Server Status")
    if st.session_state.mcp_client is None:
        st.error("Not Connected")
    else:
        st.success("Connected")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # About section
    st.markdown("---")
    st.markdown("""
    ### About
    This is an interactive chat interface for the MCP Client.
    
    Features:
    - 🤖 AI-powered responses
    - 🔍 Web search capabilities
    - ⏰ Time information
    - 💬 Natural conversations
    
    Type your message and press Enter to chat!
    """)

# Main chat interface
st.title("Chat with MCP")

# Initialize MCP client if not already done
if st.session_state.mcp_client is None:
    with st.spinner("Initializing MCP client..."):
        try:
            st.session_state.mcp_client = MCPOllamaIntegration()
            asyncio.run(st.session_state.mcp_client.initialize())
        except Exception as e:
            st.error(f"Failed to initialize MCP client: {e}")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div style="display: flex; align-items: center;">
                    <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=user" class="avatar">
                    <div class="message">{message["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div style="display: flex; align-items: center;">
                    <img src="https://api.dicebear.com/7.x/bottts/svg?seed=assistant" class="avatar">
                    <div class="message">{message["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
with st.container():
    # Create a form for the chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            placeholder="Ask me anything!",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(st.session_state.mcp_client.process_user_input(user_input))
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Rerun to update the chat
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Powered by MCP Client and Ollama
</div>
""", unsafe_allow_html=True) 