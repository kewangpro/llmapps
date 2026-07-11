# Multi-Agent LangChain Hub

A unified application that merges three different LangChain agent patterns into one comprehensive multi-agent system.

## Features

### 🎯 Plan & Execute Mode
- Strategic planning with step-by-step execution
- Dynamic replanning based on results
- Perfect for complex multi-step tasks

### 👥 Multi-Agent Supervisor Mode  
- Supervisor coordinates specialized agents
- Search agent, web scraper, and code execution agent
- Optimal for tasks requiring different specialized tools

### 🔍 Interactive Search Mode
- Conversational search with memory
- Maintains context across interactions
- Great for research and information gathering

### 🤝 Hybrid Mode
- Combines planning with multi-agent coordination
- Assigns plan steps to most appropriate agents
- Best of both worlds for complex workflows

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key (optional - will fallback to local Ollama):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Run the application:
```bash
streamlit run app.py
```

## Architecture

- **AgentManager**: Central coordinator for all agent modes
- **Plan & Execute**: Strategic planning with execution loop
- **Supervisor Agents**: Multi-agent coordination pattern
- **Search Agent**: Conversational search with memory
- **Hybrid Agent**: Combined planning and multi-agent execution
- **Tools**: Unified toolset (search, web scraping, code execution)

## Usage

1. Select your preferred agent mode from the sidebar
2. Ask questions or give tasks in natural language
3. View execution details in the expandable sections
4. Switch between modes as needed for different types of tasks