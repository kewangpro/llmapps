from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model_name="llama3.2", base_url="http://localhost:11434/v1", openai_api_key="ollama")
tools = [DuckDuckGoSearchRun(name="Search")]
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
response = agent_executor.invoke({"messages": [("user", "who is the current US president-elect?")]})

import pprint
pprint.pprint(response)
