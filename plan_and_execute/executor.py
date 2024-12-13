from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("ih/ih-react-agent-executor")
#prompt.pretty_print()

# Choose the LLM that will drive the agent
#llm = ChatOpenAI(model_name="llama3.2", base_url="http://localhost:11434/v1", openai_api_key="ollama")
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key="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")
tools = [DuckDuckGoSearchRun(name="Search")]
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
#response = agent_executor.invoke(
#    {
#        "messages": [
#            ("user", "who is the current US president-elect?")
#        ]
#    })
#print(response)
