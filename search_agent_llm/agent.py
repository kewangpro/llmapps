from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

llm = ChatOpenAI(model_name="llama3.2", base_url="http://localhost:11434/v1", openai_api_key="ollama", disable_streaming=False)
tools = [DuckDuckGoSearchRun(name="Search")]
chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

executor = AgentExecutor.from_agent_and_tools(
    agent=chat_agent,
    tools=tools,
    memory=memory,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

#response = executor.invoke({"input": "Who is the current president-elect?", "chat_history": []})
#import pprint
#pprint.pprint(response)
