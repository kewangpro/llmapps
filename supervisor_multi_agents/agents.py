from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langgraph.types import Command
from typing import Literal
from tools import search_tool, scrape_webpages, python_repl_tool


#llm = ChatOpenAI(model_name="llama3.2", base_url="http://localhost:11434/v1", openai_api_key="ollama")
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key="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")


search_agent = create_react_agent(llm, tools=[search_tool])

def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])

def web_scraper_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_react_agent(llm, tools=[python_repl_tool])

def code_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )
