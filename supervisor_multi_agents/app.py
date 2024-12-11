from langgraph.graph import StateGraph, MessagesState, START
from supervisor import make_supervisor_node
from agents import search_node, web_scraper_node, code_node, llm

research_builder = StateGraph(MessagesState)

research_builder.add_node("supervisor", make_supervisor_node(llm, ["search", "web_scraper", "coder"]))
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)
research_builder.add_node("coder", code_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


config = {"recursion_limit": 50}
inputs = {"messages": [("user", "Find the latest GDP of New York, California, Oregon and Washington, then calculate the median")]}

for s in research_graph.stream(inputs, config=config):
    print(s)
    print("---")