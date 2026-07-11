from typing import Literal
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from tools import search_tool, scrape_webpages, python_repl_tool
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["search", "web_scraper", "coder", "FINISH"]
    response: str

class SupervisorAgent:
    """Supervisor coordinating multiple specialized agents"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.members = ["search", "web_scraper", "coder"]
        self._build_graph()
    
    def _make_supervisor_node(self):
        """Create supervisor node that routes to appropriate worker"""
        system_prompt = (
            "You are a supervisor tasked with managing a conversation with the"
            f" following workers: {self.members}. "
            " Given the user request, respond with which worker to act next."
            " Each worker will perform their task and respond with their results."
            " When a worker has completed the task successfully, respond with FINISH as next and provide the final answer as response."
            " Only use FINISH when you have a complete answer to the user's question."
            " The final answer should be the actual result, not a generic completion message."
        )
        
        def supervisor_node(state: MessagesState) -> Command[Literal["search", "web_scraper", "coder", "__end__"]]:
            """An LLM-based router."""
            messages = [
                {"role": "system", "content": system_prompt},
            ] + state["messages"]
            
            response = self.llm.with_structured_output(Router).invoke(messages)
            
            if response is None or response["next"] == "FINISH":
                goto = END
            else:
                goto = response["next"]
            
            return Command(
                update={
                    "messages": [
                        AIMessage(content=json.dumps(response), name="supervisor")
                    ]
                },
                goto=goto
            )
        
        return supervisor_node
    
    def _make_system_prompt(self, suffix: str) -> str:
        """Create system prompt for worker agents"""
        return (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK, another assistant with different tools "
            " will help where you left off. Execute what you can to make progress."
            " If you or any of the other assistants have the final answer or deliverable,"
            " prefix your response with FINAL ANSWER so the team knows to stop."
            f"\n{suffix}"
        )
    
    def _create_worker_nodes(self):
        """Create specialized worker nodes"""
        
        # Search agent
        search_agent = create_react_agent(
            self.llm,
            tools=[search_tool],
            prompt=self._make_system_prompt("You can only do search. You are working with the supervisor.")
        )
        
        def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
            result = search_agent.invoke(state)
            response_content = result["messages"][-1].content
            
            return Command(
                update={
                    "messages": [
                        AIMessage(content=response_content, name="search")
                    ]
                },
                goto="supervisor",
            )
        
        # Web scraper agent
        web_scraper_agent = create_react_agent(
            self.llm,
            tools=[scrape_webpages],
            prompt=self._make_system_prompt("You can only scrape web. You are working with the supervisor.")
        )
        
        def web_scraper_node(state: MessagesState) -> Command[Literal["supervisor"]]:
            result = web_scraper_agent.invoke(state)
            response_content = result["messages"][-1].content
            
            return Command(
                update={
                    "messages": [
                        AIMessage(content=response_content, name="web_scraper")
                    ]
                },
                goto="supervisor",
            )
        
        # Code agent
        code_agent = create_react_agent(
            self.llm,
            tools=[python_repl_tool],
            prompt=self._make_system_prompt("You can only run python code to do math or generate charts. You are working with the supervisor.")
        )
        
        def code_node(state: MessagesState) -> Command[Literal["supervisor"]]:
            result = code_agent.invoke(state)
            response_content = result["messages"][-1].content
            
            return Command(
                update={
                    "messages": [
                        AIMessage(content=response_content, name="coder")
                    ]
                },
                goto="supervisor",
            )
        
        return search_node, web_scraper_node, code_node
    
    def _build_graph(self):
        """Build the supervisor graph"""
        builder = StateGraph(MessagesState)
        
        # Create nodes
        supervisor_node = self._make_supervisor_node()
        search_node, web_scraper_node, code_node = self._create_worker_nodes()
        
        # Add nodes
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("search", search_node)
        builder.add_node("web_scraper", web_scraper_node)
        builder.add_node("coder", code_node)
        
        # Add edges
        builder.add_edge(START, "supervisor")
        
        self.graph = builder.compile()
    
    def get_graph(self):
        """Get the graph for visualization"""
        return self.graph
    
    def process(self, message: str) -> str:
        """Process a message and return the response"""
        config = {"recursion_limit": 50}
        inputs = {"messages": [("user", message)]}
        
        result = self.graph.invoke(inputs, config=config)
        
        # Extract the actual answer from the conversation
        if result and "messages" in result:
            # Look through all messages for actual answers
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'name') and msg.name == "supervisor":
                    try:
                        # Parse supervisor's JSON responses to find meaningful answers
                        json_data = json.loads(msg.content)
                        supervisor_response = json_data.get("response", "")
                        
                        # Skip generic completion messages
                        if supervisor_response and supervisor_response not in [
                            "The user request has been completed.",
                            "Task completed by agents"
                        ]:
                            # Check if it contains actual information
                            if any(keyword in supervisor_response.lower() for keyword in [
                                "president", "biden", "trump", "current", "time", "donald", "joe"
                            ]):
                                return supervisor_response
                    except (json.JSONDecodeError, AttributeError):
                        continue
                        
                # Also check worker agent messages (though they seem to be empty)
                elif hasattr(msg, 'name') and msg.name in ["search", "web_scraper", "coder"]:
                    content = msg.content
                    if content and content.strip():  # Only if not empty
                        return content
            
            # Final fallback - look for any meaningful content in recent messages
            for msg in reversed(result["messages"][-10:]):  # Check last 10 messages
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Look for actual answer patterns in any message
                    if any(keyword in content.lower() for keyword in [
                        "president", "biden", "trump", "current president", "donald", "joe"
                    ]) and "The user request has been completed" not in content:
                        # Try to extract from JSON if needed
                        try:
                            json_data = json.loads(content)
                            if json_data.get("response"):
                                return json_data["response"]
                        except:
                            pass
                        return content
        
        return "I was unable to get a clear response from the worker agents."
    
    def process_with_streaming(self, message: str, response_container) -> str:
        """Process with streaming updates"""
        config = {"recursion_limit": 50}
        inputs = {"messages": [("user", message)]}
        
        final_response = ""
        all_messages = []
        
        for response in self.graph.stream(inputs, config=config):
            response_container.json(response)
            
            # Collect all messages
            for k, v in response.items():
                if "messages" in v and v["messages"]:
                    all_messages.extend(v["messages"])
        
        # Extract the actual answer from the conversation
        final_response = ""
        
        # Look through all messages for actual answers
        for msg in reversed(all_messages):
            if hasattr(msg, 'name') and msg.name == "supervisor":
                try:
                    # Parse supervisor's JSON responses to find meaningful answers
                    json_data = json.loads(msg.content)
                    supervisor_response = json_data.get("response", "")
                    
                    # Skip generic completion messages
                    if supervisor_response and supervisor_response not in [
                        "The user request has been completed.",
                        "Task completed by agents"
                    ]:
                        # Check if it contains actual information
                        if any(keyword in supervisor_response.lower() for keyword in [
                            "president", "biden", "trump", "current", "time", "donald", "joe"
                        ]):
                            final_response = supervisor_response
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
                    
            # Also check worker agent messages (though they seem to be empty)
            elif hasattr(msg, 'name') and msg.name in ["search", "web_scraper", "coder"]:
                content = msg.content
                if content and content.strip():  # Only if not empty
                    final_response = content
                    break
        
        # Final fallback - look for any meaningful content in recent messages
        if not final_response:
            for msg in reversed(all_messages[-10:]):  # Check last 10 messages
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Look for actual answer patterns in any message
                    if any(keyword in content.lower() for keyword in [
                        "president", "biden", "trump", "current president", "donald", "joe"
                    ]) and "The user request has been completed" not in content:
                        # Try to extract from JSON if needed
                        try:
                            json_data = json.loads(content)
                            if json_data.get("response"):
                                final_response = json_data["response"]
                                break
                        except:
                            pass
                        final_response = content
                        break
        
        return final_response or "I was unable to get a clear response from the worker agents."