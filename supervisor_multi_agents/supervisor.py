from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    #options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. "
        " Given the following user request, respond with which worker to act next."
        " Each worker will perform a task and respond with their results and status."
        " When finished, respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        #next: Literal[*options]
        next: Literal["search", "web_scraper", "coder", "FINISH"]

    #def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    def supervisor_node(state: MessagesState) -> Command[Literal["search", "web_scraper", "coder", "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node
