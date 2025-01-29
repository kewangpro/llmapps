from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union, Literal
from langchain_core.prompts import ChatPromptTemplate
import os

# Get the prompt to use - you can modify this!
#prompt = hub.pull("ih/ih-react-agent-executor")
prompt = "You are a helpful assistant."
#prompt.pretty_print()

openai_api_key = os.environ["OPENAI_API_KEY"]
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
#llm = ChatOpenAI(model_name="llama3.2", base_url="http://localhost:11434/v1", openai_api_key="ollama")
tools = [DuckDuckGoSearchRun(name="Search")]
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
#response = agent_executor.invoke(
#    {
#        "messages": [
#            ("user", "who is the current US president-elect?")
#        ]
#    })
#print(response)


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(description="different steps to follow, should be in sorted order as numbered list of strings")

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ])
#planner_prompt.pretty_print()

planner = planner_prompt | llm.with_structured_output(Plan)
#response = planner.invoke(
#    {
#        "messages": [
#            ("user", "what is the hometown of the current US president-elect?")
#        ]
#    })
#print(response)


class Answer(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Answer, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Answer. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{executed}

If no more steps are needed and you can return to the user, then respond with that.
Otherwise, Update your plan accordingly. Do not return previously done steps as part of the plan."""
)
#replanner_prompt.pretty_print()

replanner = replanner_prompt | llm.with_structured_output(Act)
#response = replanner.invoke(
#    {
#        "input": "what is the hometown of the current US president-elect?",
#        "plan": "['Find out who the current US President-Elect is', 'Check their official biography or news articles to find their hometown']",
#        "executed": "'Who is the current US president-elect?'",
#    })
#print(response)
