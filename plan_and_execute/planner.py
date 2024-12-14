from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union, Literal
from langchain_core.prompts import ChatPromptTemplate
from executor import llm

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


class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
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

If no more steps are needed and you can return to the user, then respond to user with that. Do not return previously done steps as part of the plan."""
)
#replanner_prompt.pretty_print()
replanner = replanner_prompt | llm.with_structured_output(Act)
