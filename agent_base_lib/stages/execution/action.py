from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class ActionInput(BaseModel):
    normalized_query: str
    action: str
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    context: str = ""
    tool_used: Optional[str] = None
    tool_result: str = ""


class ActionOutput(BaseModel):
    response: str
    tool_used: Optional[str] = None
    success: bool = True
    raw_output: Optional[str] = None


class ActionAgent(BaseSubAgent[ActionInput, ActionOutput]):
    """
    Executes the planned action and produces a direct, complete response to the
    user query. When an LLM is available it answers the query; otherwise it
    returns a safe stub response.
    """

    PROMPT_TEMPLATE = """\
Execute the following task and provide a direct, helpful response to the user.

User Query: {normalized_query}
Action: {action}
Plan: {plan_steps}
Rationale: {rationale}
Context: {context}
Tool Used: {tool_used}
Tool Result: {tool_result}

When a tool result is provided, base the answer on that result. Clearly cite
URLs or Source fields present in it. Do not claim that no tool was used.

Respond with a JSON object:
{{
  "response": "<complete answer to the user query>",
  "tool_used": "<tool name, or null>",
  "success": true,
  "raw_output": "<any raw data or intermediate output>"
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[ActionOutput]:
        return ActionOutput

    def _default_output(self, input_data: ActionInput) -> ActionOutput:
        return ActionOutput(
            response=f"Received request: {input_data.normalized_query}",
            tool_used=None,
            success=True,
            raw_output=None,
        )
