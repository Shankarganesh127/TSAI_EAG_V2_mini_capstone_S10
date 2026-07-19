from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent

class ToolSelectionInput(BaseModel):
    normalized_query: str
    action: str
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)

class ToolSelectionOutput(BaseModel):
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

class ToolSelectionAgent(BaseSubAgent[ToolSelectionInput, ToolSelectionOutput]):
    """Select one discovered MCP tool and construct schema-shaped arguments."""

    PROMPT_TEMPLATE = """\
Select the single best MCP tool for this task, if a tool is needed.

Query: {normalized_query}
Action: {action}
Plan: {plan_steps}
Available tools (use the exact name and input schema): {tools}

Respond with a JSON object:
{{
  "tool_name": "<exact available tool name, or null>",
  "arguments": {{"<exact schema-shaped arguments>": "<values>"}},
  "rationale": "<brief reason>"
}}

Use null only when none of the tools would improve the answer. Never invent a
tool name or argument. Return ONLY the JSON object."""

    @property
    def output_model(self) -> Type[ToolSelectionOutput]:
        return ToolSelectionOutput

    def _default_output(self, input_data: ToolSelectionInput) -> ToolSelectionOutput:
        return ToolSelectionOutput()