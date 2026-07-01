from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class PlanStep(BaseModel):
    step: int
    description: str
    tool: str = "none"


class PlanningInput(BaseModel):
    normalized_query: str
    action: str
    approach: str = ""
    rationale: str = ""


class PlanningOutput(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)
    estimated_steps: int = 1
    strategy: str = "sequential"


class PlanningAgent(BaseSubAgent[PlanningInput, PlanningOutput]):
    """
    Produces a concise, ordered execution plan derived from the decision output.
    Each step names the description and the tool (if any) to invoke.
    """

    PROMPT_TEMPLATE = """\
Create a step-by-step execution plan for the following task.

Query: {normalized_query}
Action: {action}
Approach: {approach}
Rationale: {rationale}

Respond with a JSON object:
{{
  "steps": [
    {{"step": 1, "description": "<what to do>", "tool": "<tool name or none>"}},
    ...
  ],
  "estimated_steps": <total number of steps>,
  "strategy": "<sequential | parallel>"
}}

Keep the plan concise (1-3 steps). Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[PlanningOutput]:
        return PlanningOutput

    def _default_output(self, input_data: PlanningInput) -> PlanningOutput:
        return PlanningOutput(
            steps=[PlanStep(step=1, description=input_data.action, tool="none")],
            estimated_steps=1,
            strategy="sequential",
        )
