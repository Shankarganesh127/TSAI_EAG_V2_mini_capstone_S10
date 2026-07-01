from typing import Any, Optional, Type
from pydantic import BaseModel
from ..base_sub_agent import BaseSubAgent


class ObservationInput(BaseModel):
    response: str
    original_query: str
    action: str = ""


class ObservationOutput(BaseModel):
    raw: str
    has_result: bool
    quality: str = "unknown"
    completeness: float = 0.0
    notes: str = ""


class ObservationAgent(BaseSubAgent[ObservationInput, ObservationOutput]):
    """
    Observes the raw action response and scores it for quality and completeness
    relative to the original user query.
    """

    PROMPT_TEMPLATE = """\
Analyse whether the response adequately answers the user query.

Original Query: {original_query}
Action Taken: {action}
Response: {response}

Respond with a JSON object:
{{
  "raw": "<the response text verbatim>",
  "has_result": <true if a meaningful result exists>,
  "quality": "<poor | fair | good | excellent>",
  "completeness": <0.0 to 1.0>,
  "notes": "<brief observations about the response>"
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[ObservationOutput]:
        return ObservationOutput

    def _default_output(self, input_data: ObservationInput) -> ObservationOutput:
        has = bool(input_data.response)
        return ObservationOutput(
            raw=input_data.response,
            has_result=has,
            quality="fair" if has else "poor",
            completeness=0.5 if has else 0.0,
            notes="",
        )
