from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class ValidatorInput(BaseModel):
    original_query: str
    response: str
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)
    quality: str = "unknown"


class ValidatorOutput(BaseModel):
    passed: bool
    score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    notes: str = ""


class ValidatorAgent(BaseSubAgent[ValidatorInput, ValidatorOutput]):
    """
    Validates whether the action response satisfactorily and accurately answers
    the original user query, returning a pass/fail decision with issues.
    """

    PROMPT_TEMPLATE = """\
Validate whether the response correctly and completely answers the user query.

Original Query: {original_query}
Response: {response}
Expected Plan: {plan_steps}
Observed Quality: {quality}

Respond with a JSON object:
{{
  "passed": <true if the response satisfactorily answers the query>,
  "score": <0.0 to 1.0>,
  "issues": ["<issue1>", "<issue2>"],
  "notes": "<validation notes>"
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[ValidatorOutput]:
        return ValidatorOutput

    def _default_output(self, input_data: ValidatorInput) -> ValidatorOutput:
        passed = bool(input_data.response)
        return ValidatorOutput(
            passed=passed,
            score=0.8 if passed else 0.0,
            issues=[] if passed else ["No valid response produced"],
            notes="",
        )
