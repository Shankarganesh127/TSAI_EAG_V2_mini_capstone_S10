from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class ReflectionInput(BaseModel):
    original_query: str
    issues: List[str] = Field(default_factory=list)
    loop_count: int = 1
    score: float = 0.0


class ReflectionOutput(BaseModel):
    reason: str
    suggested_approach: str = ""
    should_retry: bool = True
    notes: str = ""


class ReflectionAgent(BaseSubAgent[ReflectionInput, ReflectionOutput]):
    """
    Reflects on why validation failed and proposes an improved approach for
    the next execution attempt.
    """

    PROMPT_TEMPLATE = """\
Reflect on why the current attempt failed validation and suggest an improvement.

Original Query: {original_query}
Validation Issues: {issues}
Attempt Number: {loop_count}
Quality Score: {score}

Respond with a JSON object:
{{
  "reason": "<why the response failed validation>",
  "suggested_approach": "<what to try differently on the next attempt>",
  "should_retry": <true if retrying is likely to produce a better result>,
  "notes": "<any additional reflection>"
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[ReflectionOutput]:
        return ReflectionOutput

    def _default_output(self, input_data: ReflectionInput) -> ReflectionOutput:
        return ReflectionOutput(
            reason="; ".join(input_data.issues) if input_data.issues else "Validation failed",
            suggested_approach="Retry with a more specific approach",
            should_retry=input_data.loop_count < 3,
            notes="",
        )
