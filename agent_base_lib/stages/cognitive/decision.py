from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class DecisionInput(BaseModel):
    normalized_query: str
    intent: str
    entities: List[str] = Field(default_factory=list)
    context: str = ""


class DecisionOutput(BaseModel):
    action: str
    rationale: str
    approach: str = ""
    confidence: float = 0.8


class DecisionAgent(BaseSubAgent[DecisionInput, DecisionOutput]):
    """
    Decides the best action to take based on the perceived query and retrieved
    context, producing a structured decision with a rationale and confidence.
    """

    PROMPT_TEMPLATE = """\
Based on the query and available context, decide the best action to take.

Query: {normalized_query}
Intent: {intent}
Entities: {entities}
Context: {context}

Respond with a JSON object:
{{
  "action": "<primary action: execute_query | search_web | compute | summarize | other>",
  "rationale": "<brief explanation of why this action>",
  "approach": "<how to approach the task step by step>",
  "confidence": <0.0 to 1.0>
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[DecisionOutput]:
        return DecisionOutput

    def _default_output(self, input_data: DecisionInput) -> DecisionOutput:
        return DecisionOutput(
            action="execute_query",
            rationale=f"Process: {input_data.normalized_query}",
            approach="direct_response",
            confidence=0.8,
        )
