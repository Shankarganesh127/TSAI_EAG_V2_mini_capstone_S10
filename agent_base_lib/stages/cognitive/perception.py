from typing import List, Optional, Any, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class PerceptionInput(BaseModel):
    user_query: str


class PerceptionOutput(BaseModel):
    normalized_query: str
    intent: str
    entities: List[str] = Field(default_factory=list)
    is_clear: bool = True


class PerceptionAgent(BaseSubAgent[PerceptionInput, PerceptionOutput]):
    """
    Analyses the raw user query to extract a normalized form, primary intent,
    named entities, and a flag indicating whether the query is unambiguous.
    """

    PROMPT_TEMPLATE = """\
Analyse the following user query and extract key information.

User Query: {user_query}

Respond with a JSON object in this exact format:
{{
  "normalized_query": "<cleaned, normalized version of the query>",
  "intent": "<primary intent: search | compute | summarize | generate | other>",
  "entities": ["<entity1>", "<entity2>"],
  "is_clear": <true if the query is unambiguous, false otherwise>
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[PerceptionOutput]:
        return PerceptionOutput

    def _default_output(self, input_data: PerceptionInput) -> PerceptionOutput:
        return PerceptionOutput(
            normalized_query=input_data.user_query.strip(),
            intent="user_request",
            entities=[],
            is_clear=True,
        )
