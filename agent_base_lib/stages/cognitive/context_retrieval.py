from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from ..base_sub_agent import BaseSubAgent


class ContextInput(BaseModel):
    normalized_query: str
    intent: str
    entities: List[str] = Field(default_factory=list)


class ContextOutput(BaseModel):
    relevant_context: str = ""
    memory: List[Any] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    notes: str = ""


class ContextRetrievalAgent(BaseSubAgent[ContextInput, ContextOutput]):
    """
    Identifies what background context, memory, and tools are relevant to
    the current query before a decision is made.
    """

    PROMPT_TEMPLATE = """\
Given the user intent and entities, identify the context needed to answer.

Query: {normalized_query}
Intent: {intent}
Entities: {entities}

Respond with a JSON object:
{{
  "relevant_context": "<summary of context that would help answer this query>",
  "memory": [],
  "tools_available": ["web_search", "document_search", "calculator"],
  "notes": "<what additional info might be needed>"
}}

Return ONLY the JSON object with no extra text."""

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client=llm_client)

    @property
    def output_model(self) -> Type[ContextOutput]:
        return ContextOutput

    def _default_output(self, input_data: ContextInput) -> ContextOutput:
        return ContextOutput(
            relevant_context="",
            memory=[],
            tools_available=[],
            notes="",
        )
