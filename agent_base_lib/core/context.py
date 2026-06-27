from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class AgentContext(BaseModel):

    user_query: str

    loop_count: int = 0
    max_loops: int = 3

    perception: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    decision: Dict[str, Any] = Field(default_factory=dict)
    plan: Dict[str, Any] = Field(default_factory=dict)

    action_result: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)

    validation: Dict[str, Any] = Field(default_factory=dict)
    reflection: Dict[str, Any] = Field(default_factory=dict)

    final_output: Optional[Any] = None
    error: Optional[str] = None