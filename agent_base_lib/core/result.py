from pydantic import BaseModel, Field
from typing import Any, Dict
from .enums import AgentStatus


class StageResult(BaseModel):

    status: AgentStatus

    output: Dict[str, Any] = Field(default_factory=dict)

    message: str = ""