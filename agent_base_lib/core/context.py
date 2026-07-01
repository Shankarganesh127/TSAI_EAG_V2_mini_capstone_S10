from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .enums import AgentState


class AgentContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    user_query: str

    # Shared LLM client — available to every stage via ctx
    llm_client: Optional[Any] = None

    # State machine
    current_state: AgentState = AgentState.START
    state_history: List[AgentState] = Field(default_factory=list)

    loop_count: int = 0
    max_loops: int = 3

    # Cognitive outputs
    perception: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    decision: Dict[str, Any] = Field(default_factory=dict)
    plan: Dict[str, Any] = Field(default_factory=dict)

    # Execution outputs
    action_result: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)

    # Validation outputs
    validation: Dict[str, Any] = Field(default_factory=dict)
    reflection: Dict[str, Any] = Field(default_factory=dict)

    final_output: Optional[Any] = None
    error: Optional[str] = None

    def transition_to(self, state: AgentState) -> None:
        """Record current state in history and move to the new state."""
        self.state_history.append(self.current_state)
        self.current_state = state