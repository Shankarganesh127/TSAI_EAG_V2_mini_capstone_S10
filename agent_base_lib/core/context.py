import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import AgentState

_log = logging.getLogger(__name__)


class AgentContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    user_query: str
    llm_client: Optional[Any] = None
    event_handler: Optional[Any] = Field(default=None, exclude=True)

    current_state: AgentState = AgentState.START
    state_history: List[AgentState] = Field(default_factory=list)

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

    def emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Write an agent event to application and structured session logs."""
        _log.info("[%s] %s", event, payload)
        if self.event_handler is None:
            return
        try:
            self.event_handler(event, payload)
        except Exception:
            _log.exception("Could not persist agent event '%s'", event)

    def transition_to(self, state: AgentState) -> None:
        """Record and log every state transition."""
        previous = self.current_state
        self.state_history.append(previous)
        self.current_state = state
        self.emit_event(
            "state_transition",
            {
                "from_state": previous.value,
                "to_state": state.value,
                "loop_count": self.loop_count,
                "query": self.user_query,
            },
        )