
from .enums import AgentState, AgentStatus

# Linear stage transitions
TRANSITIONS: dict[AgentState, AgentState] = {
    AgentState.START: AgentState.INPUT_RECEIVED,
    AgentState.INPUT_RECEIVED: AgentState.PERCEPTION,
    AgentState.PERCEPTION: AgentState.CONTEXT_RETRIEVAL,
    AgentState.CONTEXT_RETRIEVAL: AgentState.DECISION,
    AgentState.DECISION: AgentState.PLANNING,
    AgentState.PLANNING: AgentState.ACTION,
    AgentState.ACTION: AgentState.OBSERVATION,
    AgentState.OBSERVATION: AgentState.VALIDATION,
    AgentState.REFLECTION: AgentState.REPLAN,
    AgentState.REPLAN: AgentState.ACTION,
    AgentState.OUTPUT: AgentState.END,
    AgentState.ERROR: AgentState.END,
}

# Status-driven transitions out of VALIDATION
STATUS_TRANSITIONS: dict[AgentStatus, AgentState] = {
    AgentStatus.SUCCESS: AgentState.OUTPUT,
    AgentStatus.NEED_REPLAN: AgentState.REFLECTION,
    AgentStatus.FAILED: AgentState.ERROR,
}


def next_state(current: AgentState, status: AgentStatus | None = None) -> AgentState:
    """Return the next state. Pass status only when current == VALIDATION."""
    if status is not None and current == AgentState.VALIDATION:
        return STATUS_TRANSITIONS.get(status, AgentState.ERROR)
    return TRANSITIONS.get(current, AgentState.END)