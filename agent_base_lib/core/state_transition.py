
from .enums import AgentState

TRANSITIONS = {
    AgentState.START: AgentState.INPUT_RECEIVED,
    AgentState.INPUT_RECEIVED: AgentState.PERCEPTION,
    AgentState.PERCEPTION: AgentState.CONTEXT_RETRIEVAL,
    AgentState.CONTEXT_RETRIEVAL: AgentState.DECISION,
    AgentState.DECISION: AgentState.PLANNING,
    AgentState.PLANNING: AgentState.ACTION,
    AgentState.ACTION: AgentState.OBSERVATION,
    AgentState.OBSERVATION: AgentState.VALIDATION,

    # Conditional transition after validation
    AgentState.VALIDATION: {
        "success": AgentState.OUTPUT,
        "failure": AgentState.REFLECTION,
    },

    AgentState.REFLECTION: AgentState.REPLAN,
    AgentState.REPLAN: AgentState.ACTION,
    AgentState.OUTPUT: AgentState.END,
}