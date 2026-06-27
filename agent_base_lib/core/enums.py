from enum import Enum


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEED_REPLAN = "need_replan"
    
class AgentState(str, Enum):
    START = "start"
    INPUT_RECEIVED = "input_received"

    PERCEPTION = "perception"
    CONTEXT_RETRIEVAL = "context_retrieval"

    DECISION = "decision"
    PLANNING = "planning"

    ACTION = "action"
    OBSERVATION = "observation"

    VALIDATION = "validation"
    REFLECTION = "reflection"
    REPLAN = "replan"

    OUTPUT = "output"
    END = "end"

    ERROR = "error"
    
    