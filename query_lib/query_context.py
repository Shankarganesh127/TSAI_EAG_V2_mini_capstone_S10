from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ExpectedOutput(BaseModel):
    format: Optional[str] = None
    contains: List[str] = Field(default_factory=list)


class AgentRouting(BaseModel):
    perception_agent: bool = True
    memory_agent: bool = True
    planning_agent: bool = True
    action_agent: bool = False
    validation_agent: bool = True
    reflection_agent: bool = False


class Complexity(BaseModel):
    level: Literal["low", "medium", "high"] = "low"
    requires_tools: bool = False
    requires_memory: bool = False
    requires_validation: bool = True


class QueryDissection(BaseModel):
    raw_query: str = Field(..., description="Original user query")
    normalized_query: Optional[str] = None

    intent: Optional[str] = None
    main_goal: Optional[str] = None
    sub_goals: List[str] = Field(default_factory=list)

    entities: Dict[str, Any] = Field(default_factory=dict)
    task_type: List[str] = Field(default_factory=list)

    conversation_history: List[str] = Field(default_factory=list)
    known_information: Dict[str, Any] = Field(default_factory=dict)
    relevant_memory: List[str] = Field(default_factory=list)

    missing_information: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    required_tools: List[str] = Field(default_factory=list)
    expected_output: ExpectedOutput = Field(default_factory=ExpectedOutput)
    success_criteria: List[str] = Field(default_factory=list)

    risks: List[str] = Field(default_factory=list)
    complexity: Complexity = Field(default_factory=Complexity)

    agent_routing: AgentRouting = Field(default_factory=AgentRouting)
    planning_input: Dict[str, Any] = Field(default_factory=dict)
    
    