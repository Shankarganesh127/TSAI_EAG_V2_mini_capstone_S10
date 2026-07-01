from typing import Any, Optional

from .base_stage import BaseStage
from ..core.context import AgentContext
from ..core.result import StageResult
from ..core.enums import AgentStatus, AgentState
from .cognitive.perception import PerceptionAgent, PerceptionInput
from .cognitive.context_retrieval import ContextRetrievalAgent, ContextInput
from .cognitive.decision import DecisionAgent, DecisionInput
from .cognitive.planning import PlanningAgent, PlanningInput


class CognitiveStage(BaseStage):

    stage_name = "cognitive"

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client          = llm_client
        self._perception_agent   = PerceptionAgent(llm_client=llm_client)
        self._context_agent      = ContextRetrievalAgent(llm_client=llm_client)
        self._decision_agent     = DecisionAgent(llm_client=llm_client)
        self._planning_agent     = PlanningAgent(llm_client=llm_client)

    def _sync_client(self, ctx: AgentContext) -> None:
        """Propagate ctx.llm_client to sub-agents when stage has none."""
        client = self.llm_client or ctx.llm_client
        for agent in (self._perception_agent, self._context_agent,
                      self._decision_agent, self._planning_agent):
            agent.llm_client = client

    async def execute(self, ctx: AgentContext) -> StageResult:
        self._sync_client(ctx)

        ctx.transition_to(AgentState.PERCEPTION)
        p_out = await self._perception_agent.run(
            PerceptionInput(user_query=ctx.user_query)
        )
        ctx.perception = p_out.model_dump()

        ctx.transition_to(AgentState.CONTEXT_RETRIEVAL)
        c_out = await self._context_agent.run(
            ContextInput(
                normalized_query=p_out.normalized_query,
                intent=p_out.intent,
                entities=p_out.entities,
            )
        )
        ctx.context = c_out.model_dump()

        ctx.transition_to(AgentState.DECISION)
        d_out = await self._decision_agent.run(
            DecisionInput(
                normalized_query=p_out.normalized_query,
                intent=p_out.intent,
                entities=p_out.entities,
                context=c_out.relevant_context,
            )
        )
        ctx.decision = d_out.model_dump()

        ctx.transition_to(AgentState.PLANNING)
        pl_out = await self._planning_agent.run(
            PlanningInput(
                normalized_query=p_out.normalized_query,
                action=d_out.action,
                approach=d_out.approach,
                rationale=d_out.rationale,
            )
        )
        ctx.plan = pl_out.model_dump()

        return StageResult(status=AgentStatus.SUCCESS, message="cognitive completed")