from typing import Any, Awaitable, Callable, Optional

from .base_stage import BaseStage
from ..core.context import AgentContext
from ..core.result import StageResult
from ..core.enums import AgentStatus, AgentState
from .execution.action import ActionAgent, ActionInput
from .execution.observation import ObservationAgent, ObservationInput


class ExecutionStage(BaseStage):

    stage_name = "execution"

    def __init__(
        self,
        executor: Optional[Callable[[AgentContext], Awaitable[str]]] = None,
        llm_client: Optional[Any] = None,
    ):
        self._explicit_executor  = executor
        self.llm_client          = llm_client
        self._action_agent       = ActionAgent(llm_client=llm_client)
        self._observation_agent  = ObservationAgent(llm_client=llm_client)

    def _sync_client(self, ctx: AgentContext) -> None:
        """Propagate ctx.llm_client to sub-agents when stage has none."""
        self.sync_agent_clients(
            self.llm_client,
            ctx.llm_client,
            (self._action_agent, self._observation_agent),
        )

    async def execute(self, ctx: AgentContext) -> StageResult:
        self._sync_client(ctx)

        ctx.transition_to(AgentState.ACTION)
        await self._action(ctx)

        ctx.transition_to(AgentState.OBSERVATION)
        await self._observation(ctx)

        return StageResult(status=AgentStatus.SUCCESS, message="execution completed")

    async def _action(self, ctx: AgentContext) -> None:
        if self._explicit_executor:
            response = await self._explicit_executor(ctx)
        else:
            a_out = await self._action_agent.run(
                ActionInput(
                    normalized_query=ctx.perception.get("normalized_query", ctx.user_query),
                    action=ctx.decision.get("action", "execute_query"),
                    plan_steps=ctx.plan.get("steps", []),
                    rationale=ctx.decision.get("rationale", ""),
                    context=ctx.context.get("relevant_context", ""),
                )
            )
            response = a_out.response
        ctx.action_result = {"response": response}

    async def _observation(self, ctx: AgentContext) -> None:
        o_out = await self._observation_agent.run(
            ObservationInput(
                response=ctx.action_result.get("response", ""),
                original_query=ctx.user_query,
                action=ctx.decision.get("action", ""),
            )
        )
        ctx.observation = o_out.model_dump()
