from typing import Any, Awaitable, Callable, Optional

from .base_stage import BaseStage
from ..core.context import AgentContext
from ..core.result import StageResult
from ..core.enums import AgentStatus, AgentState
from .validation.validator import ValidatorAgent, ValidatorInput
from .validation.reflection import ReflectionAgent, ReflectionInput


class ValidationStage(BaseStage):

    stage_name = "validation"

    def __init__(
        self,
        validator: Optional[Callable[[AgentContext], Awaitable[bool]]] = None,
        llm_client: Optional[Any] = None,
    ):
        self._validator          = validator
        self.llm_client          = llm_client
        self._validator_agent    = ValidatorAgent(llm_client=llm_client)
        self._reflection_agent   = ReflectionAgent(llm_client=llm_client)

    def _sync_client(self, ctx: AgentContext) -> None:
        """Propagate ctx.llm_client to sub-agents when stage has none."""
        client = self.llm_client or ctx.llm_client
        for agent in (self._validator_agent, self._reflection_agent):
            agent.llm_client = client

    async def execute(self, ctx: AgentContext) -> StageResult:
        self._sync_client(ctx)

        ctx.transition_to(AgentState.VALIDATION)

        if self._validator:
            passed = await self._validator(ctx)
            ctx.validation = {"passed": passed, "score": 1.0 if passed else 0.0, "issues": []}
        else:
            v_out = await self._validator_agent.run(
                ValidatorInput(
                    original_query=ctx.user_query,
                    response=ctx.action_result.get("response", ""),
                    plan_steps=ctx.plan.get("steps", []),
                    quality=ctx.observation.get("quality", "unknown"),
                )
            )
            ctx.validation = v_out.model_dump()
            passed = v_out.passed

        if passed:
            return StageResult(status=AgentStatus.SUCCESS, message="validation passed")

        ctx.transition_to(AgentState.REFLECTION)
        r_out = await self._reflection_agent.run(
            ReflectionInput(
                original_query=ctx.user_query,
                issues=ctx.validation.get("issues", ["Validation failed"]),
                loop_count=ctx.loop_count,
                score=ctx.validation.get("score", 0.0),
            )
        )
        ctx.reflection = r_out.model_dump()

        return StageResult(status=AgentStatus.NEED_REPLAN, message="replan required")