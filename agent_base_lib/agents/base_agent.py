from typing import Any, Optional

from ..core.context import AgentContext
from ..core.enums import AgentStatus, AgentState
from ..stages.cognitive_stage import CognitiveStage
from ..stages.execution_stage import ExecutionStage
from ..stages.validation_stage import ValidationStage


class BaseAgent:

    def __init__(
        self,
        cognitive: Optional[CognitiveStage] = None,
        execution: Optional[ExecutionStage] = None,
        validation: Optional[ValidationStage] = None,
        max_loops: int = 3,
        llm_client: Optional[Any] = None,
    ):
        self.llm_client = llm_client
        self.max_loops = max_loops
        # Pass llm_client to each stage so they can use it independently
        self.cognitive  = cognitive  or CognitiveStage(llm_client=llm_client)
        self.execution  = execution  or ExecutionStage(llm_client=llm_client)
        self.validation = validation or ValidationStage(llm_client=llm_client)

    async def run(self, user_query: str) -> AgentContext:
        if not user_query or not user_query.strip():
            raise ValueError("user_query must not be empty")

        ctx = AgentContext(user_query=user_query, max_loops=self.max_loops, llm_client=self.llm_client)
        ctx.transition_to(AgentState.INPUT_RECEIVED)

        try:
            await self.cognitive.run(ctx)
        except Exception as exc:
            return self._fail(ctx, f"Cognitive stage failed: {exc}")

        while ctx.loop_count < ctx.max_loops:
            ctx.loop_count += 1

            try:
                await self.execution.run(ctx)
            except Exception as exc:
                return self._fail(ctx, f"Execution stage failed: {exc}")

            try:
                result = await self.validation.run(ctx)
            except Exception as exc:
                return self._fail(ctx, f"Validation stage failed: {exc}")

            if result.status == AgentStatus.SUCCESS:
                ctx.final_output = ctx.action_result.get("response")
                ctx.transition_to(AgentState.OUTPUT)
                ctx.transition_to(AgentState.END)
                return ctx

            if result.status == AgentStatus.FAILED:
                return self._fail(ctx, result.message or "Stage returned FAILED")

            if result.status == AgentStatus.NEED_REPLAN:
                if ctx.loop_count >= ctx.max_loops:
                    continue
                try:
                    await self.cognitive.replan(ctx)
                except Exception as exc:
                    return self._fail(ctx, f"Replanning failed: {exc}")

        return self._fail(ctx, f"Agent did not complete after {ctx.max_loops} loop(s)")

    @staticmethod
    def _fail(ctx: AgentContext, message: str) -> AgentContext:
        ctx.error = message
        ctx.transition_to(AgentState.ERROR)
        ctx.transition_to(AgentState.END)
        return ctx
