from core.context import AgentContext
from core.enums import AgentStatus

from stages.cognitive_stage import CognitiveStage
from stages.execution_stage import ExecutionStage
from stages.validation_stage import ValidationStage


class BaseAgent:

    def __init__(self):

        self.cognitive = CognitiveStage()
        self.execution = ExecutionStage()
        self.validation = ValidationStage()

    async def run(
        self,
        user_query: str,
    ):

        ctx = AgentContext(
            user_query=user_query,
        )

        while ctx.loop_count < ctx.max_loops:

            ctx.loop_count += 1

            await self.cognitive.run(ctx)

            await self.execution.run(ctx)

            result = await self.validation.run(ctx)

            if result.status == AgentStatus.SUCCESS:
                break

        return ctx