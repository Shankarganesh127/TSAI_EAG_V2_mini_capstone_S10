from stages.base_stage import BaseStage
from core.result import StageResult
from core.enums import AgentStatus


class ExecutionStage(BaseStage):

    stage_name = "execution"

    async def execute(self, ctx):

        await self.action(ctx)
        await self.observation(ctx)

        return StageResult(
            status=AgentStatus.SUCCESS,
            message="execution completed",
        )

    async def action(self, ctx):
        pass

    async def observation(self, ctx):
        pass