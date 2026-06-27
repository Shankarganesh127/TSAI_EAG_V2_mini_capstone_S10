from stages.base_stage import BaseStage
from core.result import StageResult
from core.enums import AgentStatus


class CognitiveStage(BaseStage):

    stage_name = "cognitive"

    async def execute(self, ctx):

        await self.perception(ctx)
        await self.context(ctx)
        await self.decision(ctx)
        await self.planning(ctx)

        return StageResult(
            status=AgentStatus.SUCCESS,
            message="cognitive completed",
        )

    async def perception(self, ctx):
        pass

    async def context(self, ctx):
        pass

    async def decision(self, ctx):
        pass

    async def planning(self, ctx):
        pass