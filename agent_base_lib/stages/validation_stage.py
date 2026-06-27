from stages.base_stage import BaseStage
from core.result import StageResult
from core.enums import AgentStatus


class ValidationStage(BaseStage):

    stage_name = "validation"

    async def execute(self, ctx):

        valid = await self.validation(ctx)

        if valid:
            return StageResult(
                status=AgentStatus.SUCCESS,
                message="validation passed",
            )

        await self.reflection(ctx)

        return StageResult(
            status=AgentStatus.NEED_REPLAN,
            message="replan required",
        )

    async def validation(self, ctx):
        return True

    async def reflection(self, ctx):
        pass