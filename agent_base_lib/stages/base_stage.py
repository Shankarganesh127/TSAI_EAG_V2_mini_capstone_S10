from abc import ABC, abstractmethod

from ..core.context import AgentContext
from ..core.result import StageResult


class BaseStage(ABC):

    stage_name = "base"

    async def run(self, ctx: AgentContext) -> StageResult:
        return await self.execute(ctx)

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> StageResult:
        pass