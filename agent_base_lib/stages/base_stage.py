from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..core.context import AgentContext
from ..core.result import StageResult


class BaseStage(ABC):

    stage_name = "base"

    async def run(self, ctx: AgentContext) -> StageResult:
        return await self.execute(ctx)

    @staticmethod
    def sync_agent_clients(
        stage_client: Any,
        context_client: Any,
        agents: Iterable[Any],
    ) -> None:
        """Apply the effective LLM client to a stage's sub-agents."""
        client = stage_client or context_client
        for agent in agents:
            agent.llm_client = client

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> StageResult:
        pass
