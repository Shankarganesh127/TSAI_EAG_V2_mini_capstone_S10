import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable, Optional

from .base_stage import BaseStage
from ..core.context import AgentContext
from ..core.result import StageResult
from ..core.enums import AgentStatus, AgentState
from .execution.action import ActionAgent, ActionInput
from .execution.observation import ObservationAgent, ObservationInput
from .execution.tool_selection import (
    ToolSelectionAgent,
    ToolSelectionInput,
    ToolSelectionOutput,
)


class ExecutionStage(BaseStage):

    stage_name = "execution"

    def __init__(
        self,
        executor: Optional[Callable[[AgentContext], Awaitable[str]]] = None,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        tool_timeout_seconds: float = 45.0,
    ):
        self._explicit_executor = executor
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.tool_timeout_seconds = tool_timeout_seconds
        self._action_agent = ActionAgent(llm_client=llm_client)
        self._observation_agent = ObservationAgent(llm_client=llm_client)
        self._tool_agent = ToolSelectionAgent(llm_client=llm_client)

    def _sync_client(self, ctx: AgentContext) -> None:
        """Propagate ctx.llm_client to sub-agents when stage has none."""
        self.sync_agent_clients(
            self.llm_client,
            ctx.llm_client,
            (self._action_agent, self._observation_agent, self._tool_agent),
        )

    async def execute(self, ctx: AgentContext) -> StageResult:
        self._sync_client(ctx)
        ctx.transition_to(AgentState.ACTION)
        await self._action(ctx)
        ctx.transition_to(AgentState.OBSERVATION)
        await self._observation(ctx)
        return StageResult(status=AgentStatus.SUCCESS, message="execution completed")

    async def _action(self, ctx: AgentContext) -> None:
        tool_name: Optional[str] = None
        tool_result = ""
        if self._explicit_executor:
            response = await self._explicit_executor(ctx)
        else:
            tool_name, tool_result = await self._execute_tool(ctx)
            output = await self._action_agent.run(
                ActionInput(
                    normalized_query=ctx.perception.get("normalized_query", ctx.user_query),
                    action=ctx.decision.get("action", "execute_query"),
                    plan_steps=ctx.plan.get("steps", []),
                    rationale=ctx.decision.get("rationale", ""),
                    context=ctx.context.get("relevant_context", ""),
                    tool_used=tool_name,
                    tool_result=tool_result,
                )
            )
            response = output.response
        ctx.action_result = {
            "response": response,
            "tool_used": tool_name,
            "tool_result": tool_result,
        }

    async def _execute_tool(self, ctx: AgentContext) -> tuple[Optional[str], str]:
        """Select and invoke one discovered MCP tool for the current action."""
        if self.tool_registry is None or not self.tool_registry.tool_map:
            return None, ""

        current_query = ctx.perception.get("normalized_query", ctx.user_query)
        if (
            self._is_current_time_query(ctx.user_query)
            and "current_time" in self.tool_registry.tool_map
        ):
            selection = ToolSelectionOutput(
                tool_name="current_time",
                arguments={
                    "input": {
                        "timezones": self._timezones_from_query(ctx.user_query),
                    }
                },
                rationale="Current time requires a deterministic timezone clock.",
            )
        else:
            selection = await self._tool_agent.run(
                ToolSelectionInput(
                    normalized_query=current_query,
                    action=ctx.decision.get("action", "execute_query"),
                    plan_steps=ctx.plan.get("steps", []),
                    tools=self.tool_registry.describe_tools(),
                )
            )
        if selection.tool_name is None:
            if (
                ctx.decision.get("action") == "search_web"
                and "duckduckgo_search_results" in self.tool_registry.tool_map
            ):
                selection.tool_name = "duckduckgo_search_results"
                selection.arguments = {
                    "input": {
                        "query": ctx.perception.get("normalized_query", ctx.user_query),
                        "max_results": 5,
                    }
                }
            else:
                return None, ""
        if selection.tool_name not in self.tool_registry.tool_map:
            raise ValueError(f"Selected MCP tool is unavailable: {selection.tool_name}")

        started = time.perf_counter()
        ctx.emit_event(
            "mcp_tool_call",
            {
                "tool": selection.tool_name,
                "arguments": selection.arguments,
                "action": ctx.decision.get("action", ""),
            },
        )
        try:
            raw = await asyncio.wait_for(
                self.tool_registry.call_tool(
                    selection.tool_name,
                    selection.arguments,
                ),
                timeout=self.tool_timeout_seconds,
            )
            result = self._tool_result_text(raw)
        except Exception as exc:
            ctx.emit_event(
                "mcp_tool_error",
                {
                    "tool": selection.tool_name,
                    "arguments": selection.arguments,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": str(exc),
                },
            )
            raise

        ctx.emit_event(
            "mcp_tool_result",
            {
                "tool": selection.tool_name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": result,
            },
        )
        return selection.tool_name, result

    @staticmethod
    def _is_current_time_query(query: str) -> bool:
        lowered = query.lower()
        return any(
            phrase in lowered
            for phrase in ("current time", "local time", "what time is it", "time now")
        )

    @staticmethod
    def _timezones_from_query(query: str) -> list[str]:
        """Resolve browser-provided IANA zones and common requested cities."""
        zones = re.findall(r"\b[A-Za-z]+/[A-Za-z0-9_+\-]+\b", query)
        lowered = query.lower()
        aliases = (
            (("sheffield", "london", "united kingdom", " uk"), "Europe/London"),
            (("new york",), "America/New_York"),
            (("paris",), "Europe/Paris"),
            (("tokyo",), "Asia/Tokyo"),
            (("sydney",), "Australia/Sydney"),
        )
        for names, timezone_name in aliases:
            if any(name in lowered for name in names):
                zones.append(timezone_name)
        if not zones:
            zones.append("UTC")
        return list(dict.fromkeys(zones))

    @staticmethod
    def _tool_result_text(raw: Any) -> str:
        """Normalize MCP CallToolResult content for prompts and state."""
        blocks = getattr(raw, "content", None)
        if blocks:
            return "\n".join(getattr(block, "text", str(block)) for block in blocks)
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False)
        return str(raw)

    async def _observation(self, ctx: AgentContext) -> None:
        output = await self._observation_agent.run(
            ObservationInput(
                response=ctx.action_result.get("response", ""),
                original_query=ctx.user_query,
                action=ctx.decision.get("action", ""),
            )
        )
        ctx.observation = output.model_dump()