import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import json

from agent_base_lib import BaseAgent, AgentContext, AgentStatus, AgentState, StageResult
from agent_base_lib.stages.execution_stage import ExecutionStage
from agent_base_lib.stages.validation_stage import ValidationStage
from agent_base_lib.stages.cognitive_stage import CognitiveStage
from agent_base_lib.stages.execution.action import ActionAgent, ActionInput, ActionOutput
from agent_base_lib.stages.execution.tool_selection import ToolSelectionOutput


class TestDefaultPipeline(unittest.IsolatedAsyncioTestCase):

    async def test_successful_default_pipeline(self):
        agent = BaseAgent()
        ctx = await agent.run("Find current F1 standings")
        self.assertIsNone(ctx.error)
        self.assertIsNotNone(ctx.final_output)
        self.assertIn("F1 standings", ctx.final_output)
        self.assertEqual(ctx.current_state, AgentState.END)

    async def test_state_history_populated(self):
        agent = BaseAgent()
        ctx = await agent.run("test query")
        self.assertIn(AgentState.START, ctx.state_history)
        self.assertIn(AgentState.INPUT_RECEIVED, ctx.state_history)
        self.assertEqual(ctx.current_state, AgentState.END)

    async def test_every_state_transition_emits_an_event(self):
        events = []
        agent = BaseAgent(event_handler=lambda event, payload: events.append(
            (event, payload)
        ))

        await agent.run("trace this query")

        transitions = [payload for event, payload in events
                       if event == "state_transition"]
        self.assertGreater(len(transitions), 0)
        self.assertEqual(transitions[0]["from_state"], "start")
        self.assertEqual(transitions[0]["to_state"], "input_received")
        self.assertEqual(transitions[-1]["to_state"], "end")
        self.assertTrue(all("loop_count" in item for item in transitions))
    async def test_cognitive_populates_context(self):
        agent = BaseAgent()
        ctx = await agent.run("some query")
        self.assertIn("normalized_query", ctx.perception)
        self.assertEqual(ctx.perception["normalized_query"], "some query")
        self.assertIn("steps", ctx.plan)


class TestCustomExecutor(unittest.IsolatedAsyncioTestCase):

    async def test_custom_injected_executor(self):
        async def my_executor(ctx: AgentContext) -> str:
            return f"Custom result for: {ctx.user_query}"

        agent = BaseAgent(execution=ExecutionStage(executor=my_executor))
        ctx = await agent.run("test query")
        self.assertEqual(ctx.final_output, "Custom result for: test query")
        self.assertIsNone(ctx.error)
        self.assertEqual(ctx.current_state, AgentState.END)


class TestValidationRetry(unittest.IsolatedAsyncioTestCase):

    async def test_failed_validation_replans_before_retry(self):
        approaches = []

        async def recording_executor(ctx: AgentContext) -> str:
            approaches.append(ctx.decision.get("approach"))
            return f"attempt {len(approaches)}"

        async def pass_second_attempt(ctx: AgentContext) -> bool:
            return ctx.loop_count == 2

        agent = BaseAgent(
            execution=ExecutionStage(executor=recording_executor),
            validation=ValidationStage(validator=pass_second_attempt),
            max_loops=2,
        )
        ctx = await agent.run("some query")

        self.assertIsNone(ctx.error)
        self.assertEqual(ctx.final_output, "attempt 2")
        self.assertEqual(approaches[0], "direct_response")
        self.assertEqual(approaches[1], "Retry with a more specific approach")
        self.assertIn(AgentState.REPLAN, ctx.state_history)

    async def test_validation_retry_and_max_loop_failure(self):
        async def always_fail(ctx: AgentContext) -> bool:
            return False

        agent = BaseAgent(
            validation=ValidationStage(validator=always_fail),
            max_loops=2,
        )
        ctx = await agent.run("some query")
        self.assertIsNotNone(ctx.error)
        self.assertIsNone(ctx.final_output)
        self.assertEqual(ctx.current_state, AgentState.END)
        self.assertEqual(ctx.loop_count, 2)

    async def test_reflection_populated_on_failed_validation(self):
        async def always_fail(ctx: AgentContext) -> bool:
            return False

        agent = BaseAgent(
            validation=ValidationStage(validator=always_fail),
            max_loops=1,
        )
        ctx = await agent.run("some query")
        self.assertIn("reason", ctx.reflection)


class TestEmptyQuery(unittest.IsolatedAsyncioTestCase):

    async def test_empty_query_raises(self):
        agent = BaseAgent()
        with self.assertRaises(ValueError):
            await agent.run("")

    async def test_whitespace_query_raises(self):
        agent = BaseAgent()
        with self.assertRaises(ValueError):
            await agent.run("   ")


class TestExceptionHandling(unittest.IsolatedAsyncioTestCase):

    async def test_configured_llm_failure_is_not_reported_as_success(self):
        class BrokenClient:
            def chat(self, prompt):
                raise RuntimeError("provider unavailable")

        ctx = await BaseAgent(llm_client=BrokenClient()).run("test")
        self.assertIsNotNone(ctx.error)
        self.assertIn("provider unavailable", ctx.error)
        self.assertIsNone(ctx.final_output)

    async def test_action_output_accepts_null_optional_raw_output(self):
        output = ActionOutput.model_validate_json(
            '{"response":"A Markdown answer","tool_used":null,'
            '"success":true,"raw_output":null}'
        )

        self.assertEqual(output.response, "A Markdown answer")
        self.assertIsNone(output.raw_output)

    async def test_action_parser_preserves_inner_markdown_code_fences(self):
        payload = json.dumps({
            "response": "Example:\n\n```python\nprint('hello')\n```",
            "tool_used": None,
            "success": True,
            "raw_output": None,
        })
        parser = ActionAgent()
        parsed = parser._parse_output(
            payload,
            ActionInput(normalized_query="give Python code", action="execute_query"),
        )

        self.assertIn("```python", parsed.response)
        self.assertIn("print('hello')", parsed.response)

    async def test_malformed_json_is_repaired_once(self):
        class RepairingClient:
            def __init__(self):
                self.prompts = []

            def chat(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return (
                        '{"response":"He may say "mama" or "dada".",'
                        '"tool_used":null,"success":true,"raw_output":null}'
                    )
                return json.dumps({
                    "response": 'He may say "mama" or "dada".',
                    "tool_used": None,
                    "success": True,
                    "raw_output": None,
                })

        client = RepairingClient()
        result = await ActionAgent(llm_client=client).run(
            ActionInput(normalized_query="parenting guide", action="summarize")
        )

        self.assertEqual(result.response, 'He may say "mama" or "dada".')
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("Repair JSON syntax only", client.prompts[1])

    async def test_executor_exception_stored_in_error(self):
        async def exploding_executor(ctx: AgentContext) -> str:
            raise RuntimeError("executor exploded")

        agent = BaseAgent(execution=ExecutionStage(executor=exploding_executor))
        ctx = await agent.run("boom query")
        self.assertIsNotNone(ctx.error)
        self.assertIn("Execution stage failed", ctx.error)
        self.assertIn("executor exploded", ctx.error)
        self.assertEqual(ctx.current_state, AgentState.END)

    async def test_cognitive_exception_stored_in_error(self):
        class BrokenCognitive(CognitiveStage):
            async def execute(self, ctx):
                raise RuntimeError("cognitive broke")

        agent = BaseAgent(cognitive=BrokenCognitive())
        ctx = await agent.run("test")
        self.assertIsNotNone(ctx.error)
        self.assertIn("Cognitive stage failed", ctx.error)
        self.assertEqual(ctx.current_state, AgentState.END)


class TestMCPExecution(unittest.IsolatedAsyncioTestCase):

    async def test_current_time_query_uses_deterministic_time_tool(self):
        registry = SimpleNamespace(
            tool_map={"current_time": object()},
            describe_tools=lambda: [{"name": "current_time"}],
            call_tool=AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(
                    text="Europe/London: 2026-07-19 21:00:00 BST (UTC+01:00)"
                )]
            )),
        )
        stage = ExecutionStage(tool_registry=registry)
        stage._tool_agent.run = AsyncMock()
        ctx = AgentContext(
            user_query=(
                "What is the current time in my location and Tokyo?\n\n"
                "User location timezone (IANA): Europe/London."
            )
        )
        ctx.perception = {"normalized_query": "Current local time and Tokyo"}
        ctx.decision = {"action": "compute"}
        ctx.plan = {"steps": []}

        tool_name, result = await stage._execute_tool(ctx)

        self.assertEqual(tool_name, "current_time")
        self.assertIn("BST", result)
        registry.call_tool.assert_awaited_once_with(
            "current_time",
            {
                "input": {
                    "timezones": ["Europe/London", "Asia/Tokyo"],
                }
            },
        )
        stage._tool_agent.run.assert_not_awaited()
    async def test_selected_mcp_tool_is_called_and_result_reaches_action(self):
        registry = SimpleNamespace(
            tool_map={"add": object()},
            describe_tools=lambda: [{
                "name": "add",
                "server": "math",
                "description": "Add two numbers",
                "input_schema": {"type": "object"},
            }],
            call_tool=AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(text='{"result":7}')]
            )),
        )
        stage = ExecutionStage(tool_registry=registry)
        stage._tool_agent.run = AsyncMock(return_value=ToolSelectionOutput(
            tool_name="add",
            arguments={"input": {"a": 3, "b": 4}},
            rationale="Math is required",
        ))
        stage._action_agent.run = AsyncMock(return_value=ActionOutput(
            response="3 + 4 is 7.",
            tool_used="add",
            success=True,
        ))
        events = []
        ctx = AgentContext(
            user_query="What is 3 + 4?",
            event_handler=lambda event, payload: events.append((event, payload)),
        )
        ctx.perception = {"normalized_query": "What is 3 + 4?"}
        ctx.decision = {"action": "compute", "rationale": "Calculate"}
        ctx.plan = {"steps": [{"step": 1, "description": "Add", "tool": "add"}]}

        await stage._action(ctx)

        registry.call_tool.assert_awaited_once_with(
            "add", {"input": {"a": 3, "b": 4}}
        )
        action_input = stage._action_agent.run.await_args.args[0]
        self.assertEqual(action_input.tool_used, "add")
        self.assertIn('"result":7', action_input.tool_result)
        self.assertEqual(ctx.action_result["tool_used"], "add")
        self.assertEqual(
            [event for event, _ in events],
            ["mcp_tool_call", "mcp_tool_result"],
        )
        self.assertEqual(events[0][1]["arguments"], {"input": {"a": 3, "b": 4}})
        self.assertIn("duration_ms", events[1][1])

    async def test_mcp_failure_emits_error_event(self):
        registry = SimpleNamespace(
            tool_map={"add": object()},
            describe_tools=lambda: [{"name": "add"}],
            call_tool=AsyncMock(side_effect=RuntimeError("tool unavailable")),
        )
        stage = ExecutionStage(tool_registry=registry)
        stage._tool_agent.run = AsyncMock(return_value=ToolSelectionOutput(
            tool_name="add",
            arguments={"input": {"a": 1, "b": 2}},
        ))
        events = []
        ctx = AgentContext(
            user_query="Add numbers",
            event_handler=lambda event, payload: events.append((event, payload)),
        )
        ctx.perception = {"normalized_query": "Add 1 and 2"}
        ctx.decision = {"action": "compute"}
        ctx.plan = {"steps": []}

        with self.assertRaisesRegex(RuntimeError, "tool unavailable"):
            await stage._execute_tool(ctx)

        self.assertEqual(
            [event for event, _ in events],
            ["mcp_tool_call", "mcp_tool_error"],
        )
        self.assertEqual(events[-1][1]["error"], "tool unavailable")
        self.assertIn("duration_ms", events[-1][1])

    async def test_mcp_timeout_emits_error_and_does_not_hang(self):
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(1)

        registry = SimpleNamespace(
            tool_map={"add": object()},
            describe_tools=lambda: [{"name": "add"}],
            call_tool=slow_tool,
        )
        stage = ExecutionStage(
            tool_registry=registry,
            tool_timeout_seconds=0.01,
        )
        stage._tool_agent.run = AsyncMock(return_value=ToolSelectionOutput(
            tool_name="add",
            arguments={"input": {"a": 1, "b": 2}},
        ))
        events = []
        ctx = AgentContext(
            user_query="Add numbers",
            event_handler=lambda event, payload: events.append((event, payload)),
        )
        ctx.perception = {"normalized_query": "Add 1 and 2"}
        ctx.decision = {"action": "compute"}
        ctx.plan = {"steps": []}

        with self.assertRaises(asyncio.TimeoutError):
            await stage._execute_tool(ctx)

        self.assertEqual(
            [event for event, _ in events],
            ["mcp_tool_call", "mcp_tool_error"],
        )
        self.assertIn("duration_ms", events[-1][1])
    async def test_search_web_falls_back_to_duckduckgo_tool(self):
        registry = SimpleNamespace(
            tool_map={"duckduckgo_search_results": object()},
            describe_tools=lambda: [{
                "name": "duckduckgo_search_results",
                "server": "websearch",
                "description": "Search DuckDuckGo",
                "input_schema": {"type": "object"},
            }],
            call_tool=AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(text="Found one result")]
            )),
        )
        stage = ExecutionStage(tool_registry=registry)
        stage._tool_agent.run = AsyncMock(
            return_value=ToolSelectionOutput(tool_name=None)
        )
        ctx = AgentContext(user_query="latest Python release")
        ctx.perception = {"normalized_query": "latest Python release"}
        ctx.decision = {"action": "search_web"}
        ctx.plan = {"steps": []}

        tool_name, result = await stage._execute_tool(ctx)

        self.assertEqual(tool_name, "duckduckgo_search_results")
        self.assertEqual(result, "Found one result")
        registry.call_tool.assert_awaited_once_with(
            "duckduckgo_search_results",
            {"input": {"query": "latest Python release", "max_results": 5}},
        )

class TestPackageImports(unittest.IsolatedAsyncioTestCase):

    async def test_top_level_imports(self):
        from agent_base_lib import BaseAgent, AgentContext, AgentStatus, AgentState, StageResult
        self.assertTrue(callable(BaseAgent))
        self.assertTrue(callable(AgentContext))
        self.assertTrue(callable(StageResult))

    async def test_stage_imports(self):
        from agent_base_lib.stages import CognitiveStage, ExecutionStage, ValidationStage
        self.assertTrue(callable(CognitiveStage))
        self.assertTrue(callable(ExecutionStage))
        self.assertTrue(callable(ValidationStage))

    async def test_core_imports(self):
        from agent_base_lib.core import AgentStatus, AgentState, next_state
        self.assertEqual(
            next_state(AgentState.VALIDATION, AgentStatus.SUCCESS),
            AgentState.OUTPUT,
        )
        self.assertEqual(
            next_state(AgentState.VALIDATION, AgentStatus.NEED_REPLAN),
            AgentState.REFLECTION,
        )
        self.assertEqual(
            next_state(AgentState.VALIDATION, AgentStatus.FAILED),
            AgentState.ERROR,
        )
        self.assertEqual(
            next_state(AgentState.ERROR),
            AgentState.END,
        )


if __name__ == "__main__":
    unittest.main()

