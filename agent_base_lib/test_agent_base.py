import unittest

from agent_base_lib import BaseAgent, AgentContext, AgentStatus, AgentState, StageResult
from agent_base_lib.stages.execution_stage import ExecutionStage
from agent_base_lib.stages.validation_stage import ValidationStage
from agent_base_lib.stages.cognitive_stage import CognitiveStage


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

