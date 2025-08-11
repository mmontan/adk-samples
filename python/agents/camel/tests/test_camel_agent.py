import asyncio
import unittest
from unittest.mock import MagicMock, patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.base_llm import BaseLlm

from camel.camel_agent.camel_agent import (
    CaMeLAgent,
    CaMeLInterpreter,
    CaMelInterpreterService,
    QuarantinedLlmService,
)
from camel.camel_library.interpreter import interpreter
from camel.camel_library.security_policy import SecurityPolicyEngine


class TestCaMeLAgent(unittest.TestCase):
    def test_camel_agent_initialization(self):
        agent = CaMeLAgent(
            name="TestCaMeLAgent",
            model="gemini-2.5-pro",
        )
        self.assertIsInstance(agent.pllm_agent, LlmAgent)
        self.assertIsInstance(agent.camel_interpreter_agent, CaMeLInterpreter)
        self.assertIsInstance(agent.loop_agent, LoopAgent)


class TestCaMeLInterpreterService(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=BaseLlm)
        self.mock_security_policy_engine = MagicMock(spec=SecurityPolicyEngine)
        self.mock_qllm_service = MagicMock(spec=QuarantinedLlmService)
        with patch(
            "camel.camel_agent.camel_agent.QuarantinedLlmService",
            return_value=self.mock_qllm_service,
        ):
            self.interpreter_service = CaMelInterpreterService(
                model=self.mock_model,
                tools=[],
                eval_args=interpreter.EvalArgs(
                    eval_mode=interpreter.DependenciesPropagationMode.NORMAL,
                    security_policy_engine=self.mock_security_policy_engine,
                ),
            )

    def test_interpreter_thread_is_started(self):
        self.assertTrue(self.interpreter_service.interpreter_thread.is_alive())

    def test_execute_code_uses_queues(self):
        code = "print('hello')"
        tool_calls_chain = []
        dependencies = ()
        expected_result = (
            interpreter.result.Ok(
                interpreter.camel_value.CaMeLStr.from_raw("hello", (), ())
            ),
            {},
            [],
            (),
        )
        self.interpreter_service.response_queue.put(expected_result)

        self.interpreter_service.execute_code(code, tool_calls_chain, dependencies)

        request = self.interpreter_service.request_queue.get_nowait()
        self.assertEqual(request, (code, tool_calls_chain, dependencies))

    def tearDown(self):
        # Signal the worker to stop and wait for it to finish
        self.interpreter_service.request_queue.put((None, None, None))
        self.interpreter_service.interpreter_thread.join(timeout=1)


class TestCaMeLInterpreter(unittest.TestCase):
    def setUp(self):
        self.mock_interpreter_service = MagicMock(spec=CaMelInterpreterService)
        self.interpreter = CaMeLInterpreter(
            name="TestCaMeLInterpreter",
            camel_interpreter_service=self.mock_interpreter_service,
        )
        self.mock_context = MagicMock(spec=InvocationContext)

    @patch("asyncio.to_thread")
    async def test_run_async_impl_with_code(self, mock_to_thread):
        self.mock_context.session.state = {
            "p_llm_code": "print('hello')",
            "function_calls": [],
            "dependencies": (),
        }
        mock_to_thread.return_value = ("hello", [], None, {}, ())

        events = [
            event
            async for event in self.interpreter._run_async_impl(self.mock_context)
        ]

        mock_to_thread.assert_called_once_with(
            self.mock_interpreter_service.execute_code,
            "print('hello')",
            [],
            (),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content.parts[0].text, "hello")

    async def test_run_async_impl_no_code(self):
        self.mock_context.session.state = {}

        events = [
            event
            async for event in self.interpreter._run_async_impl(self.mock_context)
        ]

        self.assertEqual(len(events), 1)
        self.assertIn(
            "The Privileged LLM did not generate any code",
            events[0].content.parts[0].text,
        )


if __name__ == "__main__":
    unittest.main()
