import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from google.cloud import bigquery # Assuming this will be installed
from google.cloud.exceptions import BadRequest # Assuming this will be installed

# --- BEGIN CRITICAL SYS.MODULES MOCKING (from previous successful runs) ---
# This is to handle persistent import issues in the test environment for ADK and GenAI modules.
import sys # Ensure sys is imported

# Mock google.genai and google.genai.types
mock_google_genai_types = MagicMock()
mock_google_genai = MagicMock()
mock_google_genai.types = mock_google_genai_types
sys.modules['google.genai'] = mock_google_genai
sys.modules['google.genai.types'] = mock_google_genai_types

# Mock google.adk.agents, google.adk.agents.Agent, and google.adk.agents.callback_context.CallbackContext
mock_adk_agent_class = MagicMock(name="AgentMock")
mock_callback_context_class = MagicMock(name="CallbackContextMock")
mock_adk_agents_module = MagicMock(name="AgentsModuleMock")
mock_adk_agents_module.Agent = mock_adk_agent_class
mock_adk_agents_module.callback_context = MagicMock(name="CallbackContextSubModuleMock")
mock_adk_agents_module.callback_context.CallbackContext = mock_callback_context_class
sys.modules['google.adk.agents'] = mock_adk_agents_module
sys.modules['google.adk.agents.callback_context'] = mock_adk_agents_module.callback_context

# Mock google.adk.tools, google.adk.tools.ToolContext, and google.adk.tools.agent_tool.AgentTool
mock_tool_context_class_for_mocking = MagicMock(name="ToolContextMockForSysModules") # Renamed to avoid conflict
mock_agent_tool_class = MagicMock(name="AgentToolMock")
mock_adk_tools_module = MagicMock(name="ToolsModuleMock") # Corrected typo MagicMocK to MagicMock
mock_adk_tools_module.ToolContext = mock_tool_context_class_for_mocking
mock_adk_tools_module.agent_tool = MagicMock(name="AgentToolSubModuleMock")
mock_adk_tools_module.agent_tool.AgentTool = mock_agent_tool_class
sys.modules['google.adk.tools'] = mock_adk_tools_module
sys.modules['google.adk.tools.agent_tool'] = mock_adk_tools_module.agent_tool

# Mock google.adk.code_executors.VertexAiCodeExecutor
mock_vertex_ai_code_executor_class = MagicMock(name="VertexAiCodeExecutorMock")
mock_adk_code_executors_module = MagicMock(name="CodeExecutorsModuleMock")
mock_adk_code_executors_module.VertexAiCodeExecutor = mock_vertex_ai_code_executor_class
if 'google.adk' not in sys.modules:
    sys.modules['google.adk'] = MagicMock(name="GoogleAdkMockModule")
elif not isinstance(sys.modules['google.adk'], MagicMock):
    sys.modules['google.adk'] = MagicMock(name="GoogleAdkMockModuleReplaced")
sys.modules['google.adk'].code_executors = mock_adk_code_executors_module
sys.modules['google.adk.code_executors'] = mock_adk_code_executors_module
# --- END CRITICAL SYS.MODULES MOCKING ---

# Actual ToolContext class to be used by tests (can be a simple mock or the one from ADK if available)
# For consistency with the problem description, we'll define a simple one if ADK isn't truly there.
try:
    from google.adk.tools import ToolContext as ActualToolContext
except ImportError:
    class MockToolContextForSetup: # Renamed to avoid conflict
        def __init__(self):
            self.state = {}
    ActualToolContext = MockToolContextForSetup

ToolContext = ActualToolContext # This will be used by the setUp method

# Append the project root to sys.path to allow imports from the data_science package
import os
# Calculate the path to the 'agents' directory, which is three levels up
# from 'agents/data-science/tests/sub_agents/bigquery'
agents_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, agents_root_dir)

# Now that /app/agents is in sys.path, and problematic modules are mocked,
# we can import data_science directly
from data_science.sub_agents.bigquery.tools import run_bigquery_validation, get_bq_client


class TestRunBigQueryValidation(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Use the ToolContext that's either imported or mocked at the top level
        self.mock_tool_context = MagicMock(spec=ToolContext)
        self.allowed_project = "allowed_project"
        self.allowed_dataset = "allowed_dataset"
        self.mock_tool_context.state = {
            "database_settings": {
                "bq_project_id": self.allowed_project,
                "bq_dataset_id": self.allowed_dataset,
            }
        }

    @patch("data_science.sub_agents.bigquery.tools.get_bq_client")
    def test_query_accessing_correct_project_wrong_dataset(self, mock_get_bq_client):
        """Test query with correct project ID but disallowed dataset ID."""
        mock_bq_client = MagicMock(spec=bigquery.Client)
        mock_get_bq_client.return_value = mock_bq_client

        # Dry run mock setup
        mock_dry_run_job = MagicMock(spec=bigquery.QueryJob)
        mock_table_ref = MagicMock(spec=bigquery.table.TableReference)
        wrong_dataset = "wrong_dataset"
        mock_table_ref.project = self.allowed_project # Correct project
        mock_table_ref.dataset_id = wrong_dataset    # Incorrect dataset
        mock_table_ref.table_id = "some_table"
        type(mock_dry_run_job).referenced_tables = PropertyMock(return_value=[mock_table_ref])

        # Configure client.query to return dry run job for dry run config
        def query_side_effect(sql, job_config=None, **kwargs):
            if job_config and job_config.dry_run:
                return mock_dry_run_job
            # This path should not be hit if dry run correctly identifies unauthorized access
            raise AssertionError("Actual query execution should not be called for disallowed dataset access.")
        mock_bq_client.query.side_effect = query_side_effect

        sql_query = f"SELECT * FROM `{self.allowed_project}.{wrong_dataset}.some_table`"
        result = run_bigquery_validation(sql_query, self.mock_tool_context)

        expected_error_msg = (
            f"Invalid SQL: Query attempts to access unauthorized dataset '{self.allowed_project}.{wrong_dataset}'. "
            f"Only access to '{self.allowed_project}.{self.allowed_dataset}' is permitted."
        )
        self.assertEqual(result.get("error_message"), expected_error_msg)
        self.assertIsNone(result.get("query_result"))
        # Ensure only dry run query was called
        mock_bq_client.query.assert_called_once()
        args, call_kwargs = mock_bq_client.query.call_args # Use call_kwargs to avoid conflict
        self.assertTrue(call_kwargs['job_config'].dry_run)


if __name__ == "__main__":
    unittest.main()
