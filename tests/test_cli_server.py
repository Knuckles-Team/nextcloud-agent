import pytest
import runpy
import sys
from unittest.mock import patch


@pytest.mark.concept("ORCH-1.5")
def test_agent_server_coverage():
    with patch(
        "agent_utilities.build_system_prompt_from_workspace",
        return_value="mocked prompt",
    ):
        from nextcloud_agent.agent_server import agent_server

        # Standard execution test
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create:
            with patch("sys.argv", ["agent_server.py"]):
                agent_server()
                mock_create.assert_called_once()

        # Debug mode execution test
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create:
            with patch("sys.argv", ["agent_server.py", "--debug"]):
                agent_server()
                mock_create.assert_called_once()


@pytest.mark.concept("ORCH-1.5")
def test_main_execution():
    with patch("sys.argv", ["agent_server.py"]):
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create1:
            with patch("agent_utilities.create_agent_server") as mock_create2:
                with patch(
                    "agent_utilities.build_system_prompt_from_workspace",
                    return_value="mocked prompt",
                ):
                    runpy.run_module("nextcloud_agent", run_name="__main__")
                    assert mock_create1.called or mock_create2.called


@pytest.mark.concept("ORCH-1.5")
def test_agent_server_main_execution():
    with patch("sys.argv", ["agent_server.py"]):
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create1:
            with patch("agent_utilities.create_agent_server") as mock_create2:
                with patch(
                    "agent_utilities.build_system_prompt_from_workspace",
                    return_value="mocked prompt",
                ):
                    runpy.run_module(
                        "nextcloud_agent.agent_server", run_name="__main__"
                    )
                    assert mock_create1.called or mock_create2.called


@pytest.mark.concept("ECO-4.0")
def test_mcp_server_main_execution():
    with patch("sys.argv", ["mcp_server.py"]):
        with patch("fastmcp.FastMCP.run") as mock_run:
            runpy.run_module("nextcloud_agent.mcp_server", run_name="__main__")
            assert mock_run.called
