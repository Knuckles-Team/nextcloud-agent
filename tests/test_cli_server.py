import runpy
from unittest.mock import patch

import pytest








@pytest.mark.concept("ECO-4.0")
def test_mcp_server_main_execution():
    with patch("sys.argv", ["mcp_server.py"]):
        with patch("fastmcp.FastMCP.run") as mock_run:
            runpy.run_module("nextcloud_agent.mcp_server", run_name="__main__")
            assert mock_run.called
