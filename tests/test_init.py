import sys
from unittest.mock import patch

import pytest
from fastmcp import FastMCP

from agent_connector_sdk.connector import build_server


@pytest.mark.concept("OS-5.4")
def test_mcp_instance_creation():
    """Test that the MCP instance can be created successfully."""
    with patch.object(sys, "argv", [""]):
        _, mcp = build_server("nextcloud_agent", command_args=[])
    assert isinstance(mcp, FastMCP)


@pytest.mark.concept("OS-5.4")
def test_import_agent():
    """Test that the package can be imported."""
    import nextcloud_agent

    assert nextcloud_agent.__file__ is not None
