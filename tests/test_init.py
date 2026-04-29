import sys
from unittest.mock import patch

from fastmcp import FastMCP

from nextcloud_agent.mcp_server import get_mcp_instance


def test_mcp_instance_creation():
    """Test that the MCP instance can be created successfully."""
    with patch.object(sys, "argv", [""]):
        mcp, args, middlewares, registered_tags = get_mcp_instance()
    assert isinstance(mcp, FastMCP)


def test_import_agent():
    """Test that the package can be imported."""
    pass
