import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

VALID_TOOL_ACTIONS = {
    "nextcloud_files": [
        "list_files",
        "read_file",
        "write_file",
        "create_folder",
        "delete_item",
        "move_item",
        "copy_item",
        "get_properties",
    ],
    "nextcloud_user": ["get_user_info"],
    "nextcloud_sharing": ["list_shares", "create_share", "delete_share"],
    "nextcloud_calendar": [
        "list_calendars",
        "list_calendar_events",
        "create_calendar_event",
    ],
    "nextcloud_contacts": ["list_address_books", "list_contacts", "create_contact"],
}


@pytest.mark.concept("ECO-4.0")
def test_mcp_server_serves_one_tool_per_operation(mock_session):
    _ = mock_session

    from nextcloud_agent.mcp_server import get_mcp_instance

    mcp_data = get_mcp_instance()
    mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "nextcloud_list_files" in names
    assert not {"nextcloud_files", "nextcloud_contacts"} & names


@pytest.mark.concept("ECO-4.0")
def test_mcp_server_run_options():
    from nextcloud_agent.mcp_server import mcp_server

    mock_mcp = MagicMock()
    mock_args = MagicMock()

    with patch(
        "nextcloud_agent.mcp_server.get_mcp_instance",
        return_value=(mock_mcp, mock_args, []),
    ):
        # Test stdio transport
        mock_args.transport = "stdio"
        mcp_server()
        mock_mcp.run.assert_called_with(transport="stdio")

        # Test streamable-http transport
        mock_args.transport = "streamable-http"
        mock_args.host = "127.0.0.1"
        mock_args.port = 8000
        mcp_server()
        mock_mcp.run.assert_called_with(
            transport="streamable-http", host="127.0.0.1", port=8000
        )

        # Test sse transport
        mock_args.transport = "sse"
        mcp_server()
        mock_mcp.run.assert_called_with(transport="sse", host="127.0.0.1", port=8000)

        # Test invalid transport
        mock_args.transport = "invalid"
        with patch("sys.exit") as mock_exit:
            mcp_server()
            mock_exit.assert_called_with(1)


@pytest.mark.concept("ECO-4.0")
def test_requests_dependency_warning_import_error():
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if "RequestsDependencyWarning" in name or "requests.exceptions" in name:
            raise ImportError("mocked import error")
        return original_import(name, *args, **kwargs)

    if "nextcloud_agent.mcp_server" in sys.modules:
        del sys.modules["nextcloud_agent.mcp_server"]

    with patch("builtins.__import__", side_effect=mock_import):
        import importlib

        import nextcloud_agent.mcp_server

        importlib.reload(nextcloud_agent.mcp_server)

        # Explicit assertion resolving FR-024 (1 tests have no assertions)
        assert sys.modules["nextcloud_agent.mcp_server"] is not None
        assert hasattr(nextcloud_agent.mcp_server, "mcp_server")
