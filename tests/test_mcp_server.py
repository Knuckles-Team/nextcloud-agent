import sys
from unittest.mock import patch

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
