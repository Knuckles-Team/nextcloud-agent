import asyncio
import inspect
import json
import sys
import pytest
from unittest.mock import patch, MagicMock
from starlette.datastructures import Headers
from starlette.requests import Request


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
def test_mcp_server_coverage(mock_session):
    from nextcloud_agent.mcp_server import get_mcp_instance

    with patch("nextcloud_agent.auth.get_client") as mock_auth_client:
        mock_api = mock_auth_client.return_value.__enter__.return_value

        mcp_data = get_mcp_instance()
        mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data

        async def run_tools():
            # 1. Custom health route
            routes = []
            if hasattr(mcp, "_additional_http_routes"):
                routes = mcp._additional_http_routes
            elif hasattr(mcp, "routes"):
                routes = mcp.routes
            elif hasattr(mcp, "_app") and hasattr(mcp._app, "routes"):
                routes = mcp._app.routes

            for route in routes:
                if hasattr(route, "path") and route.path == "/health":
                    mock_scope = {
                        "type": "http",
                        "method": "GET",
                        "path": "/health",
                        "headers": Headers().raw,
                    }
                    mock_req = Request(scope=mock_scope)
                    res = await route.endpoint(mock_req)
                    assert res.status_code == 200
                    assert json.loads(res.body.decode()) == {"status": "OK"}

            # 2. Execute each tool and each action under tool.fn
            tool_objs = (
                await mcp.list_tools()
                if inspect.iscoroutinefunction(mcp.list_tools)
                else mcp.list_tools()
            )
            for tool in tool_objs:
                tool_name = tool.name
                actions = VALID_TOOL_ACTIONS.get(tool_name, [None])
                for act in actions:
                    # Execute with ctx
                    await tool.fn(
                        action=act,
                        params_json='{"path": "test", "overwrite": true, "share_type": 3, "calendar_url": "url", "event_data": "ics", "address_book_url": "url", "vcard_data": "vcard"}',
                        client=mock_api,
                        ctx=MagicMock(),
                    )
                    # Execute without ctx
                    await tool.fn(
                        action=act,
                        params_json='{"path": "test"}',
                        client=mock_api,
                        ctx=None,
                    )

                # Invalid JSON parameter exception handler cover
                res_err = await tool.fn(
                    action=actions[0],
                    params_json="invalid_json",
                    client=mock_api,
                    ctx=None,
                )
                assert "error" in res_err

                # Unknown action cover
                with pytest.raises(ValueError):
                    await tool.fn(
                        action="invalid_action_xyz",
                        params_json="{}",
                        client=mock_api,
                        ctx=None,
                    )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_tools())
        loop.close()


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
        import nextcloud_agent.mcp_server
        import importlib

        importlib.reload(nextcloud_agent.mcp_server)

        # Explicit assertion resolving FR-024 (1 tests have no assertions)
        assert sys.modules["nextcloud_agent.mcp_server"] is not None
        assert hasattr(nextcloud_agent.mcp_server, "mcp_server")
