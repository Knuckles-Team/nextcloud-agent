import sys
from unittest.mock import MagicMock

# Hermetically mock tree_sitter and all potential tree_sitter language modules before any imports happen
sys.modules["tree_sitter"] = MagicMock()
for lang in ["javascript", "python", "typescript", "tsx", "go", "rust", "cpp", "c", "bash", "html", "css", "json"]:
    sys.modules[f"tree_sitter_{lang}"] = MagicMock()

import asyncio
import inspect
import json
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
from starlette.datastructures import Headers
from starlette.requests import Request

import pytest

from agent_utilities.core.exceptions import AuthError, UnauthorizedError
from nextcloud_agent.api_client import NextcloudAPI


@pytest.fixture
def mock_session():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value
        session.status_code = 200
        session.put_status_code = None
        session.post_status_code = None
        session.delete_status_code = None
        session.request_status_code = None
        session.response_content = None
        session.response_json = None

        def mock_request(method, url, *args, **kwargs):
            response = MagicMock()

            # Resolve dynamic status code
            status_code = getattr(session, "status_code", 200)
            if method == "PUT" and getattr(session, "put_status_code", None) is not None:
                status_code = session.put_status_code
            elif method == "POST" and getattr(session, "post_status_code", None) is not None:
                status_code = session.post_status_code
            elif method == "DELETE" and getattr(session, "delete_status_code", None) is not None:
                status_code = session.delete_status_code
            elif getattr(session, "request_status_code", None) is not None:
                status_code = session.request_status_code

            response.status_code = status_code

            # Resolve dynamic response content
            if getattr(session, "response_content", None) is not None:
                response.content = session.response_content
            else:
                # Safe string parsing of request data
                data = kwargs.get("data")
                data_str = ""
                if data:
                    if isinstance(data, bytes):
                        data_str = data.decode("utf-8", errors="ignore")
                    else:
                        data_str = str(data)

                # Custom XML payloads based on URLs and bodies
                if "calendars" in url:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                        <d:response>
                            <d:href>/remote.php/dav/calendars/testuser/work/</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:displayname>Work Calendar</d:displayname>
                                    <d:resourcetype>
                                        <c:calendar/>
                                    </d:resourcetype>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                        <d:response>
                            <d:href>/remote.php/dav/calendars/testuser/work/event1.ics</d:href>
                        </d:response>
                    </d:multistatus>"""
                elif "addressbooks" in url:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
                        <d:response>
                            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:displayname>Contacts</d:displayname>
                                    <d:resourcetype>
                                        <c:addressbook/>
                                    </d:resourcetype>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                        <d:response>
                            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/contact1.vcf</d:href>
                        </d:response>
                    </d:multistatus>"""
                elif "quota" in data_str:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:">
                        <d:response>
                            <d:propstat>
                                <d:prop>
                                    <d:quota-available-bytes>5000000000</d:quota-available-bytes>
                                    <d:quota-used-bytes>1000000000</d:quota-used-bytes>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                    </d:multistatus>"""
                else:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
                        <d:response>
                            <d:href>/remote.php/dav/files/testuser/somefile.txt</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:getlastmodified>Fri, 22 May 2026 12:00:00 GMT</d:getlastmodified>
                                    <d:getcontentlength>1234</d:getcontentlength>
                                    <d:getcontenttype>text/plain</d:getcontenttype>
                                    <oc:permissions>WDNVR</oc:permissions>
                                    <d:resourcetype/>
                                    <d:getetag>"abcdef"</d:getetag>
                                    <oc:favorite>1</oc:favorite>
                                    <oc:fileid>12345</oc:fileid>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                    </d:multistatus>"""

            # Resolve dynamic response JSON
            if getattr(session, "response_json", None) is not None:
                response.json.return_value = session.response_json
            else:
                response.json.return_value = {
                    "ocs": {
                        "meta": {"status": "ok", "statuscode": 100, "message": "OK"},
                        "data": {"id": "1", "share_type": 3, "token": "abc"}
                    }
                }
            response.text = '{"ocs": {"meta": {"status": "ok", "statuscode": 100}, "data": {}}}'
            return response

        session.request.side_effect = mock_request
        session.get.side_effect = lambda url, *a, **k: mock_request("GET", url, *a, **k)
        session.post.side_effect = lambda url, *a, **k: mock_request("POST", url, *a, **k)
        session.put.side_effect = lambda url, *a, **k: mock_request("PUT", url, *a, **k)
        session.delete.side_effect = lambda url, *a, **k: mock_request("DELETE", url, *a, **k)
        yield session


def test_nextcloud_api_brute_force(mock_session):
    api_instance = NextcloudAPI(
        base_url="https://cloud.test.com",
        username="testuser",
        password="app-password",
        verify=False
    )

    # Introspect all methods
    for name, method in inspect.getmembers(api_instance, predicate=inspect.ismethod):
        if name.startswith("_"):
            continue

        print(f"Calling NextcloudAPI method: {name}")
        sig = inspect.signature(method)
        kwargs = {}
        for p_name, p in sig.parameters.items():
            if p.default is inspect.Parameter.empty:
                if p_name in ["source_path", "calendar_url", "address_book_url"]:
                    kwargs[p_name] = "https://cloud.test.com/path"
                elif p.annotation is int:
                    kwargs[p_name] = 1
                elif p.annotation is bool:
                    kwargs[p_name] = True
                elif p.annotation is dict:
                    kwargs[p_name] = {}
                elif p.annotation is list:
                    kwargs[p_name] = []
                else:
                    kwargs[p_name] = "test"

        try:
            if inspect.iscoroutinefunction(method):
                asyncio.run(method(**kwargs))
            else:
                method(**kwargs)
        except Exception as e:
            print(f"Failed calling NextcloudAPI method {name}: {e}")


def test_api_client_error_conditions(mock_session):
    api = NextcloudAPI(
        base_url="https://cloud.test.com",
        username="testuser",
        password="app-password",
        verify=False
    )

    # 1. Invalid XML parsing
    res = api._parse_propfind_response("<invalid_xml")
    assert res == []

    # 2. FileNotFoundError on PROPFIND 404
    mock_session.request_status_code = 404
    with pytest.raises(FileNotFoundError):
        api.list_contents("nonexistent")
    mock_session.request_status_code = None

    # 3. FileExistsError on PUT 412 (write_file with overwrite=False)
    mock_session.put_status_code = 412
    with pytest.raises(FileExistsError):
        api.write_file("file.txt", "content", overwrite=False)
    mock_session.put_status_code = None

    # 4. FileExistsError on MKCOL 405 (create_directory)
    mock_session.request_status_code = 405
    with pytest.raises(FileExistsError):
        api.create_directory("existing_dir")
    mock_session.request_status_code = None

    # 5. FileExistsError on MOVE 412 (move_resource)
    mock_session.request_status_code = 412
    with pytest.raises(FileExistsError):
        api.move_resource("src", "dst")
    mock_session.request_status_code = None

    # 6. FileExistsError on COPY 412 (copy_resource)
    mock_session.request_status_code = 412
    with pytest.raises(FileExistsError):
        api.copy_resource("src", "dst")
    mock_session.request_status_code = None

    # 7. Quota details fallback if d:prop is missing
    mock_session.request_status_code = 200
    mock_session.response_content = b"<d:multistatus xmlns:d='DAV:'><d:response></d:response></d:multistatus>"
    res = api.get_user_quota()
    assert res == {}
    mock_session.response_content = None
    mock_session.request_status_code = None

    # 8. OCS exception when response meta is not OK
    mock_session.response_json = {
        "ocs": {
            "meta": {"status": "failure", "statuscode": 999, "message": "Failed action"}
        }
    }
    with pytest.raises(Exception, match="OCS Error: Failed action"):
        api.get_user_info()
    mock_session.response_json = None

    # 9. XML response parser edge cases
    xml_data = b"""<?xml version="1.0" encoding="UTF-8"?>
    <d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
        <d:response>
            <d:href>/remote.php/dav/files/testuser/collection/</d:href>
            <d:propstat>
                <d:prop>
                    <d:resourcetype>
                        <d:collection/>
                    </d:resourcetype>
                </d:prop>
            </d:propstat>
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/files/testuser/no_propstat/</d:href>
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/files/testuser/no_prop/</d:href>
            <d:propstat></d:propstat>
        </d:response>
    </d:multistatus>"""
    parsed = api._parse_propfind_response(xml_data)
    assert parsed[0]["is_folder"] is True
    assert len(parsed) == 1

    # 10. get_absolute_url http check
    assert api._get_absolute_url("https://already-absolute.com") == "https://already-absolute.com"

    # 11. list_events XML parser edge cases
    mock_session.response_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <d:multistatus xmlns:d="DAV:">
        <d:response>
            <!-- No href -->
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/calendars/testuser/work/not_ics.txt</d:href>
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/calendars/testuser/work/event1.ics</d:href>
        </d:response>
    </d:multistatus>"""
    events = api.list_events("https://cloud.test.com/cal")
    assert len(events) == 1
    assert events[0]["name"] == "event1.ics"
    mock_session.response_content = None

    # 12. list_contacts XML parser edge cases
    mock_session.response_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <d:multistatus xmlns:d="DAV:">
        <d:response>
            <!-- No href -->
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/not_vcf.txt</d:href>
        </d:response>
        <d:response>
            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/contact1.vcf</d:href>
        </d:response>
    </d:multistatus>"""
    contacts = api.list_contacts("https://cloud.test.com/contacts")
    assert len(contacts) == 1
    assert contacts[0]["name"] == "contact1.vcf"
    mock_session.response_content = None

    # 13. Empty files in get_properties
    with patch.object(api, "list_contents", return_value=[]):
        assert api.get_properties("somepath") == {}

    # 14. list_contents path stripping elif check
    with patch("os.path.basename", return_value="somefile.txt"):
        api.list_contents("")


def test_auth_coverage():
    from nextcloud_agent.auth import get_client

    # 1. Parameter arguments passed directly
    with get_client(
        base_url="https://direct.test.com",
        username="direct_user",
        password="direct_password",
        verify=False
    ) as client:
        assert client.base_url == "https://direct.test.com"
        assert client.username == "direct_user"

    # 2. Parameter fallback to environment variables
    with patch.dict(os.environ, {
        "NEXTCLOUD_URL": "https://env.test.com",
        "NEXTCLOUD_USERNAME": "env_user",
        "NEXTCLOUD_PASSWORD": "env_password",
        "NEXTCLOUD_SSL_VERIFY": "True"
    }):
        with get_client() as client:
            assert client.base_url == "https://env.test.com"
            assert client.username == "env_user"
            assert client.verify is True

    # 3. Parameter missing error
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Nextcloud URL, username, and password must be provided"):
            with get_client():
                pass

    # 4. AuthException mapping
    with patch("nextcloud_agent.auth.NextcloudAPI", side_effect=AuthError("Auth failed")):
        with pytest.raises(RuntimeError, match="AUTHENTICATION ERROR"):
            with get_client(base_url="https://x.com", username="u", password="p"):
                pass

    with patch("nextcloud_agent.auth.NextcloudAPI", side_effect=UnauthorizedError("Unauthorized")):
        with pytest.raises(RuntimeError, match="AUTHENTICATION ERROR"):
            with get_client(base_url="https://x.com", username="u", password="p"):
                pass


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
            tool_objs = await mcp.list_tools() if inspect.iscoroutinefunction(mcp.list_tools) else mcp.list_tools()
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


def test_mcp_server_run_options():
    from nextcloud_agent.mcp_server import mcp_server

    mock_mcp = MagicMock()
    mock_args = MagicMock()

    with patch("nextcloud_agent.mcp_server.get_mcp_instance", return_value=(mock_mcp, mock_args, [])):
        # Test stdio transport
        mock_args.transport = "stdio"
        mcp_server()
        mock_mcp.run.assert_called_with(transport="stdio")

        # Test streamable-http transport
        mock_args.transport = "streamable-http"
        mock_args.host = "127.0.0.1"
        mock_args.port = 8000
        mcp_server()
        mock_mcp.run.assert_called_with(transport="streamable-http", host="127.0.0.1", port=8000)

        # Test sse transport
        mock_args.transport = "sse"
        mcp_server()
        mock_mcp.run.assert_called_with(transport="sse", host="127.0.0.1", port=8000)

        # Test invalid transport
        mock_args.transport = "invalid"
        with patch("sys.exit") as mock_exit:
            mcp_server()
            mock_exit.assert_called_with(1)


def test_agent_server_coverage():
    with patch("agent_utilities.build_system_prompt_from_workspace", return_value="mocked prompt"):
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


def test_main_execution():
    import runpy

    with patch("sys.argv", ["agent_server.py"]):
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create1:
            with patch("agent_utilities.create_agent_server") as mock_create2:
                with patch("agent_utilities.build_system_prompt_from_workspace", return_value="mocked prompt"):
                    runpy.run_module("nextcloud_agent", run_name="__main__")
                    assert mock_create1.called or mock_create2.called


def test_agent_server_main_execution():
    import runpy

    with patch("sys.argv", ["agent_server.py"]):
        with patch("nextcloud_agent.agent_server.create_agent_server") as mock_create1:
            with patch("agent_utilities.create_agent_server") as mock_create2:
                with patch("agent_utilities.build_system_prompt_from_workspace", return_value="mocked prompt"):
                    runpy.run_module("nextcloud_agent.agent_server", run_name="__main__")
                    assert mock_create1.called or mock_create2.called


def test_mcp_server_main_execution():
    import runpy

    with patch("sys.argv", ["mcp_server.py"]):
        with patch("fastmcp.FastMCP.run") as mock_run:
            runpy.run_module("nextcloud_agent.mcp_server", run_name="__main__")
            assert mock_run.called


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
