# tests/test_bug_cx_046_ctx_await.py
"""Regression test for BUG-CX-046.

`ctx.info(...)` is an async coroutine method on fastmcp's `Context`. Calling
it without `await` creates a coroutine object that is discarded, the log
line never emits, and Python raises `RuntimeWarning: coroutine
'Context.info' was never awaited`. This test proves the tool under test
actually awaits `ctx.info(...)`.

Reuses this repo's own `get_mcp_instance()` + `mcp.list_tools()` + `tool.fn`
helper pattern from tests/test_mcp_server.py.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_nextcloud_files_awaits_ctx_info(mock_session):
    from nextcloud_agent.mcp_server import get_mcp_instance

    with patch("nextcloud_agent.auth.get_client") as mock_auth_client:
        mock_api = mock_auth_client.return_value.__enter__.return_value

        mcp_data = get_mcp_instance()
        mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data

        tool_objs = (
            await mcp.list_tools()
            if inspect.iscoroutinefunction(mcp.list_tools)
            else mcp.list_tools()
        )
        tools_by_name = {tool.name: tool for tool in tool_objs}

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        nextcloud_files = tools_by_name["nextcloud_files"]
        await nextcloud_files.fn(
            action="list_files",
            params_json="{}",
            client=mock_api,
            ctx=mock_ctx,
        )

        mock_ctx.info.assert_awaited_once()
