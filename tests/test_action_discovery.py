"""Action-discovery behavior for the standardized action-routed tools."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

TOOL_NAMES = [
    "nextcloud_files",
    "nextcloud_user",
    "nextcloud_sharing",
    "nextcloud_calendar",
    "nextcloud_contacts",
]


def _get_tools():
    with patch("nextcloud_agent.auth.get_client"):
        from nextcloud_agent.mcp_server import get_mcp_instance

        mcp = get_mcp_instance()[0]

        async def _list():
            return await mcp.list_tools()

        tools = asyncio.new_event_loop().run_until_complete(_list())
    return {t.name: t for t in tools}


@pytest.mark.concept("ECO-4.0")
def test_list_actions_returns_names():
    tools = _get_tools()
    for name in TOOL_NAMES:
        tool = tools[name]
        result = asyncio.new_event_loop().run_until_complete(
            tool.fn(
                action="list_actions", params_json="{}", client=MagicMock(), ctx=None
            )
        )
        assert isinstance(result, dict)
        assert result["service"] == "nextcloud-agent"
        assert result["actions"], f"{name} returned no actions"


@pytest.mark.concept("ECO-4.0")
def test_bogus_action_raises_with_list_actions_hint():
    tools = _get_tools()
    for name in TOOL_NAMES:
        tool = tools[name]
        with pytest.raises(ValueError) as exc:
            asyncio.new_event_loop().run_until_complete(
                tool.fn(
                    action="totally_bogus_xyz",
                    params_json="{}",
                    client=MagicMock(),
                    ctx=None,
                )
            )
        assert "list_actions" in str(exc.value)
