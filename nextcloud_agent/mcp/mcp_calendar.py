"""MCP tools for calendar operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from nextcloud_agent.auth import get_client


def register_calendar_tools(mcp: FastMCP):
    """
    Register calendar tool category.

    CONCEPT:ECO-4.0
    """

    @mcp.tool(tags={"calendar"})
    async def nextcloud_calendar(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_calendars', 'list_calendar_events', 'create_calendar_event'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """
        Manage nextcloud calendar operations.

        CONCEPT:ECO-4.0
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if action == "list_calendars":
            return await run_blocking(client.list_calendars, **kwargs)
        if action == "list_calendar_events":
            return await run_blocking(client.list_calendar_events, **kwargs)
        if action == "create_calendar_event":
            return await run_blocking(client.create_calendar_event, **kwargs)
        raise ValueError(f"Unknown action: {action}")
