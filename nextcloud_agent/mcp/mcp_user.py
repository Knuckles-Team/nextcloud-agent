"""MCP tools for user operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from nextcloud_agent.auth import get_client


def register_user_tools(mcp: FastMCP):
    """
    Register user tool category.

    CONCEPT:ECO-4.0
    """

    @mcp.tool(tags={"user"})
    async def nextcloud_user(
        action: str = Field(
            description="Action to perform. Must be one of: 'get_user_info'"
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
        Manage nextcloud user operations.

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

        if action == "get_user_info":
            return await run_blocking(client.get_user_info, **kwargs)
        raise ValueError(f"Unknown action: {action}")
