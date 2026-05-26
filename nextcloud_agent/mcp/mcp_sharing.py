"""MCP tools for sharing operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from nextcloud_agent.auth import get_client


def register_sharing_tools(mcp: FastMCP):
    """
    Register sharing tool category.

    CONCEPT:ECO-4.0
    """

    @mcp.tool(tags={"sharing"})
    async def nextcloud_sharing(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_shares', 'create_share', 'delete_share'"
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
        Manage nextcloud sharing operations.

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

        if action == "list_shares":
            return client.list_shares(**kwargs)
        if action == "create_share":
            return client.create_share(**kwargs)
        if action == "delete_share":
            return client.delete_share(**kwargs)
        raise ValueError(f"Unknown action: {action}")
