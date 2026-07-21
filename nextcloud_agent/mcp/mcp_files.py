"""MCP tools for files operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from nextcloud_agent.auth import get_client


def register_files_tools(mcp: FastMCP):
    """
    Register files tool category.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction
    """

    @mcp.tool(tags={"files"})
    async def nextcloud_files(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_files', 'read_file', 'write_file', 'create_folder', 'delete_item', 'move_item', 'copy_item', 'get_properties'"
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
        Manage nextcloud files operations.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception:
            return {"error": "Operation failed"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if action == "list_files":
            return await run_blocking(client.list_files, **kwargs)
        if action == "read_file":
            return await run_blocking(client.read_file, **kwargs)
        if action == "write_file":
            return await run_blocking(client.write_file, **kwargs)
        if action == "create_folder":
            return await run_blocking(client.create_folder, **kwargs)
        if action == "delete_item":
            return await run_blocking(client.delete_item, **kwargs)
        if action == "move_item":
            return await run_blocking(client.move_item, **kwargs)
        if action == "copy_item":
            return await run_blocking(client.copy_item, **kwargs)
        if action == "get_properties":
            return await run_blocking(client.get_properties, **kwargs)
        raise ValueError(f"Unknown action: {action}")
