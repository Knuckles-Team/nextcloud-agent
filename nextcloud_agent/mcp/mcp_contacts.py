"""MCP tools for contacts operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from nextcloud_agent.auth import get_client


def register_contacts_tools(mcp: FastMCP):
    """
    Register contacts tool category.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction
    """

    @mcp.tool(tags={"contacts"})
    async def nextcloud_contacts(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_address_books', 'list_contacts', 'create_contact'"
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
        Manage nextcloud contacts operations.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if action == "list_address_books":
            return await run_blocking(client.list_address_books, **kwargs)
        if action == "list_contacts":
            return await run_blocking(client.list_contacts, **kwargs)
        if action == "create_contact":
            return await run_blocking(client.create_contact, **kwargs)
        raise ValueError(f"Unknown action: {action}")
