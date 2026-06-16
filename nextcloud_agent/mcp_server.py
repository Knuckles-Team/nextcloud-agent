#!/usr/bin/python
import warnings

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from fastmcp.utilities.logging import get_logger
from pydantic import Field

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import logging
import os
import sys
from typing import Any

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import create_mcp_server, resolve_action
from dotenv import find_dotenv, load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

from nextcloud_agent.auth import get_client

__version__ = "0.33.0"

logger = get_logger(name="nextcloud-agent")
logger.setLevel(logging.INFO)


def register_files_tools(mcp: FastMCP):
    """
    Register files tool category.
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
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(
            action,
            [
                "list_files",
                "read_file",
                "write_file",
                "create_folder",
                "delete_item",
                "move_item",
                "copy_item",
                "get_properties",
            ],
            service="nextcloud-agent",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "list_files":
            return client.list_files(**kwargs)
        if action == "read_file":
            return client.read_file(**kwargs)
        if action == "write_file":
            return client.write_file(**kwargs)
        if action == "create_folder":
            return client.create_folder(**kwargs)
        if action == "delete_item":
            return client.delete_item(**kwargs)
        if action == "move_item":
            return client.move_item(**kwargs)
        if action == "copy_item":
            return client.copy_item(**kwargs)
        if action == "get_properties":
            return client.get_properties(**kwargs)
        raise ValueError(f"Unknown action: {action}")


def register_user_tools(mcp: FastMCP):
    """
    Register user tool category.
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
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(
            action,
            ["get_user_info"],
            service="nextcloud-agent",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "get_user_info":
            return client.get_user_info(**kwargs)
        raise ValueError(f"Unknown action: {action}")


def register_sharing_tools(mcp: FastMCP):
    """
    Register sharing tool category.
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
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(
            action,
            ["list_shares", "create_share", "delete_share"],
            service="nextcloud-agent",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "list_shares":
            return client.list_shares(**kwargs)
        if action == "create_share":
            return client.create_share(**kwargs)
        if action == "delete_share":
            return client.delete_share(**kwargs)
        raise ValueError(f"Unknown action: {action}")


def register_calendar_tools(mcp: FastMCP):
    """
    Register calendar tool category.
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
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(
            action,
            ["list_calendars", "list_calendar_events", "create_calendar_event"],
            service="nextcloud-agent",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "list_calendars":
            return client.list_calendars(**kwargs)
        if action == "list_calendar_events":
            return client.list_calendar_events(**kwargs)
        if action == "create_calendar_event":
            return client.create_calendar_event(**kwargs)
        raise ValueError(f"Unknown action: {action}")


def register_contacts_tools(mcp: FastMCP):
    """
    Register contacts tool category.
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
        """
        if ctx:
            ctx.info("Executing tool...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"error": f"Invalid params_json: {e}"}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(
            action,
            ["list_address_books", "list_contacts", "create_contact"],
            service="nextcloud-agent",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "list_address_books":
            return client.list_address_books(**kwargs)
        if action == "list_contacts":
            return client.list_contacts(**kwargs)
        if action == "create_contact":
            return client.create_contact(**kwargs)
        raise ValueError(f"Unknown action: {action}")


def get_mcp_instance() -> tuple[Any, ...]:
    """
    Initialize and return the MCP instance.
    """
    load_dotenv(find_dotenv())
    args, mcp, middlewares = create_mcp_server(
        name="nextcloud-agent MCP",
        version=__version__,
        instructions="nextcloud-agent MCP Server — Condensed Action-Routed Tools.",
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

    DEFAULT_FILESTOOL = to_boolean(os.getenv("FILESTOOL", "True"))
    if DEFAULT_FILESTOOL:
        register_files_tools(mcp)
    DEFAULT_USERTOOL = to_boolean(os.getenv("USERTOOL", "True"))
    if DEFAULT_USERTOOL:
        register_user_tools(mcp)
    DEFAULT_SHARINGTOOL = to_boolean(os.getenv("SHARINGTOOL", "True"))
    if DEFAULT_SHARINGTOOL:
        register_sharing_tools(mcp)
    DEFAULT_CALENDARTOOL = to_boolean(os.getenv("CALENDARTOOL", "True"))
    if DEFAULT_CALENDARTOOL:
        register_calendar_tools(mcp)
    DEFAULT_CONTACTSTOOL = to_boolean(os.getenv("CONTACTSTOOL", "True"))
    if DEFAULT_CONTACTSTOOL:
        register_contacts_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)
    return mcp, args, middlewares


def mcp_server() -> None:
    """
    Launch the MCP server process.
    """
    mcp, args, middlewares = get_mcp_instance()
    print(f"nextcloud-agent MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
