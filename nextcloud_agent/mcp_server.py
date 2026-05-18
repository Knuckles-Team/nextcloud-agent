#!/usr/bin/python
import warnings

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
from agent_utilities.mcp_utilities import create_mcp_server
from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.utilities.logging import get_logger
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from nextcloud_agent.auth import get_client

__version__ = "0.11.0"

logger = get_logger(name="nextcloud-agent")
logger.setLevel(logging.INFO)


def register_files_tools(mcp: FastMCP):
    @mcp.tool(tags={"files"})
    async def nextcloud_files(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_files', 'list_files', 'read_file', 'write_file', 'create_folder', 'delete_item', 'move_item', 'copy_item', 'get_properties'"
        ),
        path: str | None = Field(default=None, description="path"),
        content: str | bytes | None = Field(default=None, description="content"),
        overwrite: bool | None = Field(default=None, description="overwrite"),
        client=Depends(get_client),
    ) -> dict:
        """Manage files operations.

        Actions:
          - 'list_files': Call list_files
          - 'list_files': Call list_files
          - 'read_file': Download a file.
          - 'write_file': Upload a file.
          - 'create_folder': Call create_folder
          - 'delete_item': Call delete_item
          - 'move_item': Call move_item
          - 'copy_item': Call copy_item
          - 'get_properties': Call get_properties
        """
        kwargs: dict[str, Any]
        if action == "list_files":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_files(**kwargs)
        if action == "list_files":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_files(**kwargs)
        if action == "read_file":
            kwargs = {"path": path}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.read_file(**kwargs)
        if action == "write_file":
            kwargs = {
                "path": path,
                "content": content,
                "overwrite": overwrite,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.write_file(**kwargs)
        if action == "create_folder":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_folder(**kwargs)
        if action == "delete_item":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.delete_item(**kwargs)
        if action == "move_item":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.move_item(**kwargs)
        if action == "copy_item":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.copy_item(**kwargs)
        if action == "get_properties":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.get_properties(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: list_files', 'list_files', 'read_file', 'write_file', 'create_folder', 'delete_item', 'move_item', 'copy_item', 'get_properties"
        )


def register_user_tools(mcp: FastMCP):
    @mcp.tool(tags={"user"})
    async def nextcloud_user(
        action: str = Field(
            description="Action to perform. Must be one of: 'get_user_info'"
        ),
        client=Depends(get_client),
    ) -> dict:
        """Manage user operations.

        Actions:
          - 'get_user_info': Get current user info.
        """
        kwargs: dict[str, Any]
        if action == "get_user_info":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.get_user_info(**kwargs)
        raise ValueError(f"Unknown action: {action}. Must be one of: get_user_info")


def register_sharing_tools(mcp: FastMCP):
    @mcp.tool(tags={"sharing"})
    async def nextcloud_sharing(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_shares', 'create_share', 'delete_share'"
        ),
        path: str | None = Field(default=None, description="path"),
        share_type: int | None = Field(default=None, description="share type"),
        permissions: int | None = Field(default=None, description="permissions"),
        share_id: str | None = Field(default=None, description="share id"),
        client=Depends(get_client),
    ) -> dict:
        """Manage sharing operations.

        Actions:
          - 'list_shares': List all shares.
          - 'create_share': Create a share.
          - 'delete_share': Delete a share.
        """
        kwargs: dict[str, Any]
        if action == "list_shares":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_shares(**kwargs)
        if action == "create_share":
            kwargs = {
                "path": path,
                "share_type": share_type,
                "permissions": permissions,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_share(**kwargs)
        if action == "delete_share":
            kwargs = {"share_id": share_id}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.delete_share(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: list_shares', 'create_share', 'delete_share"
        )


def register_calendar_tools(mcp: FastMCP):
    @mcp.tool(tags={"calendar"})
    async def nextcloud_calendar(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_calendars', 'list_calendar_events', 'create_calendar_event'"
        ),
        client=Depends(get_client),
    ) -> dict:
        """Manage calendar operations.

        Actions:
          - 'list_calendars': List available calendars.
          - 'list_calendar_events': Call list_calendar_events
          - 'create_calendar_event': Call create_calendar_event
        """
        kwargs: dict[str, Any]
        if action == "list_calendars":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_calendars(**kwargs)
        if action == "list_calendar_events":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_calendar_events(**kwargs)
        if action == "create_calendar_event":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_calendar_event(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: list_calendars', 'list_calendar_events', 'create_calendar_event"
        )


def register_contacts_tools(mcp: FastMCP):
    @mcp.tool(tags={"contacts"})
    async def nextcloud_contacts(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_address_books', 'list_contacts', 'create_contact'"
        ),
        address_book_url: str | None = Field(
            default=None, description="address book url"
        ),
        vcard_data: str | None = Field(default=None, description="vcard data"),
        filename: str | None = Field(default=None, description="filename"),
        client=Depends(get_client),
    ) -> dict:
        """Manage contacts operations.

        Actions:
          - 'list_address_books': List address books.
          - 'list_contacts': List contacts in address book.
          - 'create_contact': Call create_contact
        """
        kwargs: dict[str, Any]
        if action == "list_address_books":
            kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_address_books(**kwargs)
        if action == "list_contacts":
            kwargs = {"address_book_url": address_book_url}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_contacts(**kwargs)
        if action == "create_contact":
            kwargs = {
                "address_book_url": address_book_url,
                "vcard_data": vcard_data,
                "filename": filename,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_contact(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: list_address_books', 'list_contacts', 'create_contact"
        )


def get_mcp_instance() -> tuple[Any, ...]:
    """Initialize and return the MCP instance."""
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
