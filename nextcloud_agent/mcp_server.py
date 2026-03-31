#!/usr/bin/python


from dotenv import load_dotenv, find_dotenv
from agent_utilities.base_utilities import to_boolean
import os
import sys
import logging
import json
import uuid
import dateutil.parser
from typing import Any, Dict
from pydantic import Field
from fastmcp import FastMCP, Context
from fastmcp.utilities.logging import get_logger
from icalendar import Calendar, Event
from agent_utilities.mcp_utilities import (
    create_mcp_server,
)
from nextcloud_agent.auth import get_client

__version__ = "0.2.51"
print(f"Nextcloud MCP v{__version__}")

logger = get_logger(name="TokenMiddleware")
logger.setLevel(logging.DEBUG)


def register_prompts(mcp: FastMCP):
    @mcp.prompt(name="list_files", description="List files in a directory.")
    @mcp.tool(tags={"files"})
    def list_files(path: str = "/") -> str:
        """List files."""
        return f"Please list files in '{path}'"

    @mcp.prompt(name="share_file", description="Share a file with someone.")
    def share_file(path: str) -> str:
        """Share file."""
        return f"Please share the file '{path}'"

    @mcp.prompt(name="recent_files", description="Show recently modified files.")
    def recent_files() -> str:
        """Recent files."""
        return "Please show recently modified files."


def register_misc_tools(mcp: FastMCP):
    async def health_check() -> Dict:
        return {"status": "OK"}


def register_files_tools(mcp: FastMCP):
    @mcp.tool(tags={"files"})
    async def list_files(
        path: str = Field(
            default="",
            description="Path to valid directory in Nextcloud (default: root)",
        ),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        List files and directories at a specific path in Nextcloud.
        Returns a formatted string list of contents.
        """
        try:
            with get_client(base_url, username, password) as client:
                files = client.list_contents(path)

                if not files:
                    return "Directory is empty or path info not returned."

                output = [f"Contents of '{path or '/'}':"]
                for f in files:
                    ftype = "[DIR]" if f["is_folder"] else "[FILE]"
                    size = f["content_length"] if not f["is_folder"] else "-"
                    output.append(
                        f"{ftype} {f['name']} (Size: {size}, Modified: {f['last_modified']})"
                    )

                return "\n".join(output)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    @mcp.tool(tags={"files"})
    async def read_file(
        path: str = Field(..., description="Path to the file to read"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        Read the contents of a text file from Nextcloud.
        """
        try:
            with get_client(base_url, username, password) as client:
                content = client.read_file(path)
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return f"<Binary content: {len(content)} bytes>"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @mcp.tool(tags={"files"})
    async def write_file(
        path: str = Field(..., description="Path where to write the file"),
        content: str = Field(..., description="Text content to write"),
        overwrite: bool = Field(
            default=True, description="Whether to overwrite if exists"
        ),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
        ctx: Context = None,
    ) -> str:
        """
        Write text content to a file in Nextcloud.
        """
        try:
            with get_client(base_url, username, password) as client:

                client.write_file(path, content, overwrite=overwrite)
                return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @mcp.tool(tags={"files"})
    async def create_folder(
        path: str = Field(..., description="Path of the new folder"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        Create a new directory in Nextcloud.
        """
        try:
            with get_client(base_url, username, password) as client:
                client.create_directory(path)
                return f"Successfully created directory: {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    @mcp.tool(tags={"files"})
    async def delete_item(
        path: str = Field(..., description="Path of the file or folder to delete"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
        ctx: Context = None,
    ) -> str:
        """
        Delete a file or directory in Nextcloud.
        """
        if ctx:
            pass

        try:
            with get_client(base_url, username, password) as client:
                client.delete_resource(path)
                return f"Successfully deleted: {path}"
        except Exception as e:
            return f"Error deleting item: {str(e)}"

    @mcp.tool(tags={"files"})
    async def move_item(
        source: str = Field(..., description="Source path"),
        destination: str = Field(..., description="Destination path"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        Move a file or directory to a new location.
        """
        try:
            with get_client(base_url, username, password) as client:
                client.move_resource(source, destination)
                return f"Successfully moved {source} to {destination}"
        except Exception as e:
            return f"Error moving item: {str(e)}"

    @mcp.tool(tags={"files"})
    async def copy_item(
        source: str = Field(..., description="Source path"),
        destination: str = Field(..., description="Destination path"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        Copy a file or directory to a new location.
        """
        try:
            with get_client(base_url, username, password) as client:
                client.copy_resource(source, destination)
                return f"Successfully copied {source} to {destination}"
        except Exception as e:
            return f"Error copying item: {str(e)}"

    @mcp.tool(tags={"files"})
    async def get_properties(
        path: str = Field(default="", description="Path to file/folder"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """
        Get detailed properties for a file or folder.
        """
        try:
            with get_client(base_url, username, password) as client:
                contents = client.list_contents(path)

                formatted = []
                for item in contents:
                    formatted.append(str(item))

                return "\n".join(formatted) if formatted else "No properties found."

        except Exception as e:
            return f"Error getting properties: {str(e)}"


def register_user_tools(mcp: FastMCP):
    @mcp.tool(tags={"user"})
    async def get_user_info(
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """Get information about the current user."""
        try:
            with get_client(base_url, username, password) as client:
                return str(client.get_user_info())
        except Exception as e:
            return f"Error getting user info: {str(e)}"


def register_sharing_tools(mcp: FastMCP):
    @mcp.tool(tags={"sharing"})
    async def list_shares(
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """List all shares."""
        try:
            with get_client(base_url, username, password) as client:
                shares = client.list_shares()
                return json.dumps(shares, indent=2)
        except Exception as e:
            return f"Error listing shares: {str(e)}"

    @mcp.tool(tags={"sharing"})
    async def create_share(
        path: str = Field(..., description="Path to file/folder to share"),
        share_type: int = Field(
            3, description="Share type (0=User, 1=Group, 3=Public Link, 4=Email)"
        ),
        permissions: int = Field(1, description="Permissions (1=Read, 31=All)"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """Create a new share."""
        try:
            with get_client(base_url, username, password) as client:
                result = client.create_share(path, share_type, permissions)
                return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error creating share: {str(e)}"

    @mcp.tool(tags={"sharing"})
    async def delete_share(
        share_id: str = Field(..., description="ID of the share to delete"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """Delete a share."""
        try:
            with get_client(base_url, username, password) as client:
                client.delete_share(share_id)
                return f"Successfully deleted share {share_id}"
        except Exception as e:
            return f"Error deleting share: {str(e)}"


def register_calendar_tools(mcp: FastMCP):
    @mcp.tool(tags={"calendar"})
    async def list_calendars(
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """List available calendars."""
        try:
            with get_client(base_url, username, password) as client:
                cals = client.list_calendars()
                return json.dumps(cals, indent=2)
        except Exception as e:
            return f"Error listing calendars: {str(e)}"

    @mcp.tool(tags={"calendar"})
    async def list_calendar_events(
        calendar_url: str = Field(..., description="URL of the calendar"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """List events in a calendar."""
        try:
            with get_client(base_url, username, password) as client:
                events = client.list_events(calendar_url)
                return json.dumps(events, indent=2)
        except Exception as e:
            return f"Error listing events: {str(e)}"

    @mcp.tool(tags={"calendar"})
    async def create_calendar_event(
        calendar_url: str = Field(..., description="URL of the calendar"),
        summary: str = Field(..., description="Event summary/title"),
        start_time: str = Field(..., description="Start time (ISO format)"),
        end_time: str = Field(..., description="End time (ISO format)"),
        description: str = Field(default="", description="Description"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        try:
            cal = Calendar()
            cal.add("prodid", "-//Nextcloud Agent//mxm.dk//")
            cal.add("version", "2.0")

            event = Event()
            event.add("summary", summary)

            dt_start = dateutil.parser.parse(start_time)
            dt_end = dateutil.parser.parse(end_time)

            event.add("dtstart", dt_start)
            event.add("dtend", dt_end)
            event.add("uid", str(uuid.uuid4()))
            if description:
                event.add("description", description)

            cal.add_component(event)
            ics_data = cal.to_ical().decode("utf-8")

            with get_client(base_url, username, password) as client:
                client.create_event(calendar_url, ics_data)
                return "Successfully created event."
        except Exception as e:
            return f"Error creating event: {str(e)}"


def register_contacts_tools(mcp: FastMCP):
    @mcp.tool(tags={"contacts"})
    async def list_address_books(
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """List address books."""
        try:
            with get_client(base_url, username, password) as client:
                books = client.list_address_books()
                return json.dumps(books, indent=2)
        except Exception as e:
            return f"Error listing address books: {str(e)}"

    @mcp.tool(tags={"contacts"})
    async def list_contacts(
        address_book_url: str = Field(..., description="URL of the address book"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """List contacts in an address book."""
        try:
            with get_client(base_url, username, password) as client:
                contacts = client.list_contacts(address_book_url)
                return json.dumps(contacts, indent=2)
        except Exception as e:
            return f"Error listing contacts: {str(e)}"

    @mcp.tool(tags={"contacts"})
    async def create_contact(
        address_book_url: str = Field(..., description="URL of the address book"),
        vcard_data: str = Field(..., description="Raw VCF/vCard data string"),
        base_url: str = Field(
            default=None, description="Direct override for Nextcloud URL"
        ),
        username: str = Field(default=None, description="Direct override for Username"),
        password: str = Field(default=None, description="Direct override for Password"),
    ) -> str:
        """Create a new contact using raw vCard data."""
        try:
            with get_client(base_url, username, password) as client:
                client.create_contact(address_book_url, vcard_data)
                return "Successfully created contact."
        except Exception as e:
            return f"Error creating contact: {str(e)}"


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize and return the MCP instance, args, and middlewares."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="Nextcloud",
        version=__version__,
        instructions="Nextcloud Agent MCP Server - Manage files, folders, shares, calendar events, and contacts.",
    )

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)
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
    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)
    registered_tags = []
    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{args.name or 'nextcloud-agent'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

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
