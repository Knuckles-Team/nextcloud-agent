#!/usr/bin/python
# coding: utf-8

import os
import argparse
import sys
import logging
import json
import uuid
import dateutil.parser
import requests
from typing import List, Union, Dict
from pydantic import Field
from eunomia_mcp.middleware import EunomiaMcpMiddleware
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.utilities.logging import get_logger
from icalendar import Calendar, Event
from nextcloud_agent.utils import to_boolean, to_integer
from nextcloud_agent.middlewares import (
    UserTokenMiddleware,
    JWTClaimsLoggingMiddleware,
    get_client,
)

__version__ = "0.2.6"
print(f"Nextcloud MCP v{__version__}")

logger = get_logger(name="TokenMiddleware")
logger.setLevel(logging.DEBUG)

config = {
    "enable_delegation": to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
    "audience": os.environ.get("AUDIENCE", None),
    "delegated_scopes": os.environ.get("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,
    "oidc_client_id": os.environ.get("OIDC_CLIENT_ID", None),
    "oidc_client_secret": os.environ.get("OIDC_CLIENT_SECRET", None),
    "oidc_config_url": os.environ.get("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}

DEFAULT_TRANSPORT = os.getenv("TRANSPORT", "stdio")
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "8000"))


def register_prompts(mcp: FastMCP):
    @mcp.prompt(name="list_files", description="List files in a directory.")
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


def register_tools(mcp: FastMCP):
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check() -> Dict:
        return {"status": "OK"}

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


def nextcloud_mcp() -> None:
    """Run the Nextcloud MCP server with specified transport and connection parameters.

    This function parses command-line arguments to configure and start the MCP server for Nextcloud API interactions.
    It supports stdio or TCP transport modes and exits on invalid arguments or help requests.
    """
    parser = argparse.ArgumentParser(add_help=False, description="Nextcloud MCP Server")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["stdio", "streamable-http", "sse"],
        help="Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    parser.add_argument(
        "--token-algorithm",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification.",
    )
    parser.add_argument(
        "--token-public-key",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., Nextcloud.read,Nextcloud.write).",
    )
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    parser.add_argument(
        "--eunomia-type",
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=os.environ.get("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=os.getenv("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password",
        default=os.getenv("OPENAPI_PASSWORD"),
        help="Password for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=os.getenv("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret",
        default=os.getenv("OPENAPI_CLIENT_SECRET"),
        help="OAuth client secret for OpenAPI import",
    )

    parser.add_argument("--help", action="store_true", help="Show usage")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:

        usage()

        sys.exit(0)

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    config["enable_delegation"] = args.enable_delegation
    config["audience"] = args.audience or config["audience"]
    config["delegated_scopes"] = args.delegated_scopes or config["delegated_scopes"]
    config["oidc_config_url"] = args.oidc_config_url or config["oidc_config_url"]
    config["oidc_client_id"] = args.oidc_client_id or config["oidc_client_id"]
    config["oidc_client_secret"] = (
        args.oidc_client_secret or config["oidc_client_secret"]
    )

    if config["enable_delegation"]:
        if args.auth_type != "oidc-proxy":
            logger.error("Token delegation requires auth-type=oidc-proxy")
            sys.exit(1)
        if not config["audience"]:
            logger.error("audience is required for delegation")
            sys.exit(1)
        if not all(
            [
                config["oidc_config_url"],
                config["oidc_client_id"],
                config["oidc_client_secret"],
            ]
        ):
            logger.error(
                "Delegation requires complete OIDC configuration (oidc-config-url, oidc-client-id, oidc-client-secret)"
            )
            sys.exit(1)

        try:
            logger.info(
                "Fetching OIDC configuration",
                extra={"oidc_config_url": config["oidc_config_url"]},
            )
            oidc_config_resp = requests.get(config["oidc_config_url"])
            oidc_config_resp.raise_for_status()
            oidc_config = oidc_config_resp.json()
            config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not config["token_endpoint"]:
                logger.error("No token_endpoint found in OIDC configuration")
                raise ValueError("No token_endpoint found in OIDC configuration")
            logger.info(
                "OIDC configuration fetched successfully",
                extra={"token_endpoint": config["token_endpoint"]},
            )
        except Exception as e:
            print(f"Failed to fetch OIDC configuration: {e}")
            logger.error(
                "Failed to fetch OIDC configuration",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            sys.exit(1)

    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        jwks_uri = args.token_jwks_uri or os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
        issuer = args.token_issuer or os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
        audience = args.token_audience or os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
        algorithm = args.token_algorithm
        secret_or_key = args.token_secret or args.token_public_key
        public_key_pem = None

        if not (jwks_uri or secret_or_key):
            logger.error(
                "JWT auth requires either --token-jwks-uri or --token-secret/--token-public-key"
            )
            sys.exit(1)
        if not (issuer and audience):
            logger.error("JWT requires --token-issuer and --token-audience")
            sys.exit(1)

        if args.token_public_key and os.path.isfile(args.token_public_key):
            try:
                with open(args.token_public_key, "r") as f:
                    public_key_pem = f.read()
                logger.info(f"Loaded static public key from {args.token_public_key}")
            except Exception as e:
                print(f"Failed to read public key file: {e}")
                logger.error(f"Failed to read public key file: {e}")
                sys.exit(1)
        elif args.token_public_key:
            public_key_pem = args.token_public_key

        if jwks_uri and (algorithm or secret_or_key):
            logger.warning(
                "JWKS mode ignores --token-algorithm and --token-secret/--token-public-key"
            )

        if algorithm and algorithm.startswith("HS"):
            if not secret_or_key:
                logger.error(f"HMAC algorithm {algorithm} requires --token-secret")
                sys.exit(1)
            if jwks_uri:
                logger.error("Cannot use --token-jwks-uri with HMAC")
                sys.exit(1)
            public_key = secret_or_key
        else:
            public_key = public_key_pem

        required_scopes = None
        if args.required_scopes:
            required_scopes = [
                s.strip() for s in args.required_scopes.split(",") if s.strip()
            ]

        try:
            auth = JWTVerifier(
                jwks_uri=jwks_uri,
                public_key=public_key,
                issuer=issuer,
                audience=audience,
                algorithm=(
                    algorithm if algorithm and algorithm.startswith("HS") else None
                ),
                required_scopes=required_scopes,
            )
            logger.info(
                "JWTVerifier configured",
                extra={
                    "mode": (
                        "JWKS"
                        if jwks_uri
                        else (
                            "HMAC"
                            if algorithm and algorithm.startswith("HS")
                            else "Static Key"
                        )
                    ),
                    "algorithm": algorithm,
                    "required_scopes": required_scopes,
                },
            )
        except Exception as e:
            print(f"Failed to initialize JWTVerifier: {e}")
            logger.error(f"Failed to initialize JWTVerifier: {e}")
            sys.exit(1)
    elif args.auth_type == "oauth-proxy":
        if not (
            args.oauth_upstream_auth_endpoint
            and args.oauth_upstream_token_endpoint
            and args.oauth_upstream_client_id
            and args.oauth_upstream_client_secret
            and args.oauth_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience"
            )
            logger.error(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience",
                extra={
                    "auth_endpoint": args.oauth_upstream_auth_endpoint,
                    "token_endpoint": args.oauth_upstream_token_endpoint,
                    "client_id": args.oauth_upstream_client_id,
                    "base_url": args.oauth_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
            )
            sys.exit(1)
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not (
            args.oidc_config_url
            and args.oidc_client_id
            and args.oidc_client_secret
            and args.oidc_base_url
        ):
            logger.error(
                "oidc-proxy requires oidc-config-url, oidc-client-id, oidc-client-secret, oidc-base-url",
                extra={
                    "config_url": args.oidc_config_url,
                    "client_id": args.oidc_client_id,
                    "base_url": args.oidc_base_url,
                },
            )
            sys.exit(1)
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not (
            args.remote_auth_servers
            and args.remote_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            logger.error(
                "remote-oauth requires remote-auth-servers, remote-base-url, token-jwks-uri, token-issuer, token-audience",
                extra={
                    "auth_servers": args.remote_auth_servers,
                    "base_url": args.remote_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
            )
            sys.exit(1)
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )

    middlewares: List[
        Union[
            UserTokenMiddleware,
            ErrorHandlingMiddleware,
            RateLimitingMiddleware,
            TimingMiddleware,
            LoggingMiddleware,
            JWTClaimsLoggingMiddleware,
            EunomiaMcpMiddleware,
        ]
    ] = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
        TimingMiddleware(),
        LoggingMiddleware(),
        JWTClaimsLoggingMiddleware(),
    ]
    if config["enable_delegation"] or args.auth_type == "jwt":
        middlewares.insert(0, UserTokenMiddleware(config=config))

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from eunomia_mcp import create_eunomia_middleware

            policy_file = args.eunomia_policy_file or "mcp_policies.json"
            eunomia_endpoint = (
                args.eunomia_remote_url if args.eunomia_type == "remote" else None
            )
            eunomia_mw = create_eunomia_middleware(
                policy_file=policy_file, eunomia_endpoint=eunomia_endpoint
            )
            middlewares.append(eunomia_mw)
            logger.info(f"Eunomia middleware enabled ({args.eunomia_type})")
        except Exception as e:
            print(f"Failed to load Eunomia middleware: {e}")
            logger.error("Failed to load Eunomia middleware", extra={"error": str(e)})
            sys.exit(1)

    mcp = FastMCP("Nextcloud", auth=auth)
    register_tools(mcp)
    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print("\nStarting Nextcloud MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


def usage():
    print(
        f"Nextcloud Agent ({__version__}): Nextcloud MCP Server\n\n"
        "Usage:\n"
        "-t | --transport                   [ Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio) ]\n"
        "-s | --host                        [ Host address for HTTP transport (default: 0.0.0.0) ]\n"
        "-p | --port                        [ Port number for HTTP transport (default: 8000) ]\n"
        "--auth-type                        [ Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none) ]\n"
        "--token-jwks-uri                   [ JWKS URI for JWT verification ]\n"
        "--token-issuer                     [ Issuer for JWT verification ]\n"
        "--token-audience                   [ Audience for JWT verification ]\n"
        "--token-algorithm                  [ JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS. ]\n"
        "--token-secret                     [ Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification. ]\n"
        "--token-public-key                 [ Path to PEM public key file or inline PEM string (for static asymmetric keys). ]\n"
        "--required-scopes                  [ Comma-separated list of required scopes (e.g., Nextcloud.read,Nextcloud.write). ]\n"
        "--oauth-upstream-auth-endpoint     [ Upstream authorization endpoint for OAuth Proxy ]\n"
        "--oauth-upstream-token-endpoint    [ Upstream token endpoint for OAuth Proxy ]\n"
        "--oauth-upstream-client-id         [ Upstream client ID for OAuth Proxy ]\n"
        "--oauth-upstream-client-secret     [ Upstream client secret for OAuth Proxy ]\n"
        "--oauth-base-url                   [ Base URL for OAuth Proxy ]\n"
        "--oidc-config-url                  [ OIDC configuration URL ]\n"
        "--oidc-client-id                   [ OIDC client ID ]\n"
        "--oidc-client-secret               [ OIDC client secret ]\n"
        "--oidc-base-url                    [ Base URL for OIDC Proxy ]\n"
        "--remote-auth-servers              [ Comma-separated list of authorization servers for Remote OAuth ]\n"
        "--remote-base-url                  [ Base URL for Remote OAuth ]\n"
        "--allowed-client-redirect-uris     [ Comma-separated list of allowed client redirect URIs ]\n"
        "--eunomia-type                     [ Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none) ]\n"
        "--eunomia-policy-file              [ Policy file for embedded Eunomia (default: mcp_policies.json) ]\n"
        "--eunomia-remote-url               [ URL for remote Eunomia server ]\n"
        "--enable-delegation                [ Enable OIDC token delegation ]\n"
        "--audience                         [ Audience for the delegated token ]\n"
        "--delegated-scopes                 [ Scopes for the delegated token (space-separated) ]\n"
        "--openapi-file                     [ Path to the OpenAPI JSON file to import additional tools from ]\n"
        "--openapi-base-url                 [ Base URL for the OpenAPI client (overrides instance URL) ]\n"
        "--openapi-use-token                [ Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import ]\n"
        "--openapi-username                 [ Username for basic auth during OpenAPI import ]\n"
        "--openapi-password                 [ Password for basic auth during OpenAPI import ]\n"
        "--openapi-client-id                [ OAuth client ID for OpenAPI import ]\n"
        "--openapi-client-secret            [ OAuth client secret for OpenAPI import ]\n"
        "\n"
        "Examples:\n"
        "  [Simple]  nextcloud-mcp \n"
        '  [Complex] nextcloud-mcp --transport "value" --host "value" --port "value" --auth-type "value" --token-jwks-uri "value" --token-issuer "value" --token-audience "value" --token-algorithm "value" --token-secret "value" --token-public-key "value" --required-scopes "value" --oauth-upstream-auth-endpoint "value" --oauth-upstream-token-endpoint "value" --oauth-upstream-client-id "value" --oauth-upstream-client-secret "value" --oauth-base-url "value" --oidc-config-url "value" --oidc-client-id "value" --oidc-client-secret "value" --oidc-base-url "value" --remote-auth-servers "value" --remote-base-url "value" --allowed-client-redirect-uris "value" --eunomia-type "value" --eunomia-policy-file "value" --eunomia-remote-url "value" --enable-delegation --audience "value" --delegated-scopes "value" --openapi-file "value" --openapi-base-url "value" --openapi-use-token --openapi-username "value" --openapi-password "value" --openapi-client-id "value" --openapi-client-secret "value"\n'
    )


if __name__ == "__main__":
    nextcloud_mcp()
