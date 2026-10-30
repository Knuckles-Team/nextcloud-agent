from scion_sdk.connector import Connector

from nextcloud_agent.api_client import NextcloudAPI
from nextcloud_agent.auth import get_client

connector = Connector(
    api=NextcloudAPI,
    get_client=get_client,
    tools="nextcloud_agent.mcp_server",
    instructions="nextcloud-agent MCP Server — Condensed Action-Routed Tools.",
)
