from scion_sdk.connector import Connector

from nextcloud_agent.api_client import NextcloudAPI
from nextcloud_agent.auth import make_client

connector = Connector(
    api=NextcloudAPI,
    get_client=make_client,
    tools="nextcloud_agent.mcp_server",
    instructions="nextcloud-agent MCP Server — Condensed Action-Routed Tools.",
)
