from scion_sdk.connector import Connector

from nextcloud_agent.api_client import NextcloudAPI
from nextcloud_agent.auth import make_client

connector = Connector(
    api=NextcloudAPI,
    get_client=make_client,
    instructions=(
        "Nextcloud MCP server: one typed tool per operation for files, versions, search, "
        "sharing, calendar, tasks, contacts, Talk, Deck, Notes, office documents, activity, "
        "notifications and user, group and app administration."
    ),
)
