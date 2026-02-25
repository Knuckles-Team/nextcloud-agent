import os
import logging
from contextlib import contextmanager
from nextcloud_agent.nextcloud_api import NextcloudAPI
from agent_utilities.base_utilities import to_boolean

logger = logging.getLogger(__name__)

@contextmanager
def get_client(
    base_url: str = None,
    username: str = None,
    password: str = None,
    verify: bool = None,
):
    """
    Returns a NextcloudAPI client.
    """
    if not base_url:
        base_url = os.getenv("NEXTCLOUD_URL")
    if not username:
        username = os.getenv("NEXTCLOUD_USERNAME")
    if not password:
        password = os.getenv("NEXTCLOUD_PASSWORD")
    if verify is None:
        verify = to_boolean(os.getenv("NEXTCLOUD_SSL_VERIFY", "True"))

    if not base_url or not username or not password:
        raise ValueError("Nextcloud URL, username, and password must be provided via arguments or environment variables.")

    client = NextcloudAPI(base_url=base_url, username=username, password=password, verify=verify)
    yield client
