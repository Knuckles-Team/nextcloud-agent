import logging
from contextlib import contextmanager

from agent_utilities.core.config import setting
from agent_utilities.core.exceptions import AuthError, UnauthorizedError

from nextcloud_agent.api_client import NextcloudAPI

logger = logging.getLogger(__name__)


@contextmanager
def get_client(
    base_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify: bool | None = None,
):
    """
    Returns a NextcloudAPI client.

    CONCEPT:AU-OS.config.secrets-authentication
    """
    if not base_url:
        base_url = setting("NEXTCLOUD_URL", None)
    if not username:
        username = setting("NEXTCLOUD_USERNAME", None)
    if not password:
        password = setting("NEXTCLOUD_PASSWORD", None)
    if verify is None:
        verify = setting("NEXTCLOUD_SSL_VERIFY", True)

    if not base_url or not username or not password:
        raise ValueError(
            "Nextcloud URL, username, and password must be provided via arguments or environment variables."
        )

    try:
        client = NextcloudAPI(
            base_url=base_url, username=username, password=password, verify=verify
        )
    except (AuthError, UnauthorizedError) as e:
        raise RuntimeError(
            f"AUTHENTICATION ERROR: The Nextcloud credentials provided are not valid for '{base_url}'. "
            f"Please check your NEXTCLOUD_USERNAME and NEXTCLOUD_PASSWORD environment variables. "
            f"Error details: {str(e)}"
        ) from e
    yield client
