import logging

from scion_sdk.config import setting
from scion_sdk.exceptions import AuthError, UnauthorizedError

from nextcloud_agent.api_client import NextcloudAPI

logger = logging.getLogger(__name__)


def make_client(
    base_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> NextcloudAPI:
    """Build one NextcloudAPI client from arguments or the connector settings.

    The SDK per-operation tools call this factory once per call
    (CONN-NEXTCLOUD-AGENT-R002); ``get_client`` wraps it as a FastMCP
    generator dependency that also closes the client.
    """
    if not base_url:
        base_url = setting("NEXTCLOUD_URL", None)
    if not username:
        username = setting("NEXTCLOUD_USERNAME", None)
    if not password:
        password = setting("NEXTCLOUD_PASSWORD", None)
    if not base_url or not username or not password:
        raise ValueError(
            "Nextcloud URL, username, and password must be provided via arguments or environment variables."
        )
    try:
        return NextcloudAPI(base_url=base_url, username=username, password=password)
    except (AuthError, UnauthorizedError) as e:
        raise RuntimeError(
            "AUTHENTICATION ERROR: The configured credentials were rejected. "
            f"Please check your NEXTCLOUD_USERNAME and NEXTCLOUD_PASSWORD environment variables. "
            f"Error details: {type(e).__name__}"
        ) from e


def get_client(
    base_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
):
    """
    Yield a NextcloudAPI client and close it afterwards.

    A plain generator, because FastMCP's ``Depends`` drives generator
    dependencies itself; wrapping it in ``contextlib.contextmanager`` handed
    every tool the context-manager object instead of the client.

    CONCEPT:AU-OS.config.secrets-authentication
    """
    client = make_client(base_url=base_url, username=username, password=password)
    try:
        yield client
    finally:
        client.close()
