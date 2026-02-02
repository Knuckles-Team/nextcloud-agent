import contextlib
import threading
import os
from typing import Optional

from fastmcp.server.middleware import MiddlewareContext, Middleware
from fastmcp.utilities.logging import get_logger

from nextcloud_agent.nextcloud_api import NextcloudAPI

# Thread-local storage for user token
local = threading.local()
logger = get_logger(name="TokenMiddleware")


class UserTokenMiddleware(Middleware):
    def __init__(self, config: dict):
        self.config = config

    async def on_request(self, context: MiddlewareContext, call_next):
        logger.debug(f"Delegation enabled: {self.config['enable_delegation']}")
        if self.config["enable_delegation"]:
            headers = getattr(context.message, "headers", {})
            auth = headers.get("Authorization")
            if auth and auth.startswith("Bearer "):
                token = auth.split(" ")[1]
                local.user_token = token
                local.user_claims = None  # Will be populated by JWTVerifier

                # Extract claims if JWTVerifier already validated
                if hasattr(context, "auth") and hasattr(context.auth, "claims"):
                    local.user_claims = context.auth.claims
                    logger.info(
                        "Stored JWT claims for delegation",
                        extra={"subject": context.auth.claims.get("sub")},
                    )
                else:
                    logger.debug("JWT claims not yet available (will be after auth)")

                logger.info("Extracted Bearer token for delegation")
            else:
                logger.error("Missing or invalid Authorization header")
                raise ValueError("Missing or invalid Authorization header")
        return await call_next(context)


class JWTClaimsLoggingMiddleware(Middleware):
    async def on_response(self, context: MiddlewareContext, call_next):
        response = await call_next(context)
        logger.info(f"JWT Response: {response}")
        if hasattr(context, "auth") and hasattr(context.auth, "claims"):
            logger.info(
                "JWT Authentication Success",
                extra={
                    "subject": context.auth.claims.get("sub"),
                    "client_id": context.auth.claims.get("client_id"),
                    "scopes": context.auth.claims.get("scope"),
                },
            )


@contextlib.contextmanager
def get_client(
    base_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verify: bool = True,
):
    """Context manager to get a NextcloudAPI client."""
    # Use env vars as defaults
    base_url = base_url or os.environ.get("NEXTCLOUD_BASE_URL", "")
    username = username or os.environ.get("NEXTCLOUD_USERNAME", "")
    password = password or os.environ.get("NEXTCLOUD_PASSWORD", "")

    if not base_url or not username or not password:
        raise ValueError(
            "Missing Nextcloud credentials. Please provide them or set env vars: NEXTCLOUD_BASE_URL, NEXTCLOUD_USERNAME, NEXTCLOUD_PASSWORD"
        )

    client = NextcloudAPI(
        base_url=base_url, username=username, password=password, verify=verify
    )
    try:
        yield client
    finally:
        client._session.close()
