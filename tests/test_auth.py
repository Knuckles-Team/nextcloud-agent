import pytest
import os
from unittest.mock import patch
from agent_utilities.core.exceptions import AuthError, UnauthorizedError
from nextcloud_agent.auth import get_client, NextcloudAPI


@pytest.mark.concept("OS-5.1")
def test_auth_coverage():
    # 1. Parameter arguments passed directly
    with get_client(
        base_url="https://direct.test.com",
        username="direct_user",
        password="direct_password",
        verify=False,
    ) as client:
        assert client.base_url == "https://direct.test.com"
        assert client.username == "direct_user"

    # 2. Parameter fallback to environment variables
    with patch.dict(
        os.environ,
        {
            "NEXTCLOUD_URL": "https://env.test.com",
            "NEXTCLOUD_USERNAME": "env_user",
            "NEXTCLOUD_PASSWORD": "env_password",
            "NEXTCLOUD_SSL_VERIFY": "True",
        },
    ):
        with get_client() as client:
            assert client.base_url == "https://env.test.com"
            assert client.username == "env_user"
            assert client.verify is True

    # 3. Parameter missing error
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError, match="Nextcloud URL, username, and password must be provided"
        ):
            with get_client():
                pass

    # 4. AuthException mapping
    with patch(
        "nextcloud_agent.auth.NextcloudAPI", side_effect=AuthError("Auth failed")
    ):
        with pytest.raises(RuntimeError, match="AUTHENTICATION ERROR"):
            with get_client(base_url="https://x.com", username="u", password="p"):
                pass

    with patch(
        "nextcloud_agent.auth.NextcloudAPI",
        side_effect=UnauthorizedError("Unauthorized"),
    ):
        with pytest.raises(RuntimeError, match="AUTHENTICATION ERROR"):
            with get_client(base_url="https://x.com", username="u", password="p"):
                pass
