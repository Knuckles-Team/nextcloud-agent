"""User, group and app administration over the provisioning API (CONN-NEXTCLOUD-AGENT-R013)."""

from typing import Literal
from urllib.parse import quote

from nextcloud_agent.api.api_client_base import BaseApiClient


def _id(value: str) -> str:
    if (
        not value
        or len(value) > 256
        or any(ord(ch) < 32 for ch in value)
        or "/" in value
    ):
        raise ValueError("identifier is invalid")
    return quote(value, safe="")


class NextcloudAdmin(BaseApiClient):
    """Accounts, groups and installed apps (admin rights needed for writes)."""

    def get_user_info(self) -> dict:
        """Read the connected account: id, display name, email, quota, groups.

        GET /ocs/v2.php/cloud/user
        """
        return self.ocs_request("GET", "cloud/user")

    def list_users(
        self, search: str = "", limit: int = 100, offset: int = 0
    ) -> list[str]:
        """List account ids, optionally matching a search string.

        GET /ocs/v2.php/cloud/users
        """
        params = {"search": search, "limit": int(limit), "offset": int(offset)}
        return (self.ocs_request("GET", "cloud/users", params=params) or {}).get(
            "users", []
        )

    def get_user(self, user_id: str) -> dict:
        """Read one account's details.

        GET /ocs/v2.php/cloud/users/{user_id}
        """
        return self.ocs_request("GET", f"cloud/users/{_id(user_id)}")

    def create_user(
        self,
        user_id: str,
        password: str | None = None,
        email: str | None = None,
        display_name: str | None = None,
        groups: list[str] | None = None,
        quota: str | None = None,
    ) -> dict:
        """Create one account. Give a password, or an email for a set-password invite.

        POST /ocs/v2.php/cloud/users
        """
        if not password and not email:
            raise ValueError("a password or an email is required")
        data: dict = {"userid": user_id}
        for key, value in (
            ("password", password),
            ("email", email),
            ("displayName", display_name),
            ("quota", quota),
        ):
            if value:
                data[key] = value
        if groups:
            data["groups[]"] = list(groups)
        return self.ocs_request("POST", "cloud/users", data=data)

    def update_user(
        self,
        user_id: str,
        key: Literal[
            "email",
            "displayname",
            "quota",
            "password",
            "phone",
            "address",
            "website",
            "twitter",
            "language",
            "locale",
        ],
        value: str,
    ) -> bool:
        """Change one field of an account.

        PUT /ocs/v2.php/cloud/users/{user_id}
        """
        self.ocs_request(
            "PUT", f"cloud/users/{_id(user_id)}", data={"key": key, "value": value}
        )
        return True

    def disable_user(self, user_id: str) -> bool:
        """Disable one account (it can no longer log in; data is kept).

        PUT /ocs/v2.php/cloud/users/{user_id}/disable
        """
        self.ocs_request("PUT", f"cloud/users/{_id(user_id)}/disable")
        return True

    def enable_user(self, user_id: str) -> bool:
        """Enable one disabled account.

        PUT /ocs/v2.php/cloud/users/{user_id}/enable
        """
        self.ocs_request("PUT", f"cloud/users/{_id(user_id)}/enable")
        return True

    def delete_user(self, user_id: str) -> bool:
        """Delete one account and its files.

        DELETE /ocs/v2.php/cloud/users/{user_id}
        """
        self.ocs_request("DELETE", f"cloud/users/{_id(user_id)}")
        return True

    def list_groups(self, search: str = "", limit: int = 100) -> list[str]:
        """List group ids, optionally matching a search string.

        GET /ocs/v2.php/cloud/groups
        """
        params = {"search": search, "limit": int(limit)}
        return (self.ocs_request("GET", "cloud/groups", params=params) or {}).get(
            "groups", []
        )

    def create_group(self, group_id: str, display_name: str | None = None) -> bool:
        """Create one group.

        POST /ocs/v2.php/cloud/groups
        """
        data = {"groupid": group_id}
        if display_name:
            data["displayname"] = display_name
        self.ocs_request("POST", "cloud/groups", data=data)
        return True

    def delete_group(self, group_id: str) -> bool:
        """Delete one group (its members keep their accounts).

        DELETE /ocs/v2.php/cloud/groups/{group_id}
        """
        self.ocs_request("DELETE", f"cloud/groups/{_id(group_id)}")
        return True

    def list_group_members(self, group_id: str) -> list[str]:
        """List the account ids in one group.

        GET /ocs/v2.php/cloud/groups/{group_id}/users
        """
        data = self.ocs_request("GET", f"cloud/groups/{_id(group_id)}/users") or {}
        return data.get("users", [])

    def add_user_to_group(self, user_id: str, group_id: str) -> bool:
        """Add one account to a group.

        POST /ocs/v2.php/cloud/users/{user_id}/groups
        """
        self.ocs_request(
            "POST", f"cloud/users/{_id(user_id)}/groups", data={"groupid": group_id}
        )
        return True

    def remove_user_from_group(self, user_id: str, group_id: str) -> bool:
        """Remove one account from a group.

        DELETE /ocs/v2.php/cloud/users/{user_id}/groups
        """
        self.ocs_request(
            "DELETE", f"cloud/users/{_id(user_id)}/groups", data={"groupid": group_id}
        )
        return True

    def list_apps(
        self, status: Literal["enabled", "disabled", "all"] = "enabled"
    ) -> list[str]:
        """List installed app ids that are enabled, disabled, or all of them.

        GET /ocs/v2.php/cloud/apps
        """
        params = {} if status == "all" else {"filter": status}
        return (self.ocs_request("GET", "cloud/apps", params=params) or {}).get(
            "apps", []
        )

    def get_app(self, app_id: str) -> dict:
        """Read one installed app's metadata (name, version, dependencies).

        GET /ocs/v2.php/cloud/apps/{app_id}
        """
        return self.ocs_request("GET", f"cloud/apps/{_id(app_id)}")

    def enable_app(self, app_id: str) -> bool:
        """Enable one installed app.

        POST /ocs/v2.php/cloud/apps/{app_id}
        """
        self.ocs_request("POST", f"cloud/apps/{_id(app_id)}")
        return True

    def disable_app(self, app_id: str) -> bool:
        """Disable one app.

        DELETE /ocs/v2.php/cloud/apps/{app_id}
        """
        self.ocs_request("DELETE", f"cloud/apps/{_id(app_id)}")
        return True
