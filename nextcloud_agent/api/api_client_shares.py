"""Sharing over the OCS Sharing API (CONN-NEXTCLOUD-AGENT-R005)."""

from typing import Literal

from nextcloud_agent.api.api_client_base import BaseApiClient

_SHARES = "apps/files_sharing/api/v1/shares"
#: OCS share types by name.
SHARE_TYPES = {"user": 0, "group": 1, "public_link": 3, "email": 4, "talk": 10}


def _share_id(share_id: str) -> str:
    if not str(share_id).isdigit():
        raise ValueError("share identifier is invalid")
    return str(share_id)


class NextcloudShares(BaseApiClient):
    """Internal (user, group, Talk) shares and public links on files and folders."""

    def list_shares(
        self,
        path: str | None = None,
        subfiles: bool = False,
        shared_with_me: bool = False,
    ) -> list[dict]:
        """List shares: all of the account's shares, the shares on one path,
        or the shares other people gave the account.

        GET /ocs/v2.php/apps/files_sharing/api/v1/shares
        """
        params: dict = {}
        if path:
            params["path"] = path
            if subfiles:
                params["subfiles"] = "true"
        if shared_with_me:
            params["shared_with_me"] = "true"
        return self.ocs_request("GET", _SHARES, params=params) or []

    def get_share(self, share_id: str) -> dict:
        """Read one share by id.

        GET /ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}
        """
        data = self.ocs_request("GET", f"{_SHARES}/{_share_id(share_id)}")
        return data[0] if isinstance(data, list) and data else data

    def create_share(
        self,
        path: str,
        share_type: Literal[
            "user", "group", "public_link", "email", "talk"
        ] = "public_link",
        share_with: str | None = None,
        permissions: int = 1,
        password: str | None = None,
        expire_date: str | None = None,
        label: str | None = None,
        note: str | None = None,
        public_upload: bool = False,
        send_mail: bool = False,
    ) -> dict:
        """Share one file or folder and return the share (a public link carries its ``url``).

        POST /ocs/v2.php/apps/files_sharing/api/v1/shares

        ``share_type`` user/group/talk needs ``share_with`` (user id, group id
        or conversation token). ``permissions`` is the OCS bitmask: 1 read,
        2 update, 4 create, 8 delete, 16 share (31 all). ``expire_date`` is
        ``YYYY-MM-DD``. ``send_mail`` false (the default) creates the share
        without the server mailing the recipient.
        """
        kind = SHARE_TYPES[share_type]
        if kind in (0, 1, 4, 10) and not share_with:
            raise ValueError("share_with is required for this share type")
        data: dict = {
            "path": path,
            "shareType": kind,
            "permissions": permissions,
            "sendMail": "true" if send_mail else "false",
        }
        optional = {
            "shareWith": share_with,
            "password": password,
            "expireDate": expire_date,
            "label": label,
            "note": note,
        }
        data.update({k: v for k, v in optional.items() if v})
        if public_upload:
            data["publicUpload"] = "true"
        return self.ocs_request("POST", _SHARES, data=data)

    def update_share(
        self,
        share_id: str,
        permissions: int | None = None,
        password: str | None = None,
        expire_date: str | None = None,
        label: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Change one share's permissions, password, expiry, label or note.

        PUT /ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}
        """
        fields = {
            "permissions": permissions,
            "password": password,
            "expireDate": expire_date,
            "label": label,
            "note": note,
        }
        data = {k: v for k, v in fields.items() if v is not None}
        if not data:
            raise ValueError("no share field to update")
        return self.ocs_request("PUT", f"{_SHARES}/{_share_id(share_id)}", data=data)

    def delete_share(self, share_id: str) -> bool:
        """Revoke one share.

        DELETE /ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}
        """
        self.ocs_request("DELETE", f"{_SHARES}/{_share_id(share_id)}")
        return True
