from scion_sdk.storage import AlreadyExistsError, DavEntry, NotFoundError

from nextcloud_agent.api.api_client_base import BaseApiClient
from nextcloud_agent.api.xml_security import parse_untrusted_xml

_OC = "http://owncloud.org/ns"
#: The Nextcloud properties a listing keeps beside the standard DAV fields.
NC_PROPS = ((_OC, "permissions"), (_OC, "favorite"), (_OC, "fileid"))


def _entry_dict(webdav_base: str, entry: DavEntry) -> dict:
    """One SDK listing entry in the connector's listing shape."""
    suffix = "/" if entry.is_collection else ""
    href = f"{webdav_base.split('://', 1)[-1].split('/', 1)[-1]}/{entry.path}".rstrip(
        "/"
    )
    return {
        "href": f"/{href}{suffix}",
        "name": entry.name,
        "is_folder": entry.is_collection,
        "last_modified": entry.last_modified or "",
        "etag": f'"{entry.etag}"' if entry.etag else "",
        "content_type": entry.content_type or "",
        "content_length": str(entry.size),
        "permissions": entry.props.get("permissions", ""),
        "favorite": entry.props.get("favorite") or "0",
        "file_id": entry.props.get("fileid", ""),
    }


class Api(BaseApiClient):
    """File operations run through the SDK WebDAV backend (CONN-NEXTCLOUD-AGENT-R001)."""

    def list_contents(self, path: str = "") -> list[dict]:
        """List files and folders in a directory (PROPFIND, depth 1)."""
        self._get_full_url(path)  # the shared path guard
        try:
            entries = self.files_dav.propfind(path.strip("/"), 1, extra_props=NC_PROPS)
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None
        return [_entry_dict(self.webdav_base, entry) for entry in entries]

    def list_files(self, path: str = "") -> list[dict]:
        """Alias for list_contents to support MCP server action."""
        return self.list_contents(path)

    def read_file(self, path: str) -> bytes:
        """Download a file."""
        self._get_full_url(path)
        try:
            return self.files_dav.read_path(path.strip("/"))
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None

    def write_file(
        self, path: str, content: str | bytes, overwrite: bool = True
    ) -> bool:
        """Upload a file, creating missing parent folders."""
        self._get_full_url(path)
        if isinstance(content, str):
            content = content.encode("utf-8")
        try:
            self.files_dav.write_path(path.strip("/"), content, overwrite=overwrite)
        except AlreadyExistsError:
            raise FileExistsError("Configured remote file already exists") from None
        return True

    def create_directory(self, path: str) -> bool:
        """Create a directory (MKCOL)."""
        self._get_full_url(path)
        try:
            self.files_dav.make_collection(path.strip("/"), exist_ok=False)
        except AlreadyExistsError:
            raise FileExistsError(
                "Configured remote directory likely already exists"
            ) from None
        return True

    def create_folder(self, path: str) -> bool:
        """Alias for create_directory to support MCP server action."""
        return self.create_directory(path)

    def delete_resource(self, path: str) -> bool:
        """Delete a file or directory."""
        self._get_full_url(path)
        try:
            self.files_dav.delete_path(path.strip("/"))
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None
        return True

    def delete_item(self, path: str) -> bool:
        """Alias for delete_resource to support MCP server action."""
        return self.delete_resource(path)

    def _transfer(self, move: bool, source: str, dest: str, overwrite: bool) -> bool:
        self._get_full_url(source)
        self._get_full_url(dest)
        method = self.files_dav.move_path if move else self.files_dav.copy_path
        try:
            method(source.strip("/"), dest.strip("/"), overwrite=overwrite)
        except AlreadyExistsError:
            raise FileExistsError(
                "Configured remote destination already exists"
            ) from None
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None
        return True

    def move_resource(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Move a file or directory."""
        return self._transfer(True, source_path, dest_path, overwrite)

    def move_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Alias for move_resource to support MCP server action."""
        return self.move_resource(source_path, dest_path, overwrite)

    def copy_resource(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Copy a file or directory."""
        return self._transfer(False, source_path, dest_path, overwrite)

    def copy_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Alias for copy_resource to support MCP server action."""
        return self.copy_resource(source_path, dest_path, overwrite)

    def get_user_quota(self) -> dict:
        """Get storage quota information."""
        url = self.webdav_base
        body = """<?xml version="1.0" encoding="UTF-8"?>
            <d:propfind xmlns:d="DAV:">
              <d:prop>
                <d:quota-available-bytes/>
                <d:quota-used-bytes/>
              </d:prop>
            </d:propfind>"""

        response = self._session.request(
            "PROPFIND",
            url,
            data=body,
            headers={"Depth": "0", "Content-Type": "application/xml"},
            stream=True,
            timeout=(10, 30),
        )

        root = parse_untrusted_xml(self._read_xml_response(response))
        ns = {"d": "DAV:"}

        prop = root.find(".//d:prop", ns)

        if prop is None:
            return {}

        return {
            "quota_available": prop.findtext(
                "d:quota-available-bytes", default="-2", namespaces=ns
            ),
            "quota_used": prop.findtext(
                "d:quota-used-bytes", default="0", namespaces=ns
            ),
        }

    def list_shares(self) -> list[dict] | dict:
        """List all shares."""
        return self.ocs_request("GET", "apps/files_sharing/api/v1/shares")

    def create_share(
        self, path: str, share_type: int = 3, permissions: int = 1
    ) -> dict:
        """Create a share."""
        data = {"path": path, "shareType": share_type, "permissions": permissions}
        return self.ocs_request("POST", "apps/files_sharing/api/v1/shares", data=data)

    def delete_share(self, share_id: str) -> bool:
        """Delete a share."""
        self.ocs_request("DELETE", f"apps/files_sharing/api/v1/shares/{share_id}")
        return True

    def get_user_info(self) -> dict:
        """Get current user info."""
        return self.ocs_request("GET", "cloud/user")

    def get_properties(self, path: str = "") -> dict:
        """Helper to get properties of a path by listing its contents."""
        files = self.list_contents(path)
        if not files:
            return {}
        return files[0]
