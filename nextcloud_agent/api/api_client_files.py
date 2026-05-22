import os
from nextcloud_agent.api.api_client_base import BaseApiClient


class Api(BaseApiClient):
    def list_contents(self, path: str = "") -> list[dict]:
        """List files and folders in a directory using PROPFIND."""
        url = self._get_full_url(path)

        body = """<?xml version="1.0" encoding="UTF-8"?>
            <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
              <d:prop>
                <d:getlastmodified/>
                <d:getcontentlength/>
                <d:getcontenttype/>
                <oc:permissions/>
                <d:resourcetype/>
                <d:getetag/>
                <oc:favorite/>
                <oc:fileid/>
              </d:prop>
            </d:propfind>"""

        response = self._session.request(
            "PROPFIND",
            url,
            data=body,
            headers={"Depth": "1", "Content-Type": "application/xml"},
        )

        if response.status_code == 404:
            raise FileNotFoundError(f"Path not found: {path}")
        response.raise_for_status()

        files = self._parse_propfind_response(response.content)  # type: ignore

        filtered_files = []
        for f in files:
            if f["name"] and f["name"] != os.path.basename(path.strip("/")):
                filtered_files.append(f)
            elif not path.strip("/") and f["name"]:
                filtered_files.append(f)

        return files

    def list_files(self, path: str = "") -> list[dict]:
        """Alias for list_contents to support MCP server action."""
        return self.list_contents(path)

    def read_file(self, path: str) -> bytes:
        """Download a file."""
        url = self._get_full_url(path)
        response = self._session.get(url)
        response.raise_for_status()
        return response.content

    def write_file(
        self, path: str, content: str | bytes, overwrite: bool = True
    ) -> bool:
        """Upload a file."""
        url = self._get_full_url(path)

        if isinstance(content, str):
            content = content.encode("utf-8")

        if not overwrite:
            headers = {"If-None-Match": "*"}
        else:
            headers = {}

        response = self._session.put(url, data=content, headers=headers)

        if not overwrite and response.status_code == 412:
            raise FileExistsError(f"File already exists: {path}")

        response.raise_for_status()
        return True

    def create_directory(self, path: str) -> bool:
        """Create a directory (MKCOL)."""
        url = self._get_full_url(path)
        response = self._session.request("MKCOL", url)

        if response.status_code == 405:
            raise FileExistsError(f"Directory likely already exists: {path}")

        response.raise_for_status()
        return True

    def create_folder(self, path: str) -> bool:
        """Alias for create_directory to support MCP server action."""
        return self.create_directory(path)

    def delete_resource(self, path: str) -> bool:
        """Delete a file or directory."""
        url = self._get_full_url(path)
        response = self._session.delete(url)
        response.raise_for_status()
        return True

    def delete_item(self, path: str) -> bool:
        """Alias for delete_resource to support MCP server action."""
        return self.delete_resource(path)

    def move_resource(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Move a file or directory."""
        source_url = self._get_full_url(source_path)
        dest_url = self._get_full_url(dest_path)

        headers = {"Destination": dest_url, "Overwrite": "T" if overwrite else "F"}
        response = self._session.request("MOVE", source_url, headers=headers)

        if response.status_code == 412:
            raise FileExistsError(f"Destination already exists: {dest_path}")

        response.raise_for_status()
        return True

    def move_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Alias for move_resource to support MCP server action."""
        return self.move_resource(source_path, dest_path, overwrite)

    def copy_resource(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Copy a file or directory."""
        source_url = self._get_full_url(source_path)
        dest_url = self._get_full_url(dest_path)

        headers = {"Destination": dest_url, "Overwrite": "T" if overwrite else "F"}
        response = self._session.request("COPY", source_url, headers=headers)

        if response.status_code == 412:
            raise FileExistsError(f"Destination already exists: {dest_path}")

        response.raise_for_status()
        return True

    def copy_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Alias for copy_resource to support MCP server action."""
        return self.copy_resource(source_path, dest_path, overwrite)

    def get_user_quota(self) -> dict:
        """Get storage quota information."""
        import xml.etree.ElementTree as ET
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
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
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
