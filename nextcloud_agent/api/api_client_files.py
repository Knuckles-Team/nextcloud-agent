import base64
from typing import Literal
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape

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


class NextcloudFiles(BaseApiClient):
    """File operations: WebDAV through the SDK backend (CONN-NEXTCLOUD-AGENT-R001),
    plus versions, search and trash bin over the Nextcloud DAV endpoints (R004)."""

    def list_files(self, path: str = "") -> list[dict]:
        """List the files and folders in one folder (PROPFIND, depth 1).

        PROPFIND /remote.php/dav/files/{user}/{path}
        """
        self._get_full_url(path)  # the shared path guard
        try:
            entries = self.files_dav.propfind(path.strip("/"), 1, extra_props=NC_PROPS)
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None
        return [_entry_dict(self.webdav_base, entry) for entry in entries]

    def read_file(self, path: str) -> bytes:
        """Download one file; the bytes come back base64-encoded.

        GET /remote.php/dav/files/{user}/{path}
        """
        self._get_full_url(path)
        try:
            return self.files_dav.read_path(path.strip("/"))
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None

    def write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
        encoding: Literal["utf-8", "base64"] = "utf-8",
    ) -> bool:
        """Upload one file, creating missing parent folders.

        PUT /remote.php/dav/files/{user}/{path}

        Pass text with ``encoding="utf-8"``, or binary content (an image, a
        video, an office document) base64-encoded with ``encoding="base64"``.
        Uploading over an existing file keeps the old content as a version.
        """
        self._get_full_url(path)
        data = (
            base64.b64decode(content, validate=True)
            if encoding == "base64"
            else content.encode("utf-8")
        )
        try:
            self.files_dav.write_path(path.strip("/"), data, overwrite=overwrite)
        except AlreadyExistsError:
            raise FileExistsError("Configured remote file already exists") from None
        return True

    def create_folder(self, path: str) -> bool:
        """Create one folder.

        MKCOL /remote.php/dav/files/{user}/{path}
        """
        self._get_full_url(path)
        try:
            self.files_dav.make_collection(path.strip("/"), exist_ok=False)
        except AlreadyExistsError:
            raise FileExistsError(
                "Configured remote directory likely already exists"
            ) from None
        return True

    def delete_item(self, path: str) -> bool:
        """Delete one file or folder (it moves to the trash bin).

        DELETE /remote.php/dav/files/{user}/{path}
        """
        self._get_full_url(path)
        try:
            self.files_dav.delete_path(path.strip("/"))
        except NotFoundError:
            raise FileNotFoundError("Configured remote path was not found") from None
        return True

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

    def move_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Move or rename one file or folder.

        MOVE /remote.php/dav/files/{user}/{source_path}
        """
        return self._transfer(True, source_path, dest_path, overwrite)

    def copy_item(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Copy one file or folder.

        COPY /remote.php/dav/files/{user}/{source_path}
        """
        return self._transfer(False, source_path, dest_path, overwrite)

    def get_properties(self, path: str) -> dict:
        """Read the properties of one file or folder (size, type, etag, file id).

        PROPFIND /remote.php/dav/files/{user}/{path}
        """
        files = self.list_files(path)
        if not files:
            return {}
        return files[0]

    def get_user_quota(self) -> dict:
        """Read the storage quota of the connected account.

        PROPFIND /remote.php/dav/files/{user}
        """
        body = """<?xml version="1.0" encoding="UTF-8"?>
            <d:propfind xmlns:d="DAV:">
              <d:prop>
                <d:quota-available-bytes/>
                <d:quota-used-bytes/>
              </d:prop>
            </d:propfind>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "PROPFIND",
                self.webdav_base,
                body,
                {"Depth": "0", "Content-Type": "application/xml"},
            )
        )
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

    def _dav_root(self) -> str:
        return f"{self.base_url}/remote.php/dav"

    def _listing(self, url: str, props: str) -> list[dict]:
        """PROPFIND depth 1 on ``url``; one dict per child with the asked props."""
        body = (
            '<?xml version="1.0"?><d:propfind xmlns:d="DAV:" '
            f'xmlns:oc="{_OC}" xmlns:nc="http://nextcloud.org/ns"><d:prop>{props}'
            "</d:prop></d:propfind>"
        )
        root = parse_untrusted_xml(
            self._dav_request(
                "PROPFIND", url, body, {"Depth": "1", "Content-Type": "application/xml"}
            )
        )
        return _responses(root)[1:]

    def list_versions(self, path: str) -> list[dict]:
        """List the stored versions of one file, newest first.

        PROPFIND /remote.php/dav/versions/{user}/versions/{fileid}
        """
        file_id = self.get_properties(path).get("file_id")
        if not file_id:
            raise FileNotFoundError("Configured remote path was not found")
        url = f"{self._dav_root()}/versions/{quote(self.username, safe='')}/versions/{file_id}"
        rows = self._listing(
            url,
            "<d:getlastmodified/><d:getcontentlength/><d:getcontenttype/>"
            "<nc:version-label/><nc:version-author/>",
        )
        for row in rows:
            row["version_id"] = row["href"].rstrip("/").rsplit("/", 1)[-1]
            row["file_id"] = file_id
        return sorted(rows, key=lambda row: int(row["version_id"] or 0), reverse=True)

    def restore_version(self, path: str, version_id: str) -> bool:
        """Restore one stored version of a file; the current content becomes a version.

        MOVE /remote.php/dav/versions/{user}/versions/{fileid}/{version_id}
        """
        file_id = self.get_properties(path).get("file_id")
        if not file_id or not version_id.isdigit():
            raise ValueError("file or version identifier is invalid")
        user = quote(self.username, safe="")
        source = f"{self._dav_root()}/versions/{user}/versions/{file_id}/{version_id}"
        target = f"{self._dav_root()}/versions/{user}/restore/target"
        self._dav_send("MOVE", source, headers={"Destination": target})
        return True

    def search_files(
        self,
        query: str,
        path: str = "",
        mime_prefix: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search files by name (case-insensitive substring) under one folder.

        SEARCH /remote.php/dav

        ``mime_prefix`` narrows to a content type family, for example
        ``image/`` or ``video/``. Returns paths relative to the account root.
        """
        if not query or len(query) > 256:
            raise ValueError("search query is invalid")
        self._get_full_url(path)
        user = quote(self.username, safe="")
        scope = f"/files/{user}/{path.strip('/')}".rstrip("/")
        mime = ""
        if mime_prefix:
            mime = (
                "<d:like><d:prop><d:getcontenttype/></d:prop>"
                f"<d:literal>{escape(mime_prefix)}%</d:literal></d:like>"
            )
        name = (
            "<d:like><d:prop><d:displayname/></d:prop>"
            f"<d:literal>%{escape(query)}%</d:literal></d:like>"
        )
        where = f"<d:and>{name}{mime}</d:and>" if mime else name
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
<d:searchrequest xmlns:d="DAV:" xmlns:oc="{_OC}">
  <d:basicsearch>
    <d:select><d:prop><d:displayname/><d:getcontenttype/><d:getlastmodified/>
      <d:getcontentlength/><oc:fileid/><d:resourcetype/></d:prop></d:select>
    <d:from><d:scope><d:href>{escape(scope)}</d:href><d:depth>infinity</d:depth></d:scope></d:from>
    <d:where>{where}</d:where>
    <d:orderby/>
    <d:limit><d:nresults>{max(1, min(int(limit), 500))}</d:nresults></d:limit>
  </d:basicsearch>
</d:searchrequest>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "SEARCH",
                self._dav_root() + "/",
                body,
                {"Content-Type": "text/xml"},
            )
        )
        prefix = f"/remote.php/dav/files/{user}/"
        rows = _responses(root)
        for row in rows:
            href = row["href"]
            row["path"] = unquote(href.split(prefix, 1)[-1]) if prefix in href else href
        return rows

    def list_trash(self) -> list[dict]:
        """List the items in the trash bin of the connected account.

        PROPFIND /remote.php/dav/trashbin/{user}/trash
        """
        url = f"{self._dav_root()}/trashbin/{quote(self.username, safe='')}/trash"
        rows = self._listing(
            url,
            "<nc:trashbin-filename/><nc:trashbin-original-location/>"
            "<nc:trashbin-deletion-time/><d:getcontentlength/><d:resourcetype/>",
        )
        for row in rows:
            row["trash_id"] = unquote(row["href"].rstrip("/").rsplit("/", 1)[-1])
        return rows

    def restore_trash_item(self, trash_id: str) -> bool:
        """Restore one trash-bin item to its original location.

        MOVE /remote.php/dav/trashbin/{user}/trash/{trash_id}
        """
        if not trash_id or "/" in trash_id or ".." in trash_id:
            raise ValueError("trash identifier is invalid")
        user = quote(self.username, safe="")
        source = f"{self._dav_root()}/trashbin/{user}/trash/{quote(trash_id, safe='')}"
        target = (
            f"{self._dav_root()}/trashbin/{user}/restore/{quote(trash_id, safe='')}"
        )
        self._dav_send("MOVE", source, headers={"Destination": target})
        return True


def _responses(root) -> list[dict]:
    """Flatten a DAV multistatus into one dict per response (local prop names)."""
    rows: list[dict] = []
    for response in root.findall("{DAV:}response"):
        row: dict = {"href": response.findtext("{DAV:}href", default="")}
        for propstat in response.findall("{DAV:}propstat"):
            status = propstat.findtext("{DAV:}status", default="")
            if " 200 " not in f"{status} ":
                continue
            prop = propstat.find("{DAV:}prop")
            if prop is None:
                continue
            for child in prop:
                key = child.tag.rsplit("}", 1)[-1].replace("-", "_")
                if key == "resourcetype":
                    row["is_folder"] = child.find("{DAV:}collection") is not None
                else:
                    row[key] = (child.text or "").strip()
        rows.append(row)
    return rows
