import logging
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)


class BaseApiClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        verify: bool = True,
    ):
        """Initialize the Nextcloud WebDAV API client."""
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.verify = verify
        self._session = requests.Session()
        self._session.auth = (username, password)
        self._session.verify = verify
        self._session.headers.update({"OCS-APIRequest": "true"})

        self.webdav_base = (
            f"{self.base_url}/remote.php/dav/files/{quote(self.username)}"
        )
        self.caldav_base = (
            f"{self.base_url}/remote.php/dav/calendars/{quote(self.username)}"
        )
        self.carddav_base = (
            f"{self.base_url}/remote.php/dav/addressbooks/users/{quote(self.username)}"
        )
        self.ocs_base = f"{self.base_url}/ocs/v2.php"

    def _get_full_url(self, path: str) -> str:
        """Helper to construct full WebDAV URL for a path."""
        clean_path = path.strip("/")
        if not clean_path:
            return self.webdav_base + "/"
        return f"{self.webdav_base}/{quote(clean_path)}"

    def _parse_propfind_response(self, xml_content: str) -> list[dict]:
        """Parse PROPFIND XML response into a list of file dictionaries."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return []

        ns = {
            "d": "DAV:",
            "oc": "http://owncloud.org/ns",
            "nc": "http://nextcloud.org/ns",
        }

        files = []
        for response in root.findall("d:response", ns):
            href = response.find("d:href", ns).text  # type: ignore[union-attr]
            propstat = response.find("d:propstat", ns)
            if propstat is None:
                continue

            prop = propstat.find("d:prop", ns)
            if prop is None:
                continue

            file_data = {
                "href": href,
                "name": os.path.basename(href.rstrip("/")),  # type: ignore[union-attr]
                "is_folder": href.endswith("/"),  # type: ignore[union-attr]
                "last_modified": prop.findtext(
                    "d:getlastmodified", default="", namespaces=ns
                ),
                "etag": prop.findtext("d:getetag", default="", namespaces=ns),
                "content_type": prop.findtext(
                    "d:getcontenttype", default="", namespaces=ns
                ),
                "content_length": prop.findtext(
                    "d:getcontentlength", default="0", namespaces=ns
                ),
                "permissions": prop.findtext(
                    "oc:permissions", default="", namespaces=ns
                ),
                "favorite": prop.findtext("oc:favorite", default="0", namespaces=ns),
                "file_id": prop.findtext("oc:fileid", default="", namespaces=ns),
            }

            resourcetype = prop.find("d:resourcetype", ns)
            if (
                resourcetype is not None
                and resourcetype.find("d:collection", ns) is not None
            ):
                file_data["is_folder"] = True

            files.append(file_data)

        return files

    def _get_absolute_url(self, href: str) -> str:
        """Convert a WebDAV href to a full URL."""
        if href.startswith("http"):
            return href
        return urljoin(self.base_url, href)

    def ocs_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make a request to the OCS API."""
        url = f"{self.ocs_base}/{endpoint.lstrip('/')}"
        kwargs.setdefault("headers", {})
        kwargs["headers"].update({"OCS-APIRequest": "true"})
        kwargs.setdefault("params", {})
        kwargs["params"].update({"format": "json"})

        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()

        data = response.json()
        meta = data.get("ocs", {}).get("meta", {})
        if meta.get("status") != "ok" and meta.get("statuscode") not in [100, 200]:
            raise Exception(
                f"OCS Error: {meta.get('message', 'Unknown error')} (Code: {meta.get('statuscode')})"
            )

        return data.get("ocs", {}).get("data", {})
