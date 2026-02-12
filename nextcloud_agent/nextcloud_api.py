#!/usr/bin/env python
# coding: utf-8

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Union
from urllib.parse import urljoin, quote
import os
import logging
import urllib3
import uuid

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class NextcloudAPI:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        verify: bool = True,
    ):
        """
        Initialize the Nextcloud WebDAV API client.

        :param base_url: The base URL of the Nextcloud instance (e.g., https://cloud.example.com).
                         The client will automatically append /remote.php/dav/files/{username}/ if not present in requests usually.
                         However, for flexibility, we'll construct full paths.
                         Standard WebDAV files endpoint: /remote.php/dav/files/{username}/
        :param username: Username for authentication.
        :param password: Password (app password recommended) for authentication.
        :param verify: Whether to verify SSL certificates.
        """
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

    def _parse_propfind_response(self, xml_content: str) -> List[Dict]:
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
            href = response.find("d:href", ns).text
            propstat = response.find("d:propstat", ns)
            if propstat is None:
                continue

            prop = propstat.find("d:prop", ns)
            if prop is None:
                continue

            file_data = {
                "href": href,
                "name": os.path.basename(href.rstrip("/")),
                "is_folder": href.endswith("/"),
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

    def list_contents(self, path: str = "") -> List[Dict]:
        """
        List files and folders in a directory using PROPFIND.
        """
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

        files = self._parse_propfind_response(response.content)

        filtered_files = []
        for f in files:
            if f["name"] and f["name"] != os.path.basename(path.strip("/")):
                filtered_files.append(f)
            elif not path.strip("/") and f["name"]:
                filtered_files.append(f)

        return files

    def read_file(self, path: str) -> bytes:
        """Download a file."""
        url = self._get_full_url(path)
        response = self._session.get(url)
        response.raise_for_status()
        return response.content

    def write_file(
        self, path: str, content: Union[str, bytes], overwrite: bool = True
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

    def delete_resource(self, path: str) -> bool:
        """Delete a file or directory."""
        url = self._get_full_url(path)
        response = self._session.delete(url)
        response.raise_for_status()
        return True

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

    def get_user_quota(self) -> Dict:
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

    def _get_absolute_url(self, href: str) -> str:
        """Convert a WebDAV href to a full URL."""
        if href.startswith("http"):
            return href
        return urljoin(self.base_url, href)

    def ocs_request(self, method: str, endpoint: str, **kwargs) -> Dict:
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

    def list_shares(self) -> List[Dict] | Dict:
        """List all shares."""
        return self.ocs_request("GET", "apps/files_sharing/api/v1/shares")

    def create_share(
        self, path: str, share_type: int = 3, permissions: int = 1
    ) -> Dict:
        """
        Create a share.
        share_type: 0=User, 1=Group, 3=Public Link, 4=Email
        permissions: 1=Read, ...
        """
        data = {"path": path, "shareType": share_type, "permissions": permissions}
        return self.ocs_request("POST", "apps/files_sharing/api/v1/shares", data=data)

    def delete_share(self, share_id: str) -> bool:
        """Delete a share."""
        self.ocs_request("DELETE", f"apps/files_sharing/api/v1/shares/{share_id}")
        return True

    def get_user_info(self) -> Dict:
        """Get current user info."""
        return self.ocs_request("GET", "cloud/user")

    def list_calendars(self) -> List[Dict]:
        """List available calendars."""
        body = """<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
          <d:prop>
            <d:displayname />
            <c:calendar-description />
            <d:resourcetype />
          </d:prop>
        </d:propfind>"""

        response = self._session.request(
            "PROPFIND",
            self.caldav_base + "/",
            data=body,
            headers={"Depth": "1", "Content-Type": "application/xml"},
        )
        response.raise_for_status()

        calendars = []
        root = ET.fromstring(response.content)
        ns = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}

        for response in root.findall("d:response", ns):
            href = response.findtext("d:href", namespaces=ns)
            prop = response.find("d:propstat/d:prop", ns)
            if prop is None:
                continue

            resourcetype = prop.find("d:resourcetype", ns)
            if (
                resourcetype is not None
                and resourcetype.find("c:calendar", ns) is not None
            ):
                calendars.append(
                    {
                        "href": href,
                        "displayname": prop.findtext(
                            "d:displayname", default="Unnamed", namespaces=ns
                        ),
                        "url": self._get_absolute_url(href),
                    }
                )
        return calendars

    def list_events(self, calendar_url: str) -> List[Dict]:
        """List events in a calendar (returns basic info)."""
        response = self._session.request(
            "PROPFIND", calendar_url, headers={"Depth": "1"}
        )
        response.raise_for_status()

        events = []
        root = ET.fromstring(response.content)
        ns = {"d": "DAV:"}

        for resp in root.findall("d:response", ns):
            href = resp.findtext("d:href", namespaces=ns)
            if href.endswith(".ics"):
                events.append(
                    {
                        "href": href,
                        "name": os.path.basename(href),
                        "url": self._get_absolute_url(href),
                    }
                )
        return events

    def create_event(
        self, calendar_url: str, event_data: str, filename: str = None
    ) -> bool:
        """Create an event with ICS data."""
        if not filename:
            filename = f"{uuid.uuid4()}.ics"

        url = f"{calendar_url.rstrip('/')}/{filename}"
        headers = {"Content-Type": "text/calendar; charset=utf-8"}

        response = self._session.put(url, data=event_data, headers=headers)
        response.raise_for_status()
        return True

    def list_address_books(self) -> List[Dict]:
        """List address books."""
        body = """<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
          <d:prop>
            <d:displayname />
            <d:resourcetype />
          </d:prop>
        </d:propfind>"""

        response = self._session.request(
            "PROPFIND",
            self.carddav_base + "/",
            data=body,
            headers={"Depth": "1", "Content-Type": "application/xml"},
        )
        response.raise_for_status()

        books = []
        root = ET.fromstring(response.content)
        ns = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:carddav"}

        for resp in root.findall("d:response", ns):
            href = resp.findtext("d:href", namespaces=ns)
            prop = resp.find("d:propstat/d:prop", ns)
            if prop is None:
                continue

            resourcetype = prop.find("d:resourcetype", ns)
            if (
                resourcetype is not None
                and resourcetype.find("c:addressbook", ns) is not None
            ):
                books.append(
                    {
                        "href": href,
                        "displayname": prop.findtext("d:displayname", namespaces=ns),
                        "url": self._get_absolute_url(href),
                    }
                )
        return books

    def list_contacts(self, address_book_url: str) -> List[Dict]:
        """List contacts in address book."""
        response = self._session.request(
            "PROPFIND", address_book_url, headers={"Depth": "1"}
        )
        response.raise_for_status()
        contacts = []
        root = ET.fromstring(response.content)
        ns = {"d": "DAV:"}
        for resp in root.findall("d:response", ns):
            href = resp.findtext("d:href", namespaces=ns)
            if href.endswith(".vcf"):
                contacts.append(
                    {
                        "href": href,
                        "name": os.path.basename(href),
                        "url": self._get_absolute_url(href),
                    }
                )
        return contacts

    def create_contact(
        self, address_book_url: str, vcard_data: str, filename: str = None
    ) -> bool:
        if not filename:
            filename = f"{uuid.uuid4()}.vcf"
        url = f"{address_book_url.rstrip('/')}/{filename}"
        headers = {"Content-Type": "text/vcard; charset=utf-8"}
        response = self._session.put(url, data=vcard_data, headers=headers)
        response.raise_for_status()
        return True
