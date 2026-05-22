import os
import uuid
import xml.etree.ElementTree as ET
from nextcloud_agent.api.api_client_base import BaseApiClient


class Api(BaseApiClient):
    def list_address_books(self) -> list[dict]:
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
                        "url": self._get_absolute_url(href),  # type: ignore[arg-type]
                    }
                )
        return books

    def list_contacts(self, address_book_url: str) -> list[dict]:
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
            if href is None:
                continue
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
        self, address_book_url: str, vcard_data: str, filename: str | None = None
    ) -> bool:
        if not filename:
            filename = f"{uuid.uuid4()}.vcf"
        url = f"{address_book_url.rstrip('/')}/{filename}"
        headers = {"Content-Type": "text/vcard; charset=utf-8"}
        response = self._session.put(url, data=vcard_data, headers=headers)
        response.raise_for_status()
        return True
