import os
import uuid
from urllib.parse import quote

from nextcloud_agent.api.api_client_base import BaseApiClient
from nextcloud_agent.api.xml_security import parse_untrusted_xml


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
            stream=True,
            timeout=(10, 30),
        )

        books = []
        root = parse_untrusted_xml(self._read_xml_response(response))
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
            "PROPFIND",
            address_book_url,
            headers={"Depth": "1"},
            stream=True,
            timeout=(10, 30),
        )
        contacts = []
        root = parse_untrusted_xml(self._read_xml_response(response))
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
        if len(filename) > 255 or any(ord(character) < 32 for character in filename):
            raise ValueError("contact filename is invalid")
        url = f"{self._get_absolute_url(address_book_url).rstrip('/')}/{quote(filename, safe='')}"
        headers = {"Content-Type": "text/vcard; charset=utf-8"}
        response = self._session.put(url, data=vcard_data, headers=headers)
        response.raise_for_status()
        return True
