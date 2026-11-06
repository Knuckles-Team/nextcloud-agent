"""Address books and contacts over CardDAV (CONN-NEXTCLOUD-AGENT-R008)."""

import uuid

import vobject

from nextcloud_agent.api.api_client_base import BaseApiClient
from nextcloud_agent.api.xml_security import parse_untrusted_xml

_NS = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:carddav"}


def _card_fields(vcf: str) -> dict:
    try:
        card = vobject.readOne(vcf)
    except Exception:  # noqa: BLE001 - a malformed card is reported, not raised
        return {"parse_error": True}

    def values(name: str) -> list[str]:
        return [str(item.value) for item in card.contents.get(name, [])]

    org = card.contents.get("org")
    return {
        "uid": values("uid")[0] if values("uid") else "",
        "full_name": values("fn")[0] if values("fn") else "",
        "emails": values("email"),
        "phones": values("tel"),
        "organization": ";".join(org[0].value) if org else "",
        "title": values("title")[0] if values("title") else "",
        "note": values("note")[0] if values("note") else "",
    }


def _build_card(
    uid: str,
    full_name: str,
    emails: list[str] | None,
    phones: list[str] | None,
    organization: str | None,
    title: str | None,
    note: str | None,
):
    card = vobject.vCard()
    card.add("uid").value = uid
    card.add("fn").value = full_name
    parts = full_name.split(" ", 1)
    card.add("n").value = vobject.vcard.Name(
        family=parts[1] if len(parts) > 1 else "", given=parts[0]
    )
    for email in emails or []:
        card.add("email").value = email
    for phone in phones or []:
        card.add("tel").value = phone
    if organization:
        card.add("org").value = [organization]
    if title:
        card.add("title").value = title
    if note:
        card.add("note").value = note
    return card.serialize()


class NextcloudContacts(BaseApiClient):
    """Address books and contacts (vCard) over CardDAV."""

    def list_address_books(self) -> list[dict]:
        """List the account's address books.

        PROPFIND /remote.php/dav/addressbooks/users/{user}
        """
        body = """<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
          <d:prop><d:displayname/><d:resourcetype/></d:prop></d:propfind>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "PROPFIND",
                self.carddav_base + "/",
                body,
                {"Depth": "1", "Content-Type": "application/xml"},
            )
        )
        books = []
        for resp in root.findall("d:response", _NS):
            href = resp.findtext("d:href", namespaces=_NS)
            prop = resp.find("d:propstat/d:prop", _NS)
            if prop is None or not href:
                continue
            kind = prop.find("d:resourcetype", _NS)
            if kind is not None and kind.find("c:addressbook", _NS) is not None:
                books.append(
                    {
                        "href": href,
                        "displayname": prop.findtext("d:displayname", namespaces=_NS),
                        "url": self._get_absolute_url(href),
                    }
                )
        return books

    def list_contacts(self, address_book_url: str) -> list[dict]:
        """List the contacts of one address book with name, emails, phones and organization.

        REPORT /remote.php/dav/addressbooks/users/{user}/{book}
        """
        body = """<?xml version="1.0" encoding="utf-8"?>
<c:addressbook-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
  <d:prop><d:getetag/><c:address-data/></d:prop>
</c:addressbook-query>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "REPORT",
                address_book_url,
                body,
                {"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
            )
        )
        contacts = []
        for resp in root.findall("d:response", _NS):
            href = resp.findtext("d:href", namespaces=_NS)
            data = resp.findtext("d:propstat/d:prop/c:address-data", namespaces=_NS)
            if not href or not data:
                continue
            row = _card_fields(data)
            row.update(
                {
                    "href": href,
                    "url": self._get_absolute_url(href),
                    "etag": resp.findtext(
                        "d:propstat/d:prop/d:getetag", default="", namespaces=_NS
                    ),
                }
            )
            contacts.append(row)
        return contacts

    def get_contact(self, contact_url: str) -> dict:
        """Read one contact: parsed fields, etag and the raw vCard.

        GET /remote.php/dav/addressbooks/users/{user}/{book}/{contact}.vcf
        """
        response = self._session.request(
            "GET", self._get_absolute_url(contact_url), timeout=(10, 30)
        )
        response.raise_for_status()
        row = _card_fields(response.text)
        row.update(
            {
                "url": contact_url,
                "etag": response.headers.get("ETag", ""),
                "vcard": response.text,
            }
        )
        return row

    def create_contact(
        self,
        address_book_url: str,
        full_name: str,
        emails: list[str] | None = None,
        phones: list[str] | None = None,
        organization: str | None = None,
        title: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Create one contact in an address book.

        PUT /remote.php/dav/addressbooks/users/{user}/{book}/{uid}.vcf
        """
        uid = str(uuid.uuid4())
        vcf = _build_card(uid, full_name, emails, phones, organization, title, note)
        url = f"{self._get_absolute_url(address_book_url).rstrip('/')}/{uid}.vcf"
        response = self._dav_send(
            "PUT",
            url,
            vcf,
            {"Content-Type": "text/vcard; charset=utf-8", "If-None-Match": "*"},
        )
        return {"url": url, "uid": uid, "etag": response.headers.get("ETag", "")}

    def update_contact(
        self,
        contact_url: str,
        full_name: str | None = None,
        emails: list[str] | None = None,
        phones: list[str] | None = None,
        organization: str | None = None,
        title: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Change one contact; omitted fields keep their value. Guarded by the etag.

        PUT /remote.php/dav/addressbooks/users/{user}/{book}/{contact}.vcf
        """
        current = self.get_contact(contact_url)
        vcf = _build_card(
            current.get("uid") or str(uuid.uuid4()),
            full_name if full_name is not None else current.get("full_name", ""),
            emails if emails is not None else current.get("emails"),
            phones if phones is not None else current.get("phones"),
            organization if organization is not None else current.get("organization"),
            title if title is not None else current.get("title"),
            note if note is not None else current.get("note"),
        )
        headers = {"Content-Type": "text/vcard; charset=utf-8"}
        if current.get("etag"):
            headers["If-Match"] = current["etag"]
        response = self._dav_send("PUT", contact_url, vcf, headers)
        return {"url": contact_url, "etag": response.headers.get("ETag", "")}

    def delete_contact(self, contact_url: str) -> bool:
        """Delete one contact.

        DELETE /remote.php/dav/addressbooks/users/{user}/{book}/{contact}.vcf
        """
        if not self._get_absolute_url(contact_url).startswith(self.carddav_base + "/"):
            raise ValueError("contact URL is outside the account's address books")
        self._dav_send("DELETE", contact_url)
        return True
