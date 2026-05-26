import os
import uuid
import xml.etree.ElementTree as ET

from nextcloud_agent.api.api_client_base import BaseApiClient


class Api(BaseApiClient):
    def list_calendars(self) -> list[dict]:
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

        for r in root.findall("d:response", ns):
            href = r.findtext("d:href", namespaces=ns)  # type: ignore[attr-defined]
            prop = r.find("d:propstat/d:prop", ns)  # type: ignore[attr-defined]
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
                        "url": self._get_absolute_url(href),  # type: ignore[arg-type]
                    }
                )
        return calendars

    def list_events(self, calendar_url: str) -> list[dict]:
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
            if href is None:
                continue
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
        self, calendar_url: str, event_data: str, filename: str | None = None
    ) -> bool:
        """Create an event with ICS data."""
        if not filename:
            filename = f"{uuid.uuid4()}.ics"

        url = f"{calendar_url.rstrip('/')}/{filename}"
        headers = {"Content-Type": "text/calendar; charset=utf-8"}

        response = self._session.put(url, data=event_data, headers=headers)
        response.raise_for_status()
        return True

    def list_calendar_events(self, calendar_url: str) -> list[dict]:
        """Alias for list_events to support MCP server action."""
        return self.list_events(calendar_url)

    def create_calendar_event(
        self, calendar_url: str, event_data: str, filename: str | None = None
    ) -> bool:
        """Alias for create_event to support MCP server action."""
        return self.create_event(calendar_url, event_data, filename)
