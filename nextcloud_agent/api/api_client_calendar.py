"""Calendars, events and tasks over CalDAV (CONN-NEXTCLOUD-AGENT-R006, R007)."""

import uuid
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

from dateutil import parser as date_parser
from icalendar import Calendar, Event, Todo, vCalAddress, vText

from nextcloud_agent.api.api_client_base import BaseApiClient
from nextcloud_agent.api.xml_security import parse_untrusted_xml

_NS = {
    "d": "DAV:",
    "c": "urn:ietf:params:xml:ns:caldav",
    "cs": "http://calendarserver.org/ns/",
}
_C = "urn:ietf:params:xml:ns:caldav"


def _when(value: str, all_day: bool = False) -> date | datetime:
    """An ISO date or date-time; a naive time is read as UTC."""
    parsed = date_parser.isoparse(value)
    if all_day:
        return parsed.date()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _stamp(value: str) -> str:
    """An ISO date-time as a CalDAV UTC time-range stamp."""
    moment = _when(value)
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _text(component, key: str) -> str:
    value = component.get(key)
    return str(value) if value is not None else ""


def _time(component, key: str) -> str:
    value = component.get(key)
    if value is None:
        return ""
    return value.dt.isoformat()


def _summary(ics: str, kind: str) -> dict:
    """The fields an agent needs from one calendar object."""
    try:
        cal = Calendar.from_ical(ics)
    except ValueError:
        return {"parse_error": True}
    for component in cal.walk(kind):
        row = {
            "uid": _text(component, "UID"),
            "summary": _text(component, "SUMMARY"),
            "description": _text(component, "DESCRIPTION"),
            "location": _text(component, "LOCATION"),
            "status": _text(component, "STATUS"),
        }
        if kind == "VEVENT":
            row["start"] = _time(component, "DTSTART")
            row["end"] = _time(component, "DTEND")
            attendees = component.get("ATTENDEE") or []
            if not isinstance(attendees, list):
                attendees = [attendees]
            row["attendees"] = [str(a).removeprefix("mailto:") for a in attendees]
        else:
            row["due"] = _time(component, "DUE")
            row["completed"] = _time(component, "COMPLETED")
            row["percent_complete"] = int(component.get("PERCENT-COMPLETE", 0) or 0)
            row["priority"] = int(component.get("PRIORITY", 0) or 0)
        return row
    return {}


class NextcloudCalendar(BaseApiClient):
    """Calendars, events (with draft invitations) and tasks (VTODO) over CalDAV."""

    def list_calendars(self) -> list[dict]:
        """List the account's calendars and the components (VEVENT, VTODO) each holds.

        PROPFIND /remote.php/dav/calendars/{user}
        """
        body = f"""<d:propfind xmlns:d="DAV:" xmlns:c="{_C}" xmlns:cs="http://calendarserver.org/ns/">
          <d:prop><d:displayname/><c:calendar-description/><d:resourcetype/>
          <c:supported-calendar-component-set/><cs:getctag/></d:prop></d:propfind>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "PROPFIND",
                self.caldav_base + "/",
                body,
                {"Depth": "1", "Content-Type": "application/xml"},
            )
        )
        calendars = []
        for r in root.findall("d:response", _NS):
            href = r.findtext("d:href", namespaces=_NS)
            prop = r.find("d:propstat/d:prop", _NS)
            if prop is None or not href:
                continue
            kind = prop.find("d:resourcetype", _NS)
            if kind is None or kind.find("c:calendar", _NS) is None:
                continue
            comps = [
                c.get("name")
                for c in prop.findall("c:supported-calendar-component-set/c:comp", _NS)
            ]
            calendars.append(
                {
                    "href": href,
                    "displayname": prop.findtext(
                        "d:displayname", default="Unnamed", namespaces=_NS
                    ),
                    "description": prop.findtext(
                        "c:calendar-description", default="", namespaces=_NS
                    ),
                    "components": comps,
                    "url": self._get_absolute_url(href),
                }
            )
        return calendars

    def create_calendar(
        self, display_name: str, slug: str | None = None, with_tasks: bool = True
    ) -> dict:
        """Create one calendar that holds events and, by default, tasks.

        MKCALENDAR /remote.php/dav/calendars/{user}/{slug}
        """
        slug = slug or "".join(
            ch if ch.isalnum() else "-" for ch in display_name.lower()
        )
        if not slug or len(slug) > 128 or "/" in slug:
            raise ValueError("calendar slug is invalid")
        comps = '<c:comp name="VEVENT"/>' + (
            '<c:comp name="VTODO"/>' if with_tasks else ""
        )
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
<c:mkcalendar xmlns:d="DAV:" xmlns:c="{_C}"><d:set><d:prop>
<d:displayname>{vText(display_name).to_ical().decode()}</d:displayname>
<c:supported-calendar-component-set>{comps}</c:supported-calendar-component-set>
</d:prop></d:set></c:mkcalendar>"""
        url = f"{self.caldav_base}/{quote(slug, safe='')}/"
        self._dav_send("MKCALENDAR", url, body, {"Content-Type": "application/xml"})
        return {"url": url, "slug": slug, "displayname": display_name}

    def delete_calendar(self, calendar_url: str) -> bool:
        """Delete one calendar and everything in it (it moves to the calendar trash).

        DELETE /remote.php/dav/calendars/{user}/{slug}
        """
        if not self._get_absolute_url(calendar_url).startswith(self.caldav_base + "/"):
            raise ValueError("calendar URL is outside the account's calendars")
        self._dav_send("DELETE", calendar_url)
        return True

    def _query(
        self, calendar_url: str, kind: str, start: str | None, end: str | None
    ) -> list[dict]:
        time_range = ""
        if start or end:
            attrs = (f' start="{_stamp(start)}"' if start else "") + (
                f' end="{_stamp(end)}"' if end else ""
            )
            time_range = f"<c:time-range{attrs}/>"
        body = f"""<?xml version="1.0" encoding="utf-8"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="{_C}">
  <d:prop><d:getetag/><c:calendar-data/></d:prop>
  <c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="{kind}">{time_range}
  </c:comp-filter></c:comp-filter></c:filter>
</c:calendar-query>"""
        root = parse_untrusted_xml(
            self._dav_request(
                "REPORT",
                calendar_url,
                body,
                {"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
            )
        )
        rows = []
        for r in root.findall("d:response", _NS):
            href = r.findtext("d:href", namespaces=_NS)
            data = r.findtext("d:propstat/d:prop/c:calendar-data", namespaces=_NS)
            if not href or not data:
                continue
            row = _summary(data, kind)
            row.update(
                {
                    "href": href,
                    "url": self._get_absolute_url(href),
                    "etag": r.findtext(
                        "d:propstat/d:prop/d:getetag", default="", namespaces=_NS
                    ),
                }
            )
            rows.append(row)
        return rows

    def list_events(
        self, calendar_url: str, start: str | None = None, end: str | None = None
    ) -> list[dict]:
        """List the events of one calendar, optionally within an ISO time range.

        REPORT /remote.php/dav/calendars/{user}/{calendar}
        """
        return self._query(calendar_url, "VEVENT", start, end)

    def get_calendar_object(self, object_url: str) -> dict:
        """Read one event or task: its parsed fields, etag and raw iCalendar text.

        GET /remote.php/dav/calendars/{user}/{calendar}/{object}.ics
        """
        response = self._session.request(
            "GET", self._get_absolute_url(object_url), timeout=(10, 30)
        )
        response.raise_for_status()
        ics = response.text
        kind = "VTODO" if "BEGIN:VTODO" in ics else "VEVENT"
        row = _summary(ics, kind)
        row.update(
            {
                "url": object_url,
                "etag": response.headers.get("ETag", ""),
                "ics": ics,
                "kind": kind,
            }
        )
        return row

    def _organizer(self) -> vCalAddress:
        info = self.ocs_request("GET", "cloud/user") or {}
        email = info.get("email") or f"{self.username}@localhost"
        organizer = vCalAddress(f"mailto:{email}")
        organizer.params["CN"] = vText(info.get("displayname") or self.username)
        return organizer

    @staticmethod
    def _set_attendees(event, attendees: list[str], send_invitations: bool) -> None:
        if "ATTENDEE" in event:
            del event["ATTENDEE"]
        for email in attendees:
            attendee = vCalAddress(f"mailto:{email}")
            attendee.params["ROLE"] = vText("REQ-PARTICIPANT")
            attendee.params["PARTSTAT"] = vText("NEEDS-ACTION")
            attendee.params["RSVP"] = vText("TRUE")
            if not send_invitations:
                # The server schedules nothing for this attendee: the invite stays a draft.
                attendee.params["SCHEDULE-AGENT"] = vText("CLIENT")
            event.add("ATTENDEE", attendee, encode=0)

    def _put(self, url: str, cal: Calendar, etag: str | None) -> dict:
        headers = {"Content-Type": "text/calendar; charset=utf-8"}
        headers["If-Match" if etag else "If-None-Match"] = etag or "*"
        response = self._dav_send("PUT", url, cal.to_ical(), headers)
        return {"url": url, "etag": response.headers.get("ETag", "")}

    def create_event(
        self,
        calendar_url: str,
        summary: str,
        start: str,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        send_invitations: bool = False,
        all_day: bool = False,
    ) -> dict:
        """Create one event. Attendees get a draft invitation unless ``send_invitations``.

        PUT /remote.php/dav/calendars/{user}/{calendar}/{uid}.ics

        ``start``/``end`` are ISO 8601 (a naive time is UTC); ``end``
        defaults to one hour (one day when ``all_day``) after ``start``.
        With ``send_invitations`` false the attendees are marked
        SCHEDULE-AGENT=CLIENT, so the server sends no invitation email.
        """
        uid = str(uuid.uuid4())
        begin = _when(start, all_day)
        finish = (
            _when(end, all_day)
            if end
            else begin + (timedelta(days=1) if all_day else timedelta(hours=1))
        )
        cal = Calendar()
        cal.add("PRODID", "-//nextcloud-agent//EN")
        cal.add("VERSION", "2.0")
        event = Event()
        event.add("UID", uid)
        event.add("DTSTAMP", datetime.now(UTC))
        event.add("SUMMARY", summary)
        event.add("DTSTART", begin)
        event.add("DTEND", finish)
        if description:
            event.add("DESCRIPTION", description)
        if location:
            event.add("LOCATION", location)
        if attendees:
            event.add("ORGANIZER", self._organizer())
            self._set_attendees(event, attendees, send_invitations)
        cal.add_component(event)
        url = f"{self._get_absolute_url(calendar_url).rstrip('/')}/{uid}.ics"
        result = self._put(url, cal, None)
        result["uid"] = uid
        return result

    def update_event(
        self,
        event_url: str,
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        send_invitations: bool = False,
    ) -> dict:
        """Change one event's fields; the write is guarded by the event's etag.

        PUT /remote.php/dav/calendars/{user}/{calendar}/{object}.ics
        """
        current = self.get_calendar_object(event_url)
        cal = Calendar.from_ical(current["ics"])
        for event in cal.walk("VEVENT"):
            for key, value in (
                ("SUMMARY", summary),
                ("DESCRIPTION", description),
                ("LOCATION", location),
            ):
                if value is not None:
                    event.pop(key, None)
                    event.add(key, value)
            if start is not None:
                event.pop("DTSTART", None)
                event.add("DTSTART", _when(start))
            if end is not None:
                event.pop("DTEND", None)
                event.add("DTEND", _when(end))
            if attendees is not None:
                if attendees and "ORGANIZER" not in event:
                    event.add("ORGANIZER", self._organizer())
                self._set_attendees(event, attendees, send_invitations)
            event.pop("DTSTAMP", None)
            event.add("DTSTAMP", datetime.now(UTC))
            seq = int(event.get("SEQUENCE", 0) or 0) + 1
            event.pop("SEQUENCE", None)
            event.add("SEQUENCE", seq)
        return self._put(
            self._get_absolute_url(event_url), cal, current["etag"] or None
        )

    def delete_calendar_object(self, object_url: str) -> bool:
        """Delete one event or task.

        DELETE /remote.php/dav/calendars/{user}/{calendar}/{object}.ics
        """
        if not self._get_absolute_url(object_url).startswith(self.caldav_base + "/"):
            raise ValueError("object URL is outside the account's calendars")
        self._dav_send("DELETE", object_url)
        return True

    def list_tasks(
        self, calendar_url: str, include_completed: bool = False
    ) -> list[dict]:
        """List the tasks (VTODO) of one task-capable calendar.

        REPORT /remote.php/dav/calendars/{user}/{calendar}
        """
        rows = self._query(calendar_url, "VTODO", None, None)
        if include_completed:
            return rows
        return [
            r for r in rows if not r.get("completed") and r.get("status") != "COMPLETED"
        ]

    def create_task(
        self,
        calendar_url: str,
        summary: str,
        due: str | None = None,
        description: str | None = None,
        priority: int = 0,
    ) -> dict:
        """Create one task (VTODO) in a task-capable calendar (the Tasks app shows it).

        PUT /remote.php/dav/calendars/{user}/{calendar}/{uid}.ics

        ``priority`` follows iCalendar: 1 highest, 9 lowest, 0 undefined.
        """
        uid = str(uuid.uuid4())
        cal = Calendar()
        cal.add("PRODID", "-//nextcloud-agent//EN")
        cal.add("VERSION", "2.0")
        todo = Todo()
        todo.add("UID", uid)
        todo.add("DTSTAMP", datetime.now(UTC))
        todo.add("SUMMARY", summary)
        todo.add("STATUS", "NEEDS-ACTION")
        if due:
            todo.add("DUE", _when(due))
        if description:
            todo.add("DESCRIPTION", description)
        if priority:
            todo.add("PRIORITY", int(priority))
        cal.add_component(todo)
        url = f"{self._get_absolute_url(calendar_url).rstrip('/')}/{uid}.ics"
        result = self._put(url, cal, None)
        result["uid"] = uid
        return result

    def update_task(
        self,
        task_url: str,
        summary: str | None = None,
        due: str | None = None,
        description: str | None = None,
        completed: bool | None = None,
        percent_complete: int | None = None,
    ) -> dict:
        """Change one task, or mark it complete or open again; guarded by its etag.

        PUT /remote.php/dav/calendars/{user}/{calendar}/{object}.ics
        """
        current = self.get_calendar_object(task_url)
        cal = Calendar.from_ical(current["ics"])
        for todo in cal.walk("VTODO"):
            for key, value in (("SUMMARY", summary), ("DESCRIPTION", description)):
                if value is not None:
                    todo.pop(key, None)
                    todo.add(key, value)
            if due is not None:
                todo.pop("DUE", None)
                todo.add("DUE", _when(due))
            if percent_complete is not None:
                todo.pop("PERCENT-COMPLETE", None)
                todo.add("PERCENT-COMPLETE", int(percent_complete))
            if completed is not None:
                for key in ("STATUS", "COMPLETED"):
                    todo.pop(key, None)
                if completed:
                    todo.add("STATUS", "COMPLETED")
                    todo.add("COMPLETED", datetime.now(UTC))
                    todo.pop("PERCENT-COMPLETE", None)
                    todo.add("PERCENT-COMPLETE", 100)
                else:
                    todo.add("STATUS", "NEEDS-ACTION")
            todo.pop("DTSTAMP", None)
            todo.add("DTSTAMP", datetime.now(UTC))
        return self._put(self._get_absolute_url(task_url), cal, current["etag"] or None)
