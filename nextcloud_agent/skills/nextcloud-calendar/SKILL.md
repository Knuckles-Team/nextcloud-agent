---
name: nextcloud-calendar
skill_type: skill
description: >-
  Run Nextcloud calendars over CalDAV: list calendars, create or delete a calendar, list events in a time range, read, create, update and delete events, with attendee invitations kept as drafts unless sending is explicitly requested. Use when the agent must schedule, reschedule or cancel a meeting or check availability. Do NOT use for to-dos (nextcloud-tasks) or contacts (nextcloud-contacts).
license: MIT
tags: [nextcloud, calendar, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Calendar

Run Nextcloud calendars over CalDAV: list calendars, create or delete a calendar, list events in a time range, read, create, update and delete events, with attendee invitations kept as drafts unless sending is explicitly requested.

## When to use
- Check a slot: `nextcloud_list_events(calendar_url, start, end)` with ISO times.
- Book a meeting: `nextcloud_create_event(calendar_url, summary, start, end, attendees=[...])`.
- Move it: `nextcloud_update_event(event_url, start=..., end=...)`; cancel: `nextcloud_delete_calendar_object(event_url)`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_calendars` |  | List the account's calendars and the components (VEVENT, VTODO) each holds. |
| `nextcloud_create_calendar` | display_name, slug?, with_tasks? | Create one calendar that holds events and, by default, tasks. |
| `nextcloud_delete_calendar` | calendar_url | Delete one calendar and everything in it (it moves to the calendar trash). |
| `nextcloud_list_events` | calendar_url, start?, end? | List the events of one calendar, optionally within an ISO time range. |
| `nextcloud_get_calendar_object` | object_url | Read one event or task: its parsed fields, etag and raw iCalendar text. |
| `nextcloud_create_event` | calendar_url, summary, start, end?, description?, location?, attendees?, send_invitations?, all_day? | Create one event. Attendees get a draft invitation unless ``send_invitations``. |
| `nextcloud_update_event` | event_url, summary?, start?, end?, description?, location?, attendees?, send_invitations? | Change one event's fields; the write is guarded by the event's etag. |
| `nextcloud_delete_calendar_object` | object_url | Delete one event or task. |

### Contract: Schedule a meeting with draft invitations

1. `nextcloud_list_calendars` — pick the calendar.
2. `nextcloud_list_events`(calendar_url, start, end) — check the slot is free.
3. `nextcloud_create_event`(calendar_url, summary, start, end, attendees, send_invitations=false) — invites stay drafts.
4. `nextcloud_get_calendar_object`(object_url) — verify.

## Gotchas
- Invitations are drafts by default: attendees are stored with SCHEDULE-AGENT=CLIENT and the server sends nothing. Set `send_invitations=true` only on an explicit request to email real people.
- A naive time is read as UTC; pass an offset for local times.
- Updates are guarded by the event's etag; a concurrent edit fails rather than overwriting.
- `calendar_url` and `event_url` come from the list results (`url`).
