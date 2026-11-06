---
name: nextcloud-tasks
skill_type: skill
description: >-
  Manage to-dos (VTODO) in Nextcloud task-capable calendars, which the Tasks app shows: list open or all tasks, create a task with due date and priority, update it, mark it complete or reopen it, delete it. Use when the agent must track personal or team to-dos. Do NOT use for kanban work items (nextcloud-deck) or meetings (nextcloud-calendar).
license: MIT
tags: [nextcloud, tasks, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Tasks

Manage to-dos (VTODO) in Nextcloud task-capable calendars, which the Tasks app shows: list open or all tasks, create a task with due date and priority, update it, mark it complete or reopen it, delete it.

## When to use
- Find a task list: `nextcloud_list_calendars` and pick one whose `components` include `VTODO`; create one with `nextcloud_create_calendar(display_name, with_tasks=true)`.
- Add a to-do: `nextcloud_create_task(calendar_url, summary, due, priority)`.
- Close it: `nextcloud_update_task(task_url, completed=true)`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_tasks` | calendar_url, include_completed? | List the tasks (VTODO) of one task-capable calendar. |
| `nextcloud_create_task` | calendar_url, summary, due?, description?, priority? | Create one task (VTODO) in a task-capable calendar (the Tasks app shows it). |
| `nextcloud_update_task` | task_url, summary?, due?, description?, completed?, percent_complete? | Change one task, or mark it complete or open again; guarded by its etag. |
| `nextcloud_delete_calendar_object` | object_url | Delete one event or task. |
| `nextcloud_list_calendars` |  | List the account's calendars and the components (VEVENT, VTODO) each holds. |
| `nextcloud_create_calendar` | display_name, slug?, with_tasks? | Create one calendar that holds events and, by default, tasks. |

## Gotchas
- The default Personal calendar holds events only; tasks need a VTODO calendar.
- `priority` is iCalendar: 1 highest, 9 lowest, 0 none.
- `nextcloud_list_tasks` hides completed tasks unless `include_completed=true`.
