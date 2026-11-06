---
name: nextcloud-activity
skill_type: skill
description: >-
  Read the Nextcloud activity stream (who created, changed, shared or deleted what) for the whole account or one file, and read and dismiss notifications. Use when the agent must audit recent changes, confirm an action landed, or clear handled notifications. Do NOT use to change files or shares.
license: MIT
tags: [nextcloud, activity, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Activity

Read the Nextcloud activity stream (who created, changed, shared or deleted what) for the whole account or one file, and read and dismiss notifications.

## When to use
- What changed on a file: `nextcloud_get_properties(path)` for `file_id`, then `nextcloud_list_activities(object_type="files", object_id=<file_id>)`.
- Daily digest: `nextcloud_list_activities(limit=50)` and `nextcloud_list_notifications()`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_activities` | limit?, since?, object_type?, object_id? | Read the activity stream, newest first; narrow to one object (``object_type="files"`` with a file id) when given. |
| `nextcloud_list_notifications` |  | List the account's current notifications. |
| `nextcloud_dismiss_notification` | notification_id | Dismiss one notification. |

## Gotchas
- An empty list means nothing new (the server answers 304).
- Dismissing a notification is confirmation-gated and cannot be undone.
