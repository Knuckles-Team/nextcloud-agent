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

## Steps

1. Decide the scope: the whole account or one file. For one file read its id with `nextcloud_get_properties`(path). Verify the response contains a file id.
2. Read the stream with `nextcloud_list_activities`(limit=50) or `nextcloud_list_activities`(object_type="files", object_id=file_id). Verify entries are newest first and name the actor and action.
3. Read pending notices with `nextcloud_list_notifications`(). Verify each has a notification_id.
4. Dismiss only handled notices with `nextcloud_dismiss_notification`(notification_id) after the person agrees. Verify the id is gone from `nextcloud_list_notifications`().

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| 401 or connection error from a `nextcloud_*` tool | `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` or the app password in `NEXTCLOUD_PASSWORD` is missing or wrong | Set the three variables for the server, and verify the connection by calling `nextcloud_get_user_info` |
| Empty activity list | Nothing new since the last read, since the server answers 304 | Widen the window with `nextcloud_list_activities`(limit=50, since=<earlier id>) and verify entries appear |
| Dismiss refused | confirm=true was not passed because the person has not agreed | Ask the person to agree, pass confirm=true on that single call, and verify the result with the matching read tool |

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
