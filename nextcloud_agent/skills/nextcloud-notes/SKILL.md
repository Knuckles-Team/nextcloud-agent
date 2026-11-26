---
name: nextcloud-notes
skill_type: skill
description: >-
  Keep Nextcloud Notes: list notes by category, read, create, update and delete a note (markdown, stored as files in the Notes folder). Use when the agent must keep meeting notes, checklists or reference text the team can open in Notes. Do NOT use for collaborative office documents (nextcloud-office).
license: MIT
tags: [nextcloud, notes, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Notes

Keep Nextcloud Notes: list notes by category, read, create, update and delete a note (markdown, stored as files in the Notes folder).

## When to use
- Meeting notes: `nextcloud_create_note(title, content, category="Meetings")`.
- Append: `nextcloud_get_note(note_id)` then `nextcloud_update_note(note_id, content=...)`.

## Steps

1. List notes with `nextcloud_list_notes`(category). Verify whether a matching note already exists.
2. Create with `nextcloud_create_note`(title, content, category="Meetings"). Verify the response returns a note id.
3. Append by reading `nextcloud_get_note`(note_id) and writing the merged text with `nextcloud_update_note`(note_id, content). Verify the re-read note holds both old and new text.
4. Delete only on request with `nextcloud_delete_note`(note_id). Verify the id is gone from `nextcloud_list_notes`().

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| 401 or connection error from a `nextcloud_*` tool | `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` or the app password in `NEXTCLOUD_PASSWORD` is missing or wrong | Set the three variables for the server, and verify the connection by calling `nextcloud_get_user_info` |
| Update refused | No field was passed to `nextcloud_update_note` | Pass at least one of title, content, category or favorite and verify with `nextcloud_get_note`(note_id) |
| Note not found | The note_id is stale | Take the id from `nextcloud_list_notes`(category) |

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_notes` | category? | List the account's notes, optionally of one category. |
| `nextcloud_get_note` | note_id | Read one note with its content. |
| `nextcloud_create_note` | title, content?, category? | Create one note. |
| `nextcloud_update_note` | note_id, title?, content?, category?, favorite? | Change one note's title, content, category or favorite flag. |
| `nextcloud_delete_note` | note_id | Delete one note. |

## Gotchas
- Notes are files under `Notes/<category>/`; they also appear in the Files tools.
- An update with no field refuses.
