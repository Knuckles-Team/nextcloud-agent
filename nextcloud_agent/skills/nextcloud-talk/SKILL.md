---
name: nextcloud-talk
skill_type: skill
description: >-
  Use Nextcloud Talk: list, read, create, rename and delete conversations; list, add and remove participants; read chat history; post and delete messages. Use when the agent must message the team, keep a project channel, or read what was said. Do NOT use to email people or post to public social channels.
license: MIT
tags: [nextcloud, talk, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Talk

Use Nextcloud Talk: list, read, create, rename and delete conversations; list, add and remove participants; read chat history; post and delete messages.

## When to use
- Team update: find the conversation token with `nextcloud_list_conversations`, then `nextcloud_send_message(token, message)`.
- New private channel: `nextcloud_create_conversation(name, room_type="group")`, then `nextcloud_add_participant`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_conversations` |  | List the conversations the account takes part in. |
| `nextcloud_get_conversation` | token | Read one conversation by its token. |
| `nextcloud_create_conversation` | name, room_type?, invite? | Create one conversation and return it with its ``token``. |
| `nextcloud_rename_conversation` | token, name | Rename one conversation. |
| `nextcloud_delete_conversation` | token | Delete one conversation for everyone (moderators only). |
| `nextcloud_list_participants` | token | List the participants of one conversation. |
| `nextcloud_add_participant` | token, participant, source? | Add a user, group, email guest or team to one conversation. |
| `nextcloud_remove_participant` | token, attendee_id | Remove one attendee (by attendee id from list_participants) from a conversation. |
| `nextcloud_list_messages` | token, limit?, last_known_message_id? | Read the chat history of one conversation, newest first. |
| `nextcloud_send_message` | token, message, reply_to?, silent? | Post one chat message to a conversation. |
| `nextcloud_delete_message` | token, message_id | Delete one chat message. |

### Contract: Post an update to a private team conversation

1. `nextcloud_list_conversations` — find the team conversation token.
2. `nextcloud_send_message`(token, message) — internal only, never a public conversation without approval.
3. `nextcloud_list_messages`(token, limit=5) — verify it posted.

## Gotchas
- `group` conversations are private (members only); `public` ones can be joined by link. Never create a public conversation or message real people without an explicit request.
- `nextcloud_list_messages` returns newest first and includes system messages (renames, joins).
- Moderators only can delete a conversation.
