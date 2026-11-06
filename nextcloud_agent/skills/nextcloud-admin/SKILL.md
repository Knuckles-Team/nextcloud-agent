---
name: nextcloud-admin
skill_type: skill
description: >-
  Administer the Nextcloud workspace: read the connected account; list, read, create, update, disable, enable and delete accounts; list, create and delete groups and manage membership; list installed apps and enable or disable them. Use when the agent must onboard or offboard a team member, organize teams into groups, or check which apps are on. Requires an admin account. Do NOT use for contact cards (nextcloud-contacts).
license: MIT
tags: [nextcloud, admin, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Admin

Administer the Nextcloud workspace: read the connected account; list, read, create, update, disable, enable and delete accounts; list, create and delete groups and manage membership; list installed apps and enable or disable them.

## When to use
- Onboard: `nextcloud_create_user(user_id, email, display_name, groups=[team])` sends a set-password invite (needs outgoing mail) or pass a generated password through a secret channel, never chat.
- Offboard: `nextcloud_disable_user` first; `nextcloud_delete_user` deletes the account's files and is confirmation-gated.
- Apps: `nextcloud_list_apps(status="enabled")`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_get_user_info` |  | Read the connected account: id, display name, email, quota, groups. |
| `nextcloud_list_users` | search?, limit?, offset? | List account ids, optionally matching a search string. |
| `nextcloud_get_user` | user_id | Read one account's details. |
| `nextcloud_create_user` | user_id, password?, email?, display_name?, groups?, quota? | Create one account. Give a password, or an email for a set-password invite. |
| `nextcloud_update_user` | user_id, key, value | Change one field of an account. |
| `nextcloud_disable_user` | user_id | Disable one account (it can no longer log in; data is kept). |
| `nextcloud_enable_user` | user_id | Enable one disabled account. |
| `nextcloud_delete_user` | user_id | Delete one account and its files. |
| `nextcloud_list_groups` | search?, limit? | List group ids, optionally matching a search string. |
| `nextcloud_create_group` | group_id, display_name? | Create one group. |
| `nextcloud_delete_group` | group_id | Delete one group (its members keep their accounts). |
| `nextcloud_list_group_members` | group_id | List the account ids in one group. |
| `nextcloud_add_user_to_group` | user_id, group_id | Add one account to a group. |
| `nextcloud_remove_user_from_group` | user_id, group_id | Remove one account from a group. |
| `nextcloud_list_apps` | status? | List installed app ids that are enabled, disabled, or all of them. |
| `nextcloud_get_app` | app_id | Read one installed app's metadata (name, version, dependencies). |
| `nextcloud_enable_app` | app_id | Enable one installed app. |
| `nextcloud_disable_app` | app_id | Disable one app. |

### Contract: Onboard a team member

1. `nextcloud_create_user`(user_id, email, display_name, groups=[<team group>]) — email invite, no password in chat.
2. `nextcloud_get_user`(user_id) — verify.
3. `nextcloud_add_participant`(token, participant=user_id) — add to the team conversation.

## Gotchas
- Installing new apps from the app store is a deployment change, not a tool call; the deployment declares the business app set.
- `nextcloud_update_user` changes one field per call.
- Never print or store passwords.
