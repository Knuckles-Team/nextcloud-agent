---
name: nextcloud-shares
skill_type: skill
description: >-
  Grant and audit access to Nextcloud files and folders: internal shares to a user, group or Talk conversation, and public links with password, expiry, label and note. Use when the agent must give someone access, produce a public link, audit who can see a path, or revoke access. Do NOT use to move or upload files (nextcloud-files).
license: MIT
tags: [nextcloud, sharing, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Shares

Grant and audit access to Nextcloud files and folders: internal shares to a user, group or Talk conversation, and public links with password, expiry, label and note.

## When to use
- Share a folder with a team group: `share_type="group", share_with=<group id>`.
- Make a public link that expires: `share_type="public_link", expire_date="YYYY-MM-DD"`; read `url` back.
- Audit one path: `nextcloud_list_shares(path=...)`; what others shared with the account: `shared_with_me=true`.

## Steps

1. Check the current access with `nextcloud_list_shares`(path). Verify whether a share for the target already exists.
2. Create an internal share with `nextcloud_create_share`(path, share_type="group", share_with, permissions=1). Verify the response returns a share id.
3. Create a public link only on explicit request with `nextcloud_create_share`(path, share_type="public_link", expire_date, password). Verify with `nextcloud_get_share`(share_id) that expiry is set and read the url.
4. Change terms with `nextcloud_update_share`(share_id, permissions). Verify with `nextcloud_get_share`(share_id) that the value changed.
5. Revoke with `nextcloud_delete_share`(share_id). Verify the id is gone from `nextcloud_list_shares`(path).

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| 401 or connection error from a `nextcloud_*` tool | `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` or the app password in `NEXTCLOUD_PASSWORD` is missing or wrong | Set the three variables for the server, and verify the connection by calling `nextcloud_get_user_info` |
| Share creation fails | The path does not exist or share_with names an unknown user or group | Check `nextcloud_get_properties`(path) and `nextcloud_list_groups` names, then create the share again |
| Delete refused | confirm=true was not passed because the person has not agreed | Ask the person to agree, pass confirm=true on that single call, and verify the result with the matching read tool |

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_shares` | path?, subfiles?, shared_with_me? | List shares: all of the account's shares, the shares on one path, or the shares other people gave the account. |
| `nextcloud_get_share` | share_id | Read one share by id. |
| `nextcloud_create_share` | path, share_type?, share_with?, permissions?, password?, expire_date?, label?, note?, public_upload?, send_mail? | Share one file or folder and return the share (a public link carries its ``url``). |
| `nextcloud_update_share` | share_id, permissions?, password?, expire_date?, label?, note? | Change one share's permissions, password, expiry, label or note. |
| `nextcloud_delete_share` | share_id | Revoke one share. |

## Gotchas
- `permissions` is a bitmask: 1 read, 2 update, 4 create, 8 delete, 16 share (31 all). Least privilege: 1 for viewers, 15 for editors.
- `send_mail` is false by default, so nobody is emailed; the server has no outgoing mail configured anyway.
- Never widen access (public link, write permission) without an explicit request; prefer an expiry on every public link.
- Public link URLs carry the host the request used; external recipients need the public domain configured on the server.
