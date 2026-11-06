---
name: nextcloud-files
skill_type: skill
description: >-
  Browse and manage Nextcloud files: list, download, upload text or binary (base64), create folders, move, copy, delete, read properties and quota, list and restore versions, search by name and type, and restore from the trash bin. Use when the agent must find, fetch, store or reorganize files, or roll a file back. Do NOT use for sharing (nextcloud-shares), publishing a produced asset (nextcloud-publish-asset) or graph ingestion (nextcloud-ingest).
license: MIT
tags: [nextcloud, files, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Files

Browse and manage Nextcloud files: list, download, upload text or binary (base64), create folders, move, copy, delete, read properties and quota, list and restore versions, search by name and type, and restore from the trash bin.

## When to use
- Find a file by name or type (`nextcloud_search_files`, `mime_prefix="image/"`).
- Upload a photo, video or office file: `nextcloud_write_file(path, content=<base64>, encoding="base64")`.
- Undo a bad edit: `nextcloud_list_versions` then `nextcloud_restore_version`.
- Recover a deleted file: `nextcloud_list_trash` then `nextcloud_restore_trash_item`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_files` | path? | List the files and folders in one folder (PROPFIND, depth 1). |
| `nextcloud_read_file` | path | Download one file; the bytes come back base64-encoded. |
| `nextcloud_write_file` | path, content, overwrite?, encoding? | Upload one file, creating missing parent folders. |
| `nextcloud_create_folder` | path | Create one folder. |
| `nextcloud_delete_item` | path | Delete one file or folder (it moves to the trash bin). |
| `nextcloud_move_item` | source_path, dest_path, overwrite? | Move or rename one file or folder. |
| `nextcloud_copy_item` | source_path, dest_path, overwrite? | Copy one file or folder. |
| `nextcloud_get_properties` | path | Read the properties of one file or folder (size, type, etag, file id). |
| `nextcloud_get_user_quota` |  | Read the storage quota of the connected account. |
| `nextcloud_list_versions` | path | List the stored versions of one file, newest first. |
| `nextcloud_restore_version` | path, version_id | Restore one stored version of a file; the current content becomes a version. |
| `nextcloud_search_files` | query, path?, mime_prefix?, limit? | Search files by name (case-insensitive substring) under one folder. |
| `nextcloud_list_trash` |  | List the items in the trash bin of the connected account. |
| `nextcloud_restore_trash_item` | trash_id | Restore one trash-bin item to its original location. |

## Gotchas
- Paths are relative to the connected account's root (`Team/launch/poster.png`), never absolute URLs.
- `nextcloud_read_file` returns the bytes base64-encoded.
- Overwriting a file keeps the old content as a version; `overwrite=false` refuses an existing target.
- `nextcloud_delete_item` moves to the trash bin and is confirmation-gated.
- Base64 through MCP suits files up to tens of MB; large videos should be written by the producing service straight to WebDAV (the craft suite already does).
