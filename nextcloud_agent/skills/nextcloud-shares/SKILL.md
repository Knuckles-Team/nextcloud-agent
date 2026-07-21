---
name: nextcloud-shares
skill_type: skill
description: >-
  Manage Nextcloud sharing grants over the OCS Sharing API via the
  nextcloud-agent MCP server — list existing shares, create a public link / user
  / group share on a file or folder, and revoke a share. Use when the agent must
  expose a file, audit who has access, or tear a share down. Do NOT use for
  browsing or moving files (use nextcloud-files) or KG ingestion (use
  nextcloud-ingest); prefer those.
license: MIT
tags: [nextcloud, shares, sharing, ocs, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Nextcloud Shares

Domain-typed access to the Nextcloud **Sharing** API (OCS `files_sharing`) for
granting and auditing access to files and folders. Prefer the `nextcloud_sharing`
tool over raw OCS calls — it carries the share-type/permission conventions.

## When to use
- List all current shares (audit who can access what).
- Create a share on a path — public link, user, or group.
- Revoke a share by id.

## When NOT to use
- Browse, read, upload, or move files → `nextcloud-files`.
- Store file bytes/text in the knowledge graph → `nextcloud-ingest`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`nextcloud-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `NEXTCLOUD_URL` | ✅ | Base URL of the Nextcloud instance |
| `NEXTCLOUD_USERNAME` | ✅ | WebDAV/OCS user |
| `NEXTCLOUD_PASSWORD` | ✅ | Password or app password |
| `TLS_PROFILE` / `TLS_PROFILE_REF` | optional | AgentConfig transport profile; peer verification is mandatory |

`MCP_TOOL_MODE` selects the condensed vs. verbose surface. `SHARINGTOOL` gates
this tool category.

## Tools & actions
Prefer the **condensed** tool; `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `nextcloud_sharing` | `list_shares`, `create_share`, `delete_share` |

### Key parameters
- `path` — server-relative path to share (`create_share`).
- `share_type` — `3` = public link, `0` = user, `1` = group (default `3`).
- `permissions` — bitmask: `1` read, `2` update, `4` create, `8` delete, `16`
  share (default `1` = read-only).
- `share_id` — required for `delete_share`.

## Recipes (`params_json`)
List all shares:
```json
{}
```
Create a read-only public link on a file:
```json
{"path":"Documents/report.pdf","share_type":3,"permissions":1}
```
Revoke a share:
```json
{"share_id":"42"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `share_type` and `permissions` are integers (OCS choice/bitmask values), not
  labels.
- A public-link `create_share` returns the generated link `url` in its response —
  read it back rather than constructing it.

## Related
- **`nextcloud-files`** — locate the `path` to share and confirm it exists first.
