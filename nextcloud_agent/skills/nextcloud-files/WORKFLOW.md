# Nextcloud Files

Browse and manage Nextcloud files and folders over WebDAV via the nextcloud-agent MCP server — list a directory, read/download a file, upload, create/delete/move/copy, and read properties. Use when the agent must navigate a Nextcloud file tree, fetch a document's bytes, or reorganize storage. Do NOT use for sharing grants (use nextcloud-shares) or pushing files into the knowledge graph (use nextcloud-ingest); prefer those.

# Nextcloud Files

Domain-typed access to the Nextcloud **file store** (WebDAV) for browsing and
managing files and folders. Prefer the `nextcloud_files` condensed tool over raw
WebDAV requests — it carries the PROPFIND parsing and returns file-shaped records.

## When to use
- List the contents of a folder (files + subfolders with size/mtime/mime).
- Read (download) a file's bytes, or read a path's properties.
- Upload a file, create a folder, or delete/move/copy an item.

## When NOT to use
- Create/list/revoke shares → `nextcloud-shares`.
- Store a file's bytes + extracted text into the knowledge graph →
  `nextcloud-ingest`.
- Calendar events or contacts → the `nextcloud_calendar` / `nextcloud_contacts`
  tools.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`nextcloud-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `NEXTCLOUD_URL` | ✅ | Base URL of the Nextcloud instance |
| `NEXTCLOUD_USERNAME` | ✅ | WebDAV/OCS user |
| `NEXTCLOUD_PASSWORD` | ✅ | Password or app password |
| `NEXTCLOUD_TLS_PROFILE` | optional | Named AgentConfig TLS profile; peer and hostname verification are mandatory. |
| `NEXTCLOUD_TLS_PROFILE_REF` | optional | Reference-backed AgentConfig TLS profile selector. |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (used
below) vs. the 1:1 verbose tools. `FILESTOOL` gates this tool category.

## Tools & actions
Prefer the **condensed** tool; it takes `action` + a `params_json` **JSON string**
whose keys are passed straight to the client method.

| Condensed tool | Actions |
|----------------|---------|
| `nextcloud_files` | `list_files`, `read_file`, `write_file`, `create_folder`, `delete_item`, `move_item`, `copy_item`, `get_properties` |

### Key parameters
- `path` — server-relative path for most actions (e.g. `Documents/report.pdf`); `""`
  or `/` is the user root for `list_files`.
- `content` — file body (str or bytes) for `write_file`; `overwrite` (bool).
- `source_path` + `dest_path` — for `move_item` / `copy_item`.

## Recipes (`params_json`)
List the root directory:
```json
{"path":""}
```
Download a document:
```json
{"path":"Documents/report.pdf"}
```
Create a folder then move a file into it:
```json
{"path":"Reports/2026"}
```
```json
{"source_path":"report.pdf","dest_path":"Reports/2026/report.pdf"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `list_files` uses PROPFIND `Depth: 1`; it returns the folder itself plus its
  direct children.
- `read_file` returns raw bytes — for large binaries prefer `nextcloud-ingest`,
  which stores the bytes durably in the KG instead of round-tripping them.

## Related
- **`nextcloud-ingest`** — the same `read_file` bytes, but stored natively in the
  knowledge graph as a `:Blob`/`:AssetOccurrence` + extracted `:Document`.
- **`nextcloud-shares`** — grant/revoke access to the files listed here.
