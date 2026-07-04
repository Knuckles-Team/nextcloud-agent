---
name: nextcloud-ingest
description: >-
  Natively ingest Nextcloud files into the epistemic-graph knowledge graph via
  the nextcloud-agent MCP server — fetch a file over WebDAV and store its raw
  bytes as a content-addressed :Blob/:MediaAsset plus its extracted text
  (pdf/office/txt or image OCR) as a linked :Document. Use when the agent must
  make a Nextcloud file durable, deduped, and semantically searchable in the KG.
  Do NOT use to merely download or move a file (use nextcloud-files) or to manage
  shares (use nextcloud-shares); prefer those.
license: MIT
tags: [nextcloud, ingest, knowledge-graph, blob, document, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Nextcloud Ingest

Push Nextcloud files into the **epistemic-graph knowledge graph** in their richest
modality. A single call stores the file's raw bytes as a content-addressed
`:Blob` + `:MediaAsset` node and, when the file is a document or image, extracts
its text (`read_any` / OCR) into a linked `:Document` for semantic search.
This is the package's "maximum ingestion" seam — richer than the declarative
folder-listing connector preset.

## When to use
- Make a specific Nextcloud file durable in the KG (raw bytes + searchable text).
- Bring a PDF/office doc/image into the graph so it can be retrieved semantically.

## When NOT to use
- Just download bytes or reorganize storage → `nextcloud-files`.
- Manage sharing grants → `nextcloud-shares`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`nextcloud-agent`** MCP server.
A reachable **epistemic-graph engine** is required for anything to be stored; with
no engine the tool **no-ops cleanly** (returns `status: skipped`) so the call is
always safe.

| Variable | Required | Notes |
|----------|----------|-------|
| `NEXTCLOUD_URL` / `NEXTCLOUD_USERNAME` / `NEXTCLOUD_PASSWORD` | ✅ | WebDAV creds |
| `NEXTCLOUD_SSL_VERIFY` | optional | TLS verification toggle |

`INGESTTOOL` gates this tool category.

## Tools & actions
This is a single-purpose typed tool (not action-routed).

| Tool | Parameter | Purpose |
|------|-----------|---------|
| `nextcloud_ingest_file` | `path` | Fetch + natively ingest one file into the KG |

### What it stores
- **`:Blob` + `:MediaAsset`** — the raw file bytes, content-addressed (deduped by
  digest), carrying `remote_path`, `file_id`, `etag`, mime, and size.
- **`:Document`** (when the file is pdf/office/txt/csv/html or an image) — the
  extracted plain text, `source_uri` `nextcloud://<path>`, linked back to the file
  via `hasBlob`.

## Recipes
Ingest a report PDF:
```
nextcloud_ingest_file(path="Documents/report.pdf")
```
Returns e.g.:
```json
{"status":"ingested","path":"Documents/report.pdf","asset_id":"media:…","digest":"…","size_bytes":48213,"media_type":"file","doc_id":"nextcloud:document:…"}
```

## Gotchas
- Requires a live epistemic-graph engine; otherwise `status: skipped` (never an
  error) — this is by design so the connector runs with zero KG infrastructure.
- Text extraction only runs for document/image types; audio/video/binary files
  are stored as a blob with no `:Document`.
- Re-ingesting an unchanged file is cheap — the blob is deduped by content digest.

## Related
- **`nextcloud-files`** — locate the `path` to ingest with `list_files` first.
- The `nextcloud-files` connector preset mirrors folder **structure** nodes; this
  tool adds the **content** (blob + document).
