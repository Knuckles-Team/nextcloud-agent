---
name: nextcloud-office
skill_type: skill
description: >-
  Create office documents (docx, xlsx, pptx, odg) and collaborative markdown in Nextcloud and open browser editing sessions where everyone with access edits together in real time (Nextcloud Office on the built-in document server; Text for markdown). Use when the agent must start a document for people to work on. Do NOT use to upload a finished file (nextcloud-files).
license: MIT
tags: [nextcloud, office, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Office

Create office documents (docx, xlsx, pptx, odg) and collaborative markdown in Nextcloud and open browser editing sessions where everyone with access edits together in real time (Nextcloud Office on the built-in document server; Text for markdown).

## When to use
- New document: `nextcloud_create_document(path="Team/plans/q1.docx")`; the extension picks the type; pass a `template_path` from `nextcloud_list_document_templates` to start from a template.
- Hand a person an editor: `nextcloud_open_document(path)` returns a one-time URL for the connected account.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_document_templates` |  | List the document types the server can create (document, spreadsheet, presentation, diagram, markdown) with their extensions and templates. |
| `nextcloud_create_document` | path, template_path? | Create one new document at ``path``; the extension picks the type (``.docx``, ``.xlsx``, ``.pptx``, ``.odg``, ``.md``). |
| `nextcloud_open_document` | path | Open one document for editing and return a one-time editor ``url`` for the connected account (Office for office files, Text for markdown). |

### Contract: Create a document for the team to edit together

1. `nextcloud_create_document`(path=Team/<area>/<name>.docx) — the extension picks the type.
2. `nextcloud_create_share`(path, share_type=group, share_with=<team group>, permissions=15) — edit rights for the team.
3. `nextcloud_open_document`(path) — one-time editor URL for the requester.

## Gotchas
- Collaboration is by access: share the file or its folder with the team (nextcloud-shares, permissions 15) and every member edits it live.
- An existing path refuses (HTTP 403); pick a new name.
- Editor URLs are one-time and bound to the connected account; do not post them publicly.
- The document server runs inside Nextcloud and is reachable on the LAN host; browser access from the public domain needs the owner's domain setup.
