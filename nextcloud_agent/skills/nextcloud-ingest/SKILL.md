---
name: nextcloud-ingest
skill_type: skill
description: >-
  Federate Nextcloud records into the knowledge graph through the connector's declared sources:
  files and folders, shares, notes, Deck boards, Talk conversations, calendars and address books.
  Use when the agent must make Nextcloud data queryable in the graph or refresh it. Do NOT use to
  download or move a file (nextcloud-files) or to change shares (nextcloud-shares).
license: MIT
tags: [nextcloud, ingest, knowledge-graph, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud ingest

The connector declares its ingestible sources at `connector://resources`. Each source names a
read tool, the record id field and the graph class; the platform's source sync runs the tool and
upserts the records. The connector itself holds no ingest tool.

| Source id | Read tool | Record | Graph class |
|-----------|-----------|--------|-------------|
| `nextcloud-files` | `nextcloud_list_files` | file or folder (`file_id`) | `:File` / `:Folder` |
| `nextcloud-shares` | `nextcloud_list_shares` | share (`id`) | `:Share` |
| `nextcloud-notes` | `nextcloud_list_notes` | note (`id`) | `:NextcloudNote` |
| `nextcloud-deck-boards` | `nextcloud_list_boards` | board (`id`) | `:DeckBoard` |
| `nextcloud-talk-conversations` | `nextcloud_list_conversations` | conversation (`token`) | `:TalkConversation` |
| `nextcloud-calendars` | `nextcloud_list_calendars` | calendar (`href`) | `:NextcloudCalendar` |
| `nextcloud-address-books` | `nextcloud_list_address_books` | address book (`href`) | `:AddressBook` |

## Workflow
1. Read `connector://resources` on the `nextcloud-mcp` server to see the current sources.
2. Ask the platform to sync one source (through arbora: `find(intent="sync connector source")`).
3. Verify with a graph query on the class (for example `:File` nodes whose `:path` starts with the folder).

## Gotchas
- Where each record family lives (store, protocol, read tool, route parameter and federation key) is
  declared in the connector's data locations and served in its surface ontology.
- File bytes are not copied into the graph by a listing; a document's text is extracted only when the
  platform's media ingestion reads the file.
- Records carry the connected account's view: only what that account can see is ingested.
