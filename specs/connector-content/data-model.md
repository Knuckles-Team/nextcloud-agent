# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Data model

This file describes the ontology classes, individuals and content files that the rows of this spec add or reuse.

## Entities

### Pack classes

- Owner: `nextcloud_agent/ontology/nextcloud.ttl`, row `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002`.
- `:Share` under core `:FileShare`; `:FileVersion` and `:TrashItem` under `:File`; `:OfficeDocument`, `:ContactCard` and `:NextcloudNote` under `:Document`; `:NextcloudCalendar`, `:AddressBook`, `:DeckBoard` and `:DeckStack` under `:Collection`, `:DeckCard` under `:Task`; `:TalkConversation` under `:MessageChannel`; `:TalkMessage` under `:ChatMessage`; `:ActivityEvent` and `:NextcloudNotification` under `:Event`; `:NextcloudApp` under `:Application`.
- Core classes reused, not redeclared: `:File`, `:Folder`, `:CalendarEvent`, `:PersonalTask`, `:IdentityUser`, `:IdentityGroup`.

### Surface declaration

- Owner: `nextcloud_agent/connector_contracts.py`, row `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R005`, SDK dataclasses of `scion_sdk.surface_contracts`.
- `CONTRACTS`: `ToolUseContract(task, steps=(ToolStep(tool, params, result, refusals), ...))`; six business workflows and one contract per domain.
- `LINKS`: `SurfaceLink(task, tools, prompts, skills)`, one per contract plus ingest and connector operations.
- `LOCATIONS`: `DataLocation(name, holds, instance, read_tool, store, route_param, federation_key)`, one per record family (files, versions, trash, shares, calendars, contacts, Talk, Deck, notes, activity, accounts).
- `ROLES`: `BusinessRole(name, tasks)`: asset-publisher, office-coordinator, project-coordinator, workspace-admin, file-steward.
- The SDK renders the declaration as `ontology://nextcloud-mcp/contracts.ttl`, typed by its surface vocabulary that extends core.

### Prompt files

- `nextcloud_agent/prompts/<name>.json`, row `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R003`: the existing prompt schema with an `agent` object: `kind`, `business_role`, `tools`, `skills`, `data.reads`, `data.writes`, `guardrails.approval_required`, `guardrails.never`, `outputs`.
