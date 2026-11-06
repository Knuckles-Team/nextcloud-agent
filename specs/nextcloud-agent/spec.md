# CONN-NEXTCLOUD-AGENT — Nextcloud business operations

Owner: nextcloud-agent. Cross-repo IDs: `SCION-CENTRALIZED-CAPABILITIES-R042.5`.

## Purpose and user stories

The connector serves Nextcloud file, share, calendar and contact tools. Its file tools speak WebDAV. The SDK now owns one WebDAV backend, so the file tools must use it and keep no WebDAV client of their own. This spec owns that cutover.

## Why a new spec

The existing specs cover packaged content and the fleet name. No spec owns the file operations. This spec owns them.

## Scope and interfaces

Owned: `nextcloud_agent/api/api_client_files.py`, `nextcloud_agent/auth.py` and `pyproject.toml`. Provider: the SDK WebDAV backend of `SCION-CENTRALIZED-CAPABILITIES-R042.5`; see [contracts/storage-backends.md](contracts/storage-backends.md). Also owned: the per-domain clients under `nextcloud_agent/api/` (shares, calendar and tasks, contacts, Talk, Deck, Notes, activity, administration, office). Non-goals: server-side app installation, which the deployment declares.

## Open questions

- None.

## Delivery mapping

Component: nextcloud-agent connector

Code: ../../nextcloud_agent/api/api_client_files.py

## User Scenarios & Testing

### User Story 1 - File tools run through the SDK WebDAV backend (Priority: P1)

A fleet operator calls the file tools. Each call reaches Nextcloud through the SDK backend, under the SDK egress and TLS policy.

**Acceptance Scenarios**:

1. **Given** a configured connector, **When** a client writes `Docs/a.txt`, **Then** the SDK backend creates `Docs` if absent and uploads the file.
2. **Given** a configured connector, **When** a client lists `Docs`, **Then** each entry keeps the Nextcloud permissions, favorite and file ID fields.

### User Story 2 - Run the business on Nextcloud (Priority: P1)

An agent runs the business's collaboration work through typed tools: files with versions, search and trash; internal shares and public links; calendar events with draft invitations; tasks; contacts; Talk chat; Deck boards; notes; office documents; activity and notifications; and user, group and app administration.

**Acceptance Scenarios**:

1. **Given** a deployed connector, **When** an agent creates an event with attendees, **Then** the event is stored and no invitation is sent unless the agent asks.
2. **Given** a published asset, **When** an agent creates a public link with an expiry, **Then** the share returns its URL.
3. **Given** a private Talk conversation, **When** an agent posts a message, **Then** the chat history returns it.

### Edge Cases

- A listing with a document type declaration refuses with `STORAGE_NAME_INVALID` or `STORAGE_BACKEND_FAILED`, and the tool returns a typed error.
- A missing path returns the existing not-found error text.
- A write with `overwrite` false onto an existing file returns the existing conflict error.

## Requirements

- **FR-001**: The file tools use the SDK WebDAV backend and hold no WebDAV client code. Rows: `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002`.
- **FR-002**: The connector pins the SDK release with the backend. Rows: `CONN-NEXTCLOUD-AGENT-R003`.
- **FR-003**: Every business operation is one typed tool. Rows: `CONN-NEXTCLOUD-AGENT-R004` through `CONN-NEXTCLOUD-AGENT-R015`.

## Success Criteria

- **SC-001**: The file tool tests pass with every file request sent by the SDK backend. Rows: `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002`.

## Assumptions

- The SDK release that carries `SCION-CENTRALIZED-CAPABILITIES-R042.5` is published before this connector releases.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002` | plan.md Phase 1 | T-001, T-002 | landed |
| FR-002 | `CONN-NEXTCLOUD-AGENT-R003` | plan.md Phase 1 | T-003 | landed |
| FR-003 | `CONN-NEXTCLOUD-AGENT-R004`–`R015` | plan.md Phase 1 | T-004–T-015 | landed |
