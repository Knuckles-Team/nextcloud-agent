# CONN-NEXTCLOUD-AGENT — Nextcloud file operations

Owner: nextcloud-agent. Cross-repo IDs: `SCION-CENTRALIZED-CAPABILITIES-R042.5`.

## Purpose and user stories

The connector serves Nextcloud file, share, calendar and contact tools. Its file tools speak WebDAV. The SDK now owns one WebDAV backend, so the file tools must use it and keep no WebDAV client of their own. This spec owns that cutover.

## Why a new spec

The existing specs cover packaged content and the fleet name. No spec owns the file operations. This spec owns them.

## Scope and interfaces

Owned: `nextcloud_agent/api/api_client_files.py`, `nextcloud_agent/auth.py` and `pyproject.toml`. Provider: the SDK WebDAV backend of `SCION-CENTRALIZED-CAPABILITIES-R042.5`; see [contracts/storage-backends.md](contracts/storage-backends.md). Non-goals: the share, calendar and contact calls, and the tool names and schemas, which stay unchanged.

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

### Edge Cases

- A listing with a document type declaration refuses with `STORAGE_NAME_INVALID` or `STORAGE_BACKEND_FAILED`, and the tool returns a typed error.
- A missing path returns the existing not-found error text.
- A write with `overwrite` false onto an existing file returns the existing conflict error.

## Requirements

- **FR-001**: The file tools use the SDK WebDAV backend and hold no WebDAV client code. Rows: `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002`.
- **FR-002**: The connector pins the SDK release with the backend. Rows: `CONN-NEXTCLOUD-AGENT-R003`.

## Success Criteria

- **SC-001**: The file tool tests pass with every file request sent by the SDK backend. Rows: `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002`.

## Assumptions

- The SDK release that carries `SCION-CENTRALIZED-CAPABILITIES-R042.5` is published before this connector releases.
- The share, calendar and contact calls keep their current client until a later row moves them.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-NEXTCLOUD-AGENT-R001`, `CONN-NEXTCLOUD-AGENT-R002` | plan.md Phase 1 | T-001, T-002 | planned |
| FR-002 | `CONN-NEXTCLOUD-AGENT-R003` | plan.md Phase 1 | T-003 | planned |
