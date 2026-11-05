# CONN-NEXTCLOUD-AGENT — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The file methods of the client move to the SDK WebDAV backend. The live caller is the `nextcloud-mcp` console script. Each file tool calls a client method, and that method now calls a backend path method. The connector deletes its own WebDAV request code for these methods.

## Technical Context

- **Language/Version**: Python >=3.12 (environment 3.14)
- **Primary Dependencies**: the SDK release with `scion_sdk.storage.WebDavBackend`; the existing share, calendar and contact libraries
- **Storage**: N/A, the connector keeps no store of its own
- **Testing**: pytest with a mock HTTP transport
- **Target Platform**: Linux container and stdio or streamable HTTP MCP clients
- **Project Type**: connector
- **Performance Goals**: N/A, the Nextcloud server sets the latency
- **Constraints**: file requests go through the SDK governed client
- **Scale/Scope**: 3 rows

## Constitution Check

- Organization constitution, spec-first change: pass. Every behavior is a row with a planned test.
- Repository article R-I, connector scope: pass. The WebDAV backend lives in the SDK, and the connector reuses it.

## Project Structure

- `nextcloud_agent/api/api_client_files.py` (existing): file methods call the backend.
- `nextcloud_agent/api/api_client_base.py` (existing): builds the backend.
- `nextcloud_agent/auth.py` (existing): reads the private-host setting.
- `tests/test_files_webdav.py` (new, planned).

## Phase 0: Research

No unknowns. The SDK row holds the backend decisions.

## Phase 1: Design

The client keeps its public method names, so the tool layer does not change. `list_contents` maps each SDK entry to the old dictionary shape. The quota call and the share, calendar and contact calls keep the current client. The shared XML guard stays for those calls.

## Phase 2: Task approach

Pin the SDK first (R003), then build the backend (R002), then move the file methods (R001).

## Complexity Tracking

None. The constitution check passed.

## Build contract

Gates: `pre-commit run --all-files` and the pre-push stage, then `uv run pytest tests`. The file methods send no request through the connector's own session.

## Production caller chain

Console script `nextcloud-mcp` → `scion_sdk.cli:serve` → the file tool in `nextcloud_agent/mcp_server.py` → `get_client` (nextcloud_agent/auth.py:11) → the file methods in `nextcloud_agent/api/api_client_files.py` → `WebDavBackend` in `scion_sdk/storage/webdav.py`.

## Cross-repo seams

- SDK WebDAV backend: provider `SCION-CENTRALIZED-CAPABILITIES-R042.5`, consumer `CONN-NEXTCLOUD-AGENT-R001`; see [contracts/storage-backends.md](contracts/storage-backends.md).
