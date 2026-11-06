# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-11-06

### Added
- The business surface, one typed tool per operation (89 tools): file versions,
  search and trash bin, binary upload (`encoding="base64"`); internal shares and
  public links with password, expiry, label and note; calendars and events with
  draft invitations (SCHEDULE-AGENT=CLIENT unless `send_invitations`); tasks
  (VTODO); contacts with parsed fields; Talk conversations, participants and chat;
  Deck boards, stacks and cards; Notes; office documents and collaborative
  editing; activity and notifications; account, group and app administration
  (CONN-NEXTCLOUD-AGENT-R004 to R015).
- Skills per domain and `nextcloud-publish-asset`; agent-defining prompts for the
  operator and each business role; an ontology of the surface, tool-use contracts
  and data locations rolled up to core; declared ingest sources for shares,
  notes, Deck boards, Talk conversations, calendars and address books
  (CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002 to R004).

### Removed
- The action-routed tool module and the duplicate alias tools
  (`nextcloud_list_contents`, `nextcloud_create_directory`, `nextcloud_delete_resource`,
  `nextcloud_move_resource`, `nextcloud_copy_resource`, `nextcloud_list_calendar_events`,
  `nextcloud_create_calendar_event`). `create_event` and `create_contact` take
  structured fields instead of raw iCalendar or vCard text.
- The `SHARINGTOOL` and `USERTOOL` toggles; the domains are now `SHARESTOOL` and `ADMINTOOL`.

## [2.4.2] - 2026-11-05

### Fixed
- A host listed in `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` may use plain `http`, as the
  2.4.0 notes state; the connector's own origin check still refused it
  ("unencrypted service URL is restricted to loopback") (CONN-NEXTCLOUD-AGENT-R002).

## [2.4.1] - 2026-11-05

### Fixed
- The per-operation tools (`nextcloud_list_files` and the rest) failed with
  "'generator' object has no attribute ..." because the connector handed the SDK
  the generator dependency as its client factory. The connector now registers
  `make_client`; `get_client` stays the generator dependency of the action tools
  (CONN-NEXTCLOUD-AGENT-R002).

## [2.4.0] - 2026-11-05

### Changed
- The file tools (list, read, write, create folder, delete, move, copy) run
  through the scion-sdk WebDAV backend under the SDK egress and TLS policy
  (CONN-NEXTCLOUD-AGENT-R001); the connector's own WebDAV listing parser is gone.
- New setting `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` (comma list): hosts that may
  resolve to private addresses, and may use plain `http` (CONN-NEXTCLOUD-AGENT-R002).
  A private Nextcloud address now refuses unless listed.
- Requires `scion-sdk>=0.8.0,<0.9` (CONN-NEXTCLOUD-AGENT-R003).

## [2.3.8] - 2026-10-30

### Changed
- Require `scion-sdk>=0.6.4,<0.7.0`; `uv.lock` relocked.

## [2.3.5] - 2026-10-26

### Added
- Canonical spec-delivery Pages publication and a packaged connector development skill.

## [2.3.4] - 2026-10-24

### Changed
- Central connector migration: requires `agent-connector-sdk>=0.5.4`; server bootstrap, version, containers, ontology artifacts, security scaffold and pre-commit config are provided centrally.

## [2.3.3] - 2026-10-23

### Changed
- Requires `agent-connector-sdk>=0.4.1`; data sources are declared in code (`connector_sources`).

## [Unreleased]

### Added
-

### Changed
-

### Fixed
-

## [0.2.55] - 2026-04-29

### Added
- Initial release
