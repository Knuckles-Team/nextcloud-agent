# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
