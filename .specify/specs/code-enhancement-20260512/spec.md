# Code Enhancement: nextcloud-agent

> Automated code enhancement review for nextcloud-agent. Covers 17 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: C, score: 74)**, so that **improve project project analysis from C to at least B (80+)**.
- As a **developer**, I want to **address Test Coverage findings (grade: D, score: 65)**, so that **improve project test coverage from D to at least B (80+)**.
- As a **developer**, I want to **address Architecture & Design Patterns findings (grade: C, score: 75)**, so that **improve project architecture & design patterns from C to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 42)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.

## Functional Requirements

- **FR-001**: 1 functions exceed 200 lines (actionable refactoring targets): register_files_tools (241L)
- **FR-002**: Monolithic: mcp_server.py (650L) — 1 functions with high complexity (worst: register_files_tools at 241L, CC=16); Low cohesion: 11 distinct concepts in one file
- **FR-003**: Test suite lacks intent diversity (only one type)
- **FR-004**: 36 potential doc-test drift items
- **FR-005**: README.md missing sections: installation
- **FR-006**: README missing: Has a Table of Contents
- **FR-007**: README missing: References /docs directory material
- **FR-008**: SRP: 1 modules exceed 500 lines (god modules)
- **FR-009**: SRP: 1 classes have >15 methods
- **FR-010**: No discernible layer architecture (no domain/service/adapter separation)
- **FR-011**: Low traceability ratio: 0% concepts fully traced
- **FR-012**: 4 test functions missing concept markers
- **FR-013**: 44 significant functions (>10 lines) missing concept markers in docstrings
- **FR-014**: Total lint findings: 5 (high/error: 2, medium/warning: 3, low: 0)
- **FR-015**: 2 hook(s) may be outdated: ruff-pre-commit, uv-pre-commit
- **FR-016**: 1 rogue/throwaway scripts detected (fix_*, validate_*, patch_*, etc.): scripts/validate_a2a_agent.py
- **FR-017**: CHANGELOG.md exists but could not be parsed — check format compliance
- **FR-018**: No changelog entries within the last 30 days
- **FR-019**: keepachangelog not installed — pip install 'universal-skills[code-enhancer]'
- **FR-020**: 2 tests have no assertions
- **FR-021**: Undocumented env vars: EUNOMIA_REMOTE_URL, NEXTCLOUD_CLIENTID, NEXTCLOUD_HOSTED, NEXTCLOUD_SECRET, OAUTH_BASE_URL, OAUTH_UPSTREAM_AUTH_ENDPOINT, OAUTH_UPSTREAM_CLIENT_ID, OAUTH_UPSTREAM_CLIENT_SECRET, OAUTH_UPSTREAM_TOKEN_ENDPOINT, REMOTE_AUTH_SERVERS
- **FR-022**: 9 Python env vars not in .env.example: CALENDARTOOL, CONTACTSTOOL, FILESTOOL, MISCTOOL, NEXTCLOUD_PASSWORD

## Success Criteria

- Overall GPA: 2.94 → 3.0
- Domains at B or above: 12 → 17
- Actionable findings: 22 → 0
