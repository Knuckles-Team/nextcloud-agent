# Code Enhancement: nextcloud-agent

> Automated code enhancement review for nextcloud-agent. Covers 17 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: C, score: 74)**, so that **improve project project analysis from C to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: C, score: 79)**, so that **improve project codebase optimization from C to at least B (80+)**.
- As a **developer**, I want to **address Architecture & Design Patterns findings (grade: C, score: 75)**, so that **improve project architecture & design patterns from C to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 42)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Test Execution findings (grade: F, score: 25)**, so that **improve project test execution from F to at least B (80+)**.
- As a **developer**, I want to **address Version Sync Analysis findings (grade: D, score: 60)**, so that **improve project version sync analysis from D to at least B (80+)**.
- As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.
- As a **developer**, I want to **address Environment Variables findings (grade: C, score: 79)**, so that **improve project environment variables from C to at least B (80+)**.
- As a **developer**, I want to **address analyze_xdg_kg findings (grade: F, score: 0)**, so that **improve project analyze_xdg_kg from F to at least B (80+)**.

## Functional Requirements

- **FR-001**: Minor update: agent-utilities 0.2.40 (installed) -> 0.16.0
- **FR-002**: Minor update: pytest-xdist 3.6.0 (constraint — not installed) -> 3.8.0
- **FR-003**: Minor update: python-dateutil 2.8.2 (constraint — not installed) -> 2.9.0.post0
- **FR-004**: MAJOR update: icalendar 6.1.1 (constraint — not installed) -> 7.1.2
- **FR-005**: Test suite lacks intent diversity (only one type)
- **FR-006**: 15 potential doc-test drift items
- **FR-007**: README.md missing sections: usage|quick start
- **FR-008**: 2 broken internal links in README.md
- **FR-009**: README missing: Has a Table of Contents
- **FR-010**: README missing: Has usage examples with code blocks
- **FR-011**: SRP: 1 classes have >15 methods
- **FR-012**: No discernible layer architecture (no domain/service/adapter separation)
- **FR-013**: Low traceability ratio: 14% concepts fully traced
- **FR-014**: 10 orphaned concepts (only in one source)
- **FR-015**: 3 test functions missing concept markers
- **FR-016**: Total lint findings: 0 (high/error: 0, medium/warning: 0, low: 0)
- **FR-017**: 2 hook(s) may be outdated: ruff-pre-commit, uv-pre-commit
- **FR-018**: Found 5 file(s) with version '0.15.0' that are NOT tracked in .bumpversion.cfg:
- **FR-019**:   - .specify/reports/nextcloud-agent/results.json
- **FR-020**:   - .specify/specs/code-enhancement-20260522/tasks.json
- **FR-021**:   - .specify/specs/code-enhancement-20260522/tasks.md
- **FR-022**:   - .specify/specs/code-enhancement-20260522/spec.md
- **FR-023**:   - .specify/specs/code-enhancement-20260522/spec.json
- **FR-024**: CHANGELOG.md exists but could not be parsed — check format compliance
- **FR-025**: No changelog entries within the last 30 days
- **FR-026**: keepachangelog not installed — pip install 'universal-skills[code-enhancer]'
- **FR-027**: Test directory lacks subdirectory organization (consider unit/, integration/, e2e/)
- **FR-028**: No @pytest.mark.parametrize usage — consider data-driven tests
- **FR-029**: 1 tests exceed 100 lines — likely doing too much per test
- **FR-030**: Partial env var documentation: 40% coverage
- **FR-031**: Undocumented env vars: AUTH_TYPE, CALENDARTOOL, CONTACTSTOOL, EUNOMIA_POLICY_FILE, EUNOMIA_TYPE, FILESTOOL, NEXTCLOUD_PASSWORD, TLS_PROFILE, NEXTCLOUD_USERNAME, OTEL_EXPORTER_OTLP_ENDPOINT
- **FR-032**: 3 Python env vars not in .env.example: NEXTCLOUD_PASSWORD, TLS_PROFILE, NEXTCLOUD_USERNAME
- **FR-033**: Analysis error: No module named 'agent_utilities.knowledge_graph'

## Success Criteria

- Overall GPA: 2.35 → 3.0
- Domains at B or above: 8 → 17
- Actionable findings: 33 → 0
