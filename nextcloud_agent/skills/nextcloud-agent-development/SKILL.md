---
name: nextcloud-agent-development
skill_type: skill
description: >-
  Develop and verify nextcloud-agent as a connector-owned package while preserving
  its SDK Connector wiring, action-routed MCP tools, Nextcloud authentication and
  TLS boundary, packaged content, and source-backed spec-delivery publication.
---

# Nextcloud Agent Development

Use this skill for changes to the `nextcloud-agent` connector. The package owns
Nextcloud API calls, mappings, MCP tools, and connector content; it does not own
the connector SDK, Arbora admission, the shared Pages renderer, or Pipelines
hooks.

## Repository map and runtime boundary

- `nextcloud_agent/__init__.py` creates the SDK `Connector` with `NextcloudAPI`,
  `get_client`, and `nextcloud_agent.mcp_server` as its tool module. Preserve this
  wiring when changing a client or tool.
- `nextcloud_agent/mcp_server.py` registers the condensed action-routed tools
  `nextcloud_files`, `nextcloud_user`, `nextcloud_sharing`, `nextcloud_calendar`,
  and `nextcloud_contacts`. Their action lists and client calls are the runtime
  contract; the domain clients live under `nextcloud_agent/api/` and are composed
  by `nextcloud_agent/api/__init__.py`.
- `nextcloud_agent/auth.py` reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, and
  `NEXTCLOUD_PASSWORD` through the SDK setting boundary. The API client resolves
  the named `nextcloud` TLS profile through the SDK and configures a verified,
  same-origin requests session; preserve that transport boundary and its cleanup.
- `nextcloud_agent/ontology/`, `nextcloud_agent/prompts/`, and
  `nextcloud_agent/skills/` contain connector content. The package-data table in
  `pyproject.toml` declares those trees plus `agent_data/**` and `connectors/**`;
  inspect a built wheel when changing packaged content.
- The current `pyproject.toml` declares `nextcloud-mcp` as its console script and
  `nextcloud-mcp` as its connector identity. Older README, AGENTS, and deployment
  examples mention a `nextcloud-agent` console script and `agent_server.py`, but
  those are not declared or present in this source tree. Do not invent that
  runtime or rewrite it as part of a Pages/skill change.

## Rapid delivery: the 5-minute contract

1. Read the owning `specs/` row and this repository's `AGENTS.md` before editing.
2. Inspect the active client/tool path and focused tests before changing API
   behavior, auth, ontology content, or packaged skills.
3. Prefer the smallest connector-owned change. Preserve action names, parameter
   schemas, SDK `Connector` wiring, Nextcloud credentials, TLS verification,
   same-origin checks, and package-data declarations unless the owning requirement
   changes them.
4. Run focused tests through the narrow test-slot wrapper, then run the repository
   pre-commit and pre-push gates without bypasses.
5. For public packaged content, bump the patch version and refresh `uv.lock`
   together; inspect the built wheel for the new skill.
6. Treat Pages as a separate proof: after the exact main SHA has a successful
   Pages run, check `/spec-delivery/`, its `index.html`, and `snapshot.json` over
   HTTPS and record the revision from the snapshot.

## Focused development commands

From the repository root:

```bash
uv sync --locked

/var/tmp/arbora-delivery/scale/run-narrow-test-slot.sh \
  uv run --locked --extra test pytest -q \
    tests/test_init.py \
    tests/test_init_dynamics.py \
    tests/test_auth.py \
    tests/test_api_client.py \
    tests/test_mcp_server.py \
    tests/test_kg_ingest.py \
    tests/test_xml_security.py \
    tests/test_fleet_name.py \
    tests/test_publication_workflow.py

uvx pre-commit run --all-files
uvx pre-commit run --all-files --hook-stage pre-push
```

These checks cover SDK server construction, runtime entry points, action
dispatch, authentication, XML security, ingestion, fleet identity, and the
Pages/release contract. Do not replace them with a live Nextcloud call or a
credential-bearing test.

## Pages and package publication

`.github/workflows/pages.yml` is the one Pages caller. It invokes the shared
Pipelines `pages_pipeline.yml@main` with `content_source: docs` and the version 1
`.github/spec-dashboard.json` source declaration for `specs/*/status.json`.
Keep manual dispatch and the docs/spec/config/workflow path filters. The old direct
`mkdocs gh-deploy` publisher must not be restored because it would compete with the
shared dashboard build.

After a merge, verify the exact main SHA in the successful Pages run and probe:

```bash
curl -fsSI https://knuckles-team.github.io/nextcloud-agent/
curl -fsSI https://knuckles-team.github.io/nextcloud-agent/spec-delivery/
curl -fsSI https://knuckles-team.github.io/nextcloud-agent/spec-delivery/index.html
curl -fsS https://knuckles-team.github.io/nextcloud-agent/spec-delivery/snapshot.json
```

Keep `.github/workflows/pipeline.yml` on the shared `connector: true` and
`render_connector: true` flags. Central generation owns generated connector
manifests and release container artifacts; preserve this package's existing
runtime, auth, and port semantics and inspect generated output before making
claims.

## Change discipline

- Keep Nextcloud API calls, request models, ontology mappings, MCP schemas, and
  ingestion declarations under this repository's `CONN-NEXTCLOUD-AGENT-*` boundary.
  SDK transport and central renderer behavior belong to their owner repositories.
- Use `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, and `NEXTCLOUD_PASSWORD` only in
  injected runtime configuration; never print or commit their values. Preserve
  TLS certificate verification, same-origin checks, and the SDK-resolved named
  profile path.
- For mutations, inspect current state first, require operator approval where the
  existing connector procedure calls for it, and verify the result. Keep KG
  ingestion opt-in and preserve its provenance/tenant contract.
- Keep generated status changes and central promotion with the Pipelines/status
  owner. Do not claim Pages or public package results from a local build.
