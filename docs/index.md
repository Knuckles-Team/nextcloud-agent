# nextcloud-agent

Nextcloud **MCP server + A2A agent** for the Arbora ecosystem — typed,
typed per-operation tools over the Nextcloud Files, Sharing, Calendar, Tasks, Contacts, Talk, Deck, Notes, Office, Activity and Administration
surfaces, plus an integrated Pydantic-AI graph agent.

!!! info "Official documentation"
    This site is the canonical reference for `nextcloud-agent`, maintained alongside
    every release.

The generated [spec-delivery dashboard](spec-delivery/) records the source-backed
delivery state for this connector.

[![PyPI](https://img.shields.io/pypi/v/nextcloud-agent)](https://pypi.org/project/nextcloud-agent/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/nextcloud-agent)](https://github.com/Knuckles-Team/nextcloud-agent/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/nextcloud-agent)

## Overview

`nextcloud-agent` wraps the Nextcloud WebDAV, CalDAV, CardDAV, and OCS REST surfaces
with typed, deterministic MCP tools, and ships a graph agent that an operator or
upstream orchestrator can call. It provides:

- **`NextcloudAPI`** — a `requests`-based client over the Nextcloud WebDAV /
  CalDAV / CardDAV / OCS / app REST endpoints, composed from per-domain sub-clients
  (files, shares, calendar and tasks, contacts, Talk, Deck, notes, office, activity,
  administration).
- **One typed MCP tool per operation** — 89 `nextcloud_<operation>` tools, grouped
  into ten domains, each domain gated by a toggle environment variable.
- **Skills, prompts and ontology** — a skill per domain plus the asset-publishing
  workflow, agent-defining prompts per business role, and an ontology of the tools,
  tool-use contracts and data locations.
- **An integrated A2A agent** — a Pydantic-AI graph agent (console script
  `nextcloud-agent`) with an optional web UI that calls the MCP server.

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-start: **[Installation](installation.md)** — pip, source, uv, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP server and the agent, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `NextcloudAPI` client, and the agent CLI.
- :material-database-cog: **[Backing Platform](platform.md)** — deploy Nextcloud with Docker.
- :material-sitemap: **[Overview](overview.md)** — the per-operation tool surface and ecosystem role.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:NC-*` registry.

</div>

## Quick start

```bash
pip install nextcloud-agent
nextcloud-mcp                    # stdio MCP server (default transport)
```

Connect it to a Nextcloud instance:

```bash
export NEXTCLOUD_URL=https://nextcloud.example.com
export NEXTCLOUD_USERNAME=your-user
export NEXTCLOUD_PASSWORD=your-app-password
nextcloud-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI, Docker image, all transports, the agent server, reverse proxy,
DNS).
