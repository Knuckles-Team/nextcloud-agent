# nextcloud-agent

Nextcloud **MCP server + A2A agent** for the agent-utilities ecosystem — typed,
action-routed tools over the Nextcloud Files, Sharing, Calendar, Contacts, and User
surfaces, plus an integrated Pydantic-AI graph agent.

!!! info "Official documentation"
    This site is the canonical reference for `nextcloud-agent`, maintained alongside
    every release.

[![PyPI](https://img.shields.io/pypi/v/nextcloud-agent)](https://pypi.org/project/nextcloud-agent/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/nextcloud-agent)](https://github.com/Knuckles-Team/nextcloud-agent/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/nextcloud-agent)

## Overview

`nextcloud-agent` wraps the Nextcloud WebDAV, CalDAV, CardDAV, and OCS REST surfaces
with typed, deterministic MCP tools, and ships a graph agent that an operator or
upstream orchestrator can call. It provides:

- **`NextcloudAPI`** — a `requests`-based client over the Nextcloud WebDAV /
  CalDAV / CardDAV / OCS endpoints, composed from per-domain sub-clients (files,
  calendar, contacts).
- **Action-routed MCP tools** — five condensed tool modules (`nextcloud_files`,
  `nextcloud_user`, `nextcloud_sharing`, `nextcloud_calendar`,
  `nextcloud_contacts`), each gated by a toggle environment variable to keep the LLM
  tool surface compact.
- **An integrated A2A agent** — a Pydantic-AI graph agent (console script
  `nextcloud-agent`) with an optional web UI that calls the MCP server.

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, uv, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP server and the agent, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `NextcloudAPI` client, and the agent CLI.
- :material-database-cog: **[Backing Platform](platform.md)** — deploy Nextcloud with Docker.
- :material-sitemap: **[Overview](overview.md)** — the action-routed tool surface and ecosystem role.
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
