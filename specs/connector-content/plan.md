# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `nextcloud-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `nextcloud_agent` from it.
