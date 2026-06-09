# Installation

`nextcloud-agent` is a standard Python package and a prebuilt container image. Pick
the path that matches how you want to run it.

## Requirements

- **Python 3.11 – 3.14**.
- A reachable **Nextcloud instance** with a user and an app password — see
  [Backing Platform](platform.md) to deploy one locally.

## From PyPI (recommended)

```bash
pip install nextcloud-agent
```

The base install pulls in `agent-utilities[agent,logfire]`, so both the MCP server
(`nextcloud-mcp`) and the graph agent (`nextcloud-agent`) are available immediately.

### Optional extras

| Extra | Install | Pulls in |
|---|---|---|
| _(base)_ | `pip install nextcloud-agent` | MCP server + agent runtime via `agent-utilities[agent,logfire]` |
| `test` | `pip install "nextcloud-agent[test]"` | `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-xdist` for the test suite |

```bash
# Typical: run the MCP server and the agent
pip install nextcloud-agent
```

## From source

```bash
git clone https://github.com/Knuckles-Team/nextcloud-agent.git
cd nextcloud-agent
pip install -e ".[test]"          # editable install with the test extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[test]"
uv run nextcloud-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (entrypoint
`nextcloud-mcp`):

```bash
docker pull knucklessg1/nextcloud-agent:latest

docker run --rm -i \
  -e NEXTCLOUD_URL=https://nextcloud.example.com \
  -e NEXTCLOUD_USERNAME=your-user \
  -e NEXTCLOUD_PASSWORD=your-app-password \
  knucklessg1/nextcloud-agent:latest        # stdio transport (default)
```

For an HTTP server with a published port and the agent container, see
[Deployment](deployment.md).

## Verify the install

```bash
nextcloud-mcp --help
nextcloud-agent --help
python -c "import nextcloud_agent; print('nextcloud-agent ready')"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP server and agent behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the API, and the agent CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.
