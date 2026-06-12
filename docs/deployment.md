# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`nextcloud-agent` exposes its MCP server (console script `nextcloud-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Replace the `<your-…>` placeholders with the values from the **Configuration / Environment Variables** section.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code):

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "command": "uvx",
      "args": ["--from", "nextcloud-agent", "nextcloud-mcp"],
      "env": {
        "NEXTCLOUD_URL": "<your-nextcloud_url>"
      }
    }
  }
}
```

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from nextcloud-agent nextcloud-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "command": "uvx",
      "args": ["--from", "nextcloud-agent", "nextcloud-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "NEXTCLOUD_URL": "<your-nextcloud_url>"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "nextcloud-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "-e", "NEXTCLOUD_URL=<your-nextcloud_url>",
        "knucklessg1/nextcloud-agent:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name nextcloud-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e NEXTCLOUD_URL="<your-nextcloud_url>" \
  knucklessg1/nextcloud-agent:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "nextcloud-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run nextcloud-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "nextcloud-mcp": { "url": "http://nextcloud-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://nextcloud-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://nextcloud-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

This page covers running `nextcloud-agent` as a long-lived service: the MCP server
transports, the integrated A2A agent, a Docker Compose stack, putting it behind a
Caddy reverse proxy, and giving it a DNS name with Technitium. To provision the
**Nextcloud instance** it connects to, see [Backing Platform](platform.md).

> `nextcloud-agent` ships **two** console scripts: an **MCP server**
> (`nextcloud-mcp`) — a typed, deterministic tool surface a policy router / agent
> calls — and an **A2A agent server** (`nextcloud-agent`), a Pydantic-AI graph agent
> with an optional web UI that calls the MCP server over `MCP_URL`.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    nextcloud-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    nextcloud-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    nextcloud-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`nextcloud-agent` is configured entirely from the environment. The **required** set:

| Var | Default | Meaning |
|---|---|---|
| `NEXTCLOUD_URL` | `https://nextcloud.example.com` | Nextcloud base URL |
| `NEXTCLOUD_USERNAME` | _(unset)_ | Nextcloud user id |
| `NEXTCLOUD_PASSWORD` | _(unset)_ | Password or app password |
| `NEXTCLOUD_SSL_VERIFY` | `True` | Verify TLS (set `False` for self-signed homelab) |
| `HOST` | `0.0.0.0` | Bind host for HTTP transports |
| `PORT` | `8000` | Bind port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |

The five tool modules are each gated by a toggle (all default `True`): `FILESTOOL`,
`USERTOOL`, `SHARINGTOOL`, `CALENDARTOOL`, `CONTACTSTOOL`. The full set — including
telemetry (`ENABLE_OTEL`, `OTEL_*`) and access governance (`EUNOMIA_*`) variables —
is documented in
[`.env.example`](https://github.com/Knuckles-Team/nextcloud-agent/blob/main/.env.example).
Copy it to `.env` and fill in only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/nextcloud-agent/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
services:
  nextcloud-agent-mcp:
    image: knucklessg1/nextcloud-agent:latest
    container_name: nextcloud-agent-mcp
    hostname: nextcloud-agent-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
cp .env.example .env          # then edit NEXTCLOUD_* values
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Run the agent server

The A2A agent (`nextcloud-agent`) is a Pydantic-AI graph agent that calls the MCP
server. It is exposed on port `9016` and wires to the MCP server through `MCP_URL`.

```bash
# Point the agent at a running MCP server and a model provider
export MCP_URL=http://localhost:8000/mcp
nextcloud-agent --provider openai --model-id gpt-4o
```

The repo ships
[`docker/agent.compose.yml`](https://github.com/Knuckles-Team/nextcloud-agent/blob/main/docker/agent.compose.yml),
which runs the MCP server and the agent together. The agent container reaches the MCP
server by container name:

```yaml
services:
  nextcloud-agent-mcp:
    image: knucklessg1/nextcloud-agent:latest
    container_name: nextcloud-agent-mcp
    hostname: nextcloud-agent-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"

  nextcloud-agent-agent:
    image: knucklessg1/nextcloud-agent:latest
    container_name: nextcloud-agent-agent
    hostname: nextcloud-agent-agent
    restart: always
    depends_on:
      - nextcloud-agent-mcp
    env_file:
      - ../.env
    command: [ "nextcloud-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9016
      - MCP_URL=http://nextcloud-agent-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9016:9016"
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

## Behind a Caddy reverse proxy

Expose the HTTP servers on hostnames with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
nextcloud-agent.arpa {
    tls internal
    reverse_proxy nextcloud-agent-mcp:8000
}

nextcloud-agent-ui.arpa {
    tls internal
    reverse_proxy nextcloud-agent-agent:9016
}
```

```caddy
# Public — automatic Let's Encrypt
nextcloud-agent.example.com {
    reverse_proxy nextcloud-agent-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=nextcloud-agent.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `nextcloud-agent.arpa → <caddy-host-ip>` in the Technitium
web console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json` (multiplexer nickname `nc`):

```json
{
  "mcpServers": {
    "nextcloud-agent": {
      "command": "uv",
      "args": ["run", "nextcloud-mcp"],
      "env": {
        "NEXTCLOUD_URL": "https://nextcloud.example.com",
        "NEXTCLOUD_USERNAME": "your-user",
        "NEXTCLOUD_PASSWORD": "your-app-password"
      }
    }
  }
}
```

For a remote HTTP server, point the client at `http://nextcloud-agent.arpa/mcp`
instead.
