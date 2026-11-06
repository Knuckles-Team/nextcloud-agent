# Nextcloud Agent
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/nextcloud-agent)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/nextcloud-agent)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/nextcloud-agent)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/nextcloud-agent)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/nextcloud-agent)
![PyPI - License](https://img.shields.io/pypi/l/nextcloud-agent)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/nextcloud-agent)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/nextcloud-agent)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/nextcloud-agent)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/nextcloud-agent)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/nextcloud-agent)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/nextcloud-agent)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/nextcloud-agent)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/nextcloud-agent)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/nextcloud-agent)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/nextcloud-agent)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/nextcloud-agent)

> **Documentation** — Installation, deployment, usage across the API, CLI, and MCP
> interfaces, and guidance for provisioning the backing Nextcloud instance are
> maintained in the [official documentation](https://knuckles-team.github.io/nextcloud-agent/).

---

## Overview

**Nextcloud Agent** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Nextcloud MCP Server for Agentic AI!.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Complete support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## CLI or API

This agent wraps the Nextcloud MCP Server for Agentic AI! API. The operator can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server use dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools


<!-- MCP-TOOLS-TABLE:START -->

One typed tool per operation (89 tools), grouped by domain. Each domain is enabled unless its toggle is set false. Destructive tools ask for confirmation.

**activity** (toggle `ACTIVITYTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_dismiss_notification` | Dismiss one notification. |
| `nextcloud_list_activities` | Read the activity stream, newest first; narrow to one object (``object_type="files"`` with a file id) when given. |
| `nextcloud_list_notifications` | List the account's current notifications. |

**admin** (toggle `ADMINTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_add_user_to_group` | Add one account to a group. |
| `nextcloud_create_group` | Create one group. |
| `nextcloud_create_user` | Create one account. Give a password, or an email for a set-password invite. |
| `nextcloud_delete_group` | Delete one group (its members keep their accounts). |
| `nextcloud_delete_user` | Delete one account and its files. |
| `nextcloud_disable_app` | Disable one app. |
| `nextcloud_disable_user` | Disable one account (it can no longer log in; data is kept). |
| `nextcloud_enable_app` | Enable one installed app. |
| `nextcloud_enable_user` | Enable one disabled account. |
| `nextcloud_get_app` | Read one installed app's metadata (name, version, dependencies). |
| `nextcloud_get_user` | Read one account's details. |
| `nextcloud_get_user_info` | Read the connected account: id, display name, email, quota, groups. |
| `nextcloud_list_apps` | List installed app ids that are enabled, disabled, or all of them. |
| `nextcloud_list_group_members` | List the account ids in one group. |
| `nextcloud_list_groups` | List group ids, optionally matching a search string. |
| `nextcloud_list_users` | List account ids, optionally matching a search string. |
| `nextcloud_remove_user_from_group` | Remove one account from a group. |
| `nextcloud_update_user` | Change one field of an account. |

**calendar** (toggle `CALENDARTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_create_calendar` | Create one calendar that holds events and, by default, tasks. |
| `nextcloud_create_event` | Create one event. Attendees get a draft invitation unless ``send_invitations``. |
| `nextcloud_create_task` | Create one task (VTODO) in a task-capable calendar (the Tasks app shows it). |
| `nextcloud_delete_calendar` | Delete one calendar and everything in it (it moves to the calendar trash). |
| `nextcloud_delete_calendar_object` | Delete one event or task. |
| `nextcloud_get_calendar_object` | Read one event or task: its parsed fields, etag and raw iCalendar text. |
| `nextcloud_list_calendars` | List the account's calendars and the components (VEVENT, VTODO) each holds. |
| `nextcloud_list_events` | List the events of one calendar, optionally within an ISO time range. |
| `nextcloud_list_tasks` | List the tasks (VTODO) of one task-capable calendar. |
| `nextcloud_update_event` | Change one event's fields; the write is guarded by the event's etag. |
| `nextcloud_update_task` | Change one task, or mark it complete or open again; guarded by its etag. |

**contacts** (toggle `CONTACTSTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_create_contact` | Create one contact in an address book. |
| `nextcloud_delete_contact` | Delete one contact. |
| `nextcloud_get_contact` | Read one contact: parsed fields, etag and the raw vCard. |
| `nextcloud_list_address_books` | List the account's address books. |
| `nextcloud_list_contacts` | List the contacts of one address book with name, emails, phones and organization. |
| `nextcloud_update_contact` | Change one contact; omitted fields keep their value. Guarded by the etag. |

**deck** (toggle `DECKTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_assign_card_user` | Assign one user to a card. |
| `nextcloud_create_board` | Create one board; ``color`` is a hex RGB value without ``#``. |
| `nextcloud_create_card` | Create one card in a stack; ``duedate`` is ISO 8601. |
| `nextcloud_create_stack` | Create one stack (column) on a board. |
| `nextcloud_delete_board` | Delete one board (Deck keeps it restorable for a while). |
| `nextcloud_delete_card` | Delete one card. |
| `nextcloud_delete_stack` | Delete one stack and its cards. |
| `nextcloud_get_board` | Read one board with its labels and members. |
| `nextcloud_get_card` | Read one card. |
| `nextcloud_list_boards` | List the Deck boards the account can see. |
| `nextcloud_list_stacks` | List the stacks (columns) of one board, each with its cards. |
| `nextcloud_move_card` | Move one card to a stack (column) of the board, at position ``order``. |
| `nextcloud_update_card` | Change one card's title, description or due date. |

**files** (toggle `FILESTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_copy_item` | Copy one file or folder. |
| `nextcloud_create_folder` | Create one folder. |
| `nextcloud_delete_item` | Delete one file or folder (it moves to the trash bin). |
| `nextcloud_get_properties` | Read the properties of one file or folder (size, type, etag, file id). |
| `nextcloud_get_user_quota` | Read the storage quota of the connected account. |
| `nextcloud_list_files` | List the files and folders in one folder (PROPFIND, depth 1). |
| `nextcloud_list_trash` | List the items in the trash bin of the connected account. |
| `nextcloud_list_versions` | List the stored versions of one file, newest first. |
| `nextcloud_move_item` | Move or rename one file or folder. |
| `nextcloud_read_file` | Download one file; the bytes come back base64-encoded. |
| `nextcloud_restore_trash_item` | Restore one trash-bin item to its original location. |
| `nextcloud_restore_version` | Restore one stored version of a file; the current content becomes a version. |
| `nextcloud_search_files` | Search files by name (case-insensitive substring) under one folder. |
| `nextcloud_write_file` | Upload one file, creating missing parent folders. |

**notes** (toggle `NOTESTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_create_note` | Create one note. |
| `nextcloud_delete_note` | Delete one note. |
| `nextcloud_get_note` | Read one note with its content. |
| `nextcloud_list_notes` | List the account's notes, optionally of one category. |
| `nextcloud_update_note` | Change one note's title, content, category or favorite flag. |

**office** (toggle `OFFICETOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_create_document` | Create one new document at ``path``; the extension picks the type (``.docx``, ``.xlsx``, ``.pptx``, ``.odg``, ``.md``). |
| `nextcloud_list_document_templates` | List the document types the server can create (document, spreadsheet, presentation, diagram, markdown) with their extensions and templates. |
| `nextcloud_open_document` | Open one document for editing and return a one-time editor ``url`` for the connected account (Office for office files, Text for markdown). |

**shares** (toggle `SHARESTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_create_share` | Share one file or folder and return the share (a public link carries its ``url``). |
| `nextcloud_delete_share` | Revoke one share. |
| `nextcloud_get_share` | Read one share by id. |
| `nextcloud_list_shares` | List shares: all of the account's shares, the shares on one path, or the shares other people gave the account. |
| `nextcloud_update_share` | Change one share's permissions, password, expiry, label or note. |

**talk** (toggle `TALKTOOL`)

| MCP Tool | Description |
|----------|-------------|
| `nextcloud_add_participant` | Add a user, group, email guest or team to one conversation. |
| `nextcloud_create_conversation` | Create one conversation and return it with its ``token``. |
| `nextcloud_delete_conversation` | Delete one conversation for everyone (moderators only). |
| `nextcloud_delete_message` | Delete one chat message. |
| `nextcloud_get_conversation` | Read one conversation by its token. |
| `nextcloud_list_conversations` | List the conversations the account takes part in. |
| `nextcloud_list_messages` | Read the chat history of one conversation, newest first. |
| `nextcloud_list_participants` | List the participants of one conversation. |
| `nextcloud_remove_participant` | Remove one attendee (by attendee id from list_participants) from a conversation. |
| `nextcloud_rename_conversation` | Rename one conversation. |
| `nextcloud_send_message` | Post one chat message to a conversation. |


<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/usage.md](docs/usage.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows the operator to restrict the set of exposed tools to prevent blowing up the LLM's context window.

The operator can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to the operator's transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the connector-focused `[mcp]` extra.** Examples use `nextcloud-agent[mcp]` to add
> FastMCP / FastAPI through `scion-sdk[mcp]`; the required Agent Utilities core
> still carries `epistemic-graph[full]`. The `[agent-runtime]` extra additionally
> enables model orchestration.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "nextcloud-agent[mcp]",
        "nextcloud-mcp"
      ],
      "env": {
        "CALENDARTOOL": "True",
        "CONTACTSTOOL": "True",
        "FILESTOOL": "True",
        "NEXTCLOUD_PASSWORD": "your_password",
        "NEXTCLOUD_URL": "https://nextcloud.example.com",
        "NEXTCLOUD_USERNAME": "your_username",
        "SHARESTOOL": "True",
        "ADMINTOOL": "True"
      }
    }
  }
}
```

Runtime references require an alias-aware launcher such as Arbora. Other
launchers must omit those entries and inject the resolved values through their
own runtime secret boundary.

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "nextcloud-agent[mcp]",
        "nextcloud-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "CALENDARTOOL": "True",
        "CONTACTSTOOL": "True",
        "FILESTOOL": "True",
        "NEXTCLOUD_PASSWORD": "your_password",
        "NEXTCLOUD_URL": "https://nextcloud.example.com",
        "NEXTCLOUD_USERNAME": "your_username",
        "SHARESTOOL": "True",
        "ADMINTOOL": "True"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "nextcloud-mcp": {
      "url": "http://localhost:8000/nextcloud-mcp/mcp"
    }
  }
}
```

Run a reviewed container image as a least-privilege stdio child (no
listener or published port):

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  -e MCP_TOOL_MODE=intent \
  -e CALENDARTOOL=True \
  -e CONTACTSTOOL=True \
  -e FILESTOOL=True \
  -e INGESTTOOL=True \
  -e NEXTCLOUD_PASSWORD=your_password \
  -e NEXTCLOUD_URL=https://nextcloud.example.com \
  -e NEXTCLOUD_USERNAME=your_username \
  -e SHARESTOOL=True \
  -e ADMINTOOL=True \
  registry.example.invalid/nextcloud-agent@sha256:<digest> nextcloud-mcp
```

For containerized network HTTP, supply an authenticated TLS ingress (or
direct server TLS), exact `MCP_ALLOWED_HOSTS`, and an exact trusted-proxy
CIDR policy through the operator-owned deployment profile. The generator
does not emit an unauthenticated non-loopback listener.

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`nextcloud-agent` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/nextcloud-agent/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
  The published container runs as a reviewed, least-privilege stdio child with no
  listener or published port by default (see the hardened `docker run` example above).
- **Remote URL** — connect to a server deployed behind Caddy at
  `https://nextcloud-mcp.example.invalid/mcp` using the `"url"` key, or through any
  operator-supplied authenticated HTTPS ingress. Keep its URL, outbound identity
  references, trust profile, and exact `MCP_ALLOWED_HOSTS` in `AgentConfig`.
<!-- END GENERATED: additional-deployment-options -->

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts smoothly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export NEXTCLOUD_URL="your_value"
export NEXTCLOUD_USERNAME="your_value"
export NEXTCLOUD_PASSWORD="your_value"

# Run the agent server
nextcloud-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  nextcloud-agent-mcp:
    image: knucklessg1/nextcloud-agent:mcp
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
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  nextcloud-agent-agent:
    image: knucklessg1/nextcloud-agent:2.3.3
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
      - ENABLE_OTEL=True
    ports:
      - "9016:9016"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9016/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/deployment.md](docs/deployment.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`scion-sdk`](https://github.com/Knuckles-Team/scion-sdk) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Feature | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF` | `vault-ref-to-pk` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY_REF` | `vault-ref-to-sk` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `NEXTCLOUD_URL` | `https://nextcloud.example.com` |  |
| `NEXTCLOUD_USERNAME` | `your_username` |  |
| `NEXTCLOUD_PASSWORD` | secret-injected |  |
| `TLS_PROFILE` | `private-ca` | AgentConfig named transport profile |
| `TLS_PROFILE_REF` | `secret://transport/provider` | Direct runtime profile reference |
| `TLS_PROFILES_REF` | `secret://transport/catalog` | Named runtime profile catalog |
| `FILESTOOL` | `True` |  |
| `ADMINTOOL` | `True` |  |
| `SHARESTOOL` | `True` |  |
| `CALENDARTOOL` | `True` |  |
| `CONTACTSTOOL` | `True` |  |

#### Inherited scion-sdk variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP child auth: `oidc-client-credentials` \| `basic` \| `none` |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET_REF` | `secret://identity/oidc-client-secret` | Runtime secret reference for the OIDC service account |
| `MCP_BASIC_AUTH_USERNAME` | — | HTTP Basic username (`MCP_CLIENT_AUTH=basic`) |
| `MCP_BASIC_AUTH_PASSWORD_REF` | `secret://identity/mcp-basic-password` | Runtime secret reference for HTTP Basic auth (`MCP_CLIENT_AUTH=basic`) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |

_23 package + 16 inherited variable(s). Auto-generated from `.env.example` + the shared scion-sdk set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads, grouped by purpose. See [`.env.example`](.env.example)
for a copy-paste starting point.

### Connection & Credentials
| Variable | Description | Default |
|----------|-------------|---------|
| `NEXTCLOUD_URL` | Base URL of the Nextcloud instance | `https://nextcloud.example.com` |
| `NEXTCLOUD_USERNAME` | Username (basic auth) | — |
| `NEXTCLOUD_PASSWORD` | Password / app password (basic auth) | — |
| `TLS_PROFILE` / `TLS_PROFILE_REF` | AgentConfig transport profile selector; peer verification is mandatory | — |

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |

### Tool toggles
Each tool domain can be disabled via its toggle env var (set to `false`):
`FILESTOOL`, `SHARESTOOL`, `CALENDARTOOL`, `CONTACTSTOOL`, `TALKTOOL`, `DECKTOOL`, `NOTESTOOL`, `OFFICETOOL`, `ACTIVITYTOOL`, `ADMINTOOL`. See the
[Available MCP Tools](#available-mcp-tools) table above.

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

---

## Installation

Pick the extra that matches what the operator want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `nextcloud-agent[mcp]` | Connector-focused MCP server (`scion-sdk[mcp]` — FastMCP/FastAPI + `epistemic-graph[full]`) | The operator only run the **MCP server** (smallest install / image) |
| `nextcloud-agent[agent]` | Agent runtime (`scion-sdk[agent-runtime,logfire]` — model orchestration + `epistemic-graph[full]`) | The operator run the **integrated agent** |
| `nextcloud-agent[all]` | Everything (`mcp` + `agent` + `logfire`) | Development / both surfaces |

```bash
# Connector-focused MCP server (includes the shared graph engine)
uv pip install "nextcloud-agent[mcp]"

# Agent runtime (adds model orchestration to the shared graph engine)
uv pip install "nextcloud-agent[agent]"

# Everything (development)
uv pip install "nextcloud-agent[all]"      # or: python -m pip install "nextcloud-agent[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/nextcloud-agent:mcp` | `--target mcp` | `nextcloud-agent[mcp]` — **connector-focused**, includes `epistemic-graph[full]`; no model-orchestration stack | `nextcloud-mcp` |
| `knucklessg1/nextcloud-agent:2.3.3` | `--target agent` (default) | `nextcloud-agent[agent]` — **agent runtime**, model orchestration + `epistemic-graph[full]` | `nextcloud-agent` |

```bash
docker build --target mcp   -t knucklessg1/nextcloud-agent:mcp    docker/   # connector-focused MCP server
docker build --target agent -t knucklessg1/nextcloud-agent:2.3.3 docker/   # agent runtime
```

`docker/mcp.compose.yml` runs the connector-focused `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:2.3.3`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

Both `[mcp]` and `[agent]` carry the **epistemic-graph** engine through the required
Agent Utilities core dependency (`epistemic-graph[full]`). The `[mcp]` extra keeps
the server connector-focused; `[agent]` also enables model orchestration. Local
deployments can use the bundled engine. For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/nextcloud-agent/) and is
the recommended reference for installation, deployment, and day-to-day operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/nextcloud-agent/installation/) | pip, source, uv, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/nextcloud-agent/deployment/) | run the MCP server and the agent, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/nextcloud-agent/usage/) | the MCP tools, the `NextcloudAPI` client, the agent CLI |
| [Backing Platform](https://knuckles-team.github.io/nextcloud-agent/platform/) | deploy Nextcloud with Docker |
| [Overview](https://knuckles-team.github.io/nextcloud-agent/overview/) | the per-operation tool surface and ecosystem role |
| [Concepts](https://knuckles-team.github.io/nextcloud-agent/concepts/) | concept registry (`CONCEPT:NC-*`) |

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=example&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/example)
![GitHub User's stars](https://img.shields.io/github/stars/example)

---

## Contribute

Contributions are welcome! Please ensure code quality by running local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Ran test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `nextcloud-agent` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx nextcloud-mcp` · or `uv tool install nextcloud-agent` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/nextcloud-agent:2.3.3` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->

<!-- BEGIN scion-sdk-deployment (generated; do not edit between markers) -->

## Deploy with `scion-sdk-deployment`

Provision this package with the consolidated **`scion-sdk-deployment`**
workflow. It selects an installed-package, editable-source, or immutable-container
path; records only runtime secret and TLS-profile references in `AgentConfig`; and
runs doctor, registration, policy, observability, and rollback gates. Ask your agent
to **"deploy `nextcloud-agent` with scion-sdk-deployment"**.

| Install mode | Command |
|------|---------|
| Installed package | `uv tool install "nextcloud-agent[mcp]"`, then run `nextcloud-mcp` |
| Editable source | `uv pip install -e ".[agent]"`, then run `nextcloud-mcp` |
| Immutable container | deploy `knucklessg1/nextcloud-agent:2.3.3` (or a pinned `@sha256:<digest>`) through the operator-selected orchestrator |

The repository embeds no deployment profile, credential value, certificate path, or
environment-specific endpoint. Supply those at runtime through `AgentConfig` and the
configured secret provider.

<!-- END scion-sdk-deployment -->

<!-- GOVERNED-CAPABILITY:START -->
## Governed capability contract

This package ships a compact canonical skill surface with specialist procedures
kept as referenced workflows. The current MCP tools, skill metadata,
`connector_manifest.yml`, ontology, mappings, shapes, fixtures, migrations,
tool-schema fingerprints, and certification metadata form one versioned
capability contract. Validate them together; do not rely on stale tool names or
historical per-task skill wrappers.

Runtime endpoints, credentials, certificate trust, tenant identity, retention,
and observability policy are deployment inputs and are never packaged values.
See [Configuration, trust, and privacy](docs/configuration.md) before enabling a
network transport, connector ingestion, Arbora delegation, or trace export.
<!-- GOVERNED-CAPABILITY:END -->
