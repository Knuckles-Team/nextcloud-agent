# Usage — API / CLI / MCP

`nextcloud-agent` exposes the same capability three ways: as **MCP tools** an agent
calls, as a **Python API** (`NextcloudAPI`) you import, and as a **graph-agent CLI**.
The complete tool surface and ecosystem role are in [Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers five condensed, action-routed
tool modules. Each module is gated by a toggle environment variable (all default
`True`):

| Tool | Toggle | Actions |
|---|---|---|
| `nextcloud_files` | `FILESTOOL` | `list_files`, `read_file`, `write_file`, `create_folder`, `delete_item`, `move_item`, `copy_item`, `get_properties` |
| `nextcloud_user` | `USERTOOL` | `get_user_info` |
| `nextcloud_sharing` | `SHARINGTOOL` | `list_shares`, `create_share`, `delete_share` |
| `nextcloud_calendar` | `CALENDARTOOL` | `list_calendars`, `list_calendar_events`, `create_calendar_event` |
| `nextcloud_contacts` | `CONTACTSTOOL` | `list_address_books`, `list_contacts`, `create_contact` |

Each tool takes an `action` plus a `params_json` JSON string of arguments. Example
agent prompts that map onto these tools:

- *"List the files in my Documents folder"* → `nextcloud_files` (`list_files`)
- *"Share /reports/q3.pdf with a public link"* → `nextcloud_sharing` (`create_share`)
- *"What events are on my work calendar this week?"* → `nextcloud_calendar` (`list_calendar_events`)

## As a Python API

`NextcloudAPI` is a `requests`-based facade over the Nextcloud WebDAV, CalDAV,
CardDAV, and OCS surfaces, composed from per-domain sub-clients (files, calendar,
contacts).

```python
from nextcloud_agent.api_client import NextcloudAPI

api = NextcloudAPI(
    base_url="https://nextcloud.example.com",
    username="your-user",
    password="your-app-password",
)

# Reads
files = api.list_files("/Documents")           # WebDAV listing
calendars = api.list_calendars()               # CalDAV calendars
events = api.list_calendar_events("personal")  # CalDAV events
contacts = api.list_contacts()                 # CardDAV contacts
shares = api.list_shares()                      # OCS shares
```

Build a client straight from the environment with the `get_client` context manager,
which reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, and `NEXTCLOUD_PASSWORD`.
TLS trust is resolved through the shared Agent Utilities TLS profile contract;
select `NEXTCLOUD_TLS_PROFILE` or supply a runtime CA-bundle secret reference
without changing application code:

```python
from nextcloud_agent.auth import get_client

with get_client() as api:                      # reads NEXTCLOUD_* from the environment / .env
    info = api.get_user_info()
```

`get_client` raises a clear authentication error when the supplied credentials are
not valid for the target instance, and remains inactive when credentials are absent.

### Writes

```python
api.create_folder("/Reports/2026")
api.write_file("/Reports/2026/summary.txt", b"...")
api.create_share("/Reports/2026/summary.txt", share_type=3)   # public link
api.create_calendar_event("personal", summary="Standup", start="2026-06-10T09:00:00")
```

## As an agent CLI

The integrated **Pydantic-AI graph agent** (`nextcloud-agent`) calls the MCP server
and exposes an optional web UI. Point it at a running MCP server and a model
provider:

```bash
export MCP_URL=http://localhost:8000/mcp
export NEXTCLOUD_URL=https://nextcloud.example.com
export NEXTCLOUD_USERNAME=your-user
export NEXTCLOUD_PASSWORD=your-app-password

nextcloud-agent --provider openai --model-id gpt-4o
```

A lightweight router node classifies each query into a domain (files, calendar,
contacts, sharing, user) and dispatches it to a focused executor, keeping the LLM
context compact. See [Deployment](deployment.md#run-the-agent-server) for running the
agent as a long-lived service.
