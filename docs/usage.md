# Usage — API / CLI / MCP

`nextcloud-agent` exposes the same capability three ways: as **MCP tools** an agent
calls, as a **Python API** (`NextcloudAPI`) the operator import, and as a **graph-agent CLI**.
The complete tool surface and ecosystem role are in [Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers one typed tool per operation,
`nextcloud_<operation>`, in ten domains. Each domain is gated by a toggle (all default
`True`): `FILESTOOL`, `SHARESTOOL`, `CALENDARTOOL`, `CONTACTSTOOL`, `TALKTOOL`,
`DECKTOOL`, `NOTESTOOL`, `OFFICETOOL`, `ACTIVITYTOOL`, `ADMINTOOL`. The full list is in
the README. Example requests:

- *"List the files in my Documents folder"* → `nextcloud_list_files(path="Documents")`
- *"Share Team/launch/poster.png with a link that expires in two weeks"* →
  `nextcloud_create_share(path=..., share_type="public_link", expire_date=...)`
- *"Book a review on Friday at 3pm with Ana"* → `nextcloud_create_event(..., attendees=[...])`
  (the invitation stays a draft unless `send_invitations=true`)
- *"Post the update in the launch channel"* → `nextcloud_send_message(token, message)`

## As a Python API

`NextcloudAPI` is a `requests`-based facade over the Nextcloud WebDAV, CalDAV,
CardDAV, and OCS surfaces, composed from per-domain sub-clients.

```python
from nextcloud_agent.api_client import NextcloudAPI

api = NextcloudAPI(
    base_url="https://nextcloud.example.com",
    username="your-user",
    password="your-app-password",
)

# Reads
files = api.list_files("/Documents")  # WebDAV listing
calendars = api.list_calendars()  # CalDAV calendars
events = api.list_events(calendars[0]["url"])  # CalDAV events
contacts = api.list_contacts(api.list_address_books()[0]["url"])  # CardDAV contacts
shares = api.list_shares()  # OCS shares
```

Build a client straight from the environment with the `get_client` context manager,
which reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, and `NEXTCLOUD_PASSWORD`.
TLS trust is resolved through the shared Meristem TLS profile contract;
select `NEXTCLOUD_TLS_PROFILE` or supply a runtime CA-bundle secret reference
without changing application code:

```python
from nextcloud_agent.auth import get_client

with get_client() as api:  # reads NEXTCLOUD_* from the environment / .env
    info = api.get_user_info()
```

`get_client` raises a clear authentication error when the supplied credentials are
not valid for the target instance, and remains inactive when credentials are absent.

### Writes

```python
api.create_folder("/Reports/2026")
api.write_file("/Reports/2026/summary.txt", "...")
api.create_share("/Reports/2026/summary.txt", share_type="public_link", expire_date="2026-12-31")
api.create_event(calendars[0]["url"], summary="Standup", start="2026-06-10T09:00:00Z")
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
