# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R001` | **Default content.** `pyproject.toml` declares `nextcloud-mcp` as the single `scion_sdk.connector` entry point. The server lists the packaged `skill://` and `ontology://` resources and the packaged prompts with no `content=` argument. Caller chain: `nextcloud-mcp` in `pyproject.toml` runs `scion_sdk.cli:serve`, which builds the server from the `scion_sdk.connector` entry point `nextcloud_agent:connector` (`nextcloud_agent/__init__.py:6`); the SDK `Connector` then lists the packaged skills, ontology and prompts. Shape: `pyproject.toml` entry point `nextcloud-mcp` plus the `connector = Connector(.)` object in `nextcloud_agent/__init__.py`; the packaged `skills/`, `ontology/` and `prompts/` directories of `nextcloud_agent` are reused as is, with no connector-local registration code. Proving test (planned): `tests/test_default_content_surfaces.py::test_default_content_surfaces`. Proving test (planned): `tests/test_default_content_surfaces.py::test_default_content_surfaces`. Proving test (planned): `tests/test_default_content_surfaces.py::test_default_content_surfaces`. Proving test (planned): `tests/test_default_content_surfaces.py::test_default_content_surfaces`. | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |
