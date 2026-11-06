# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Quickstart

The shortest proof builds the server from the connector entry point and checks the served tools against the packaged ontology, prompts and skills.

## Scenario

1. Build the server with `scion_sdk.connector.build_server("nextcloud_agent")` and list its tools.
2. Parse `nextcloud_agent/ontology/nextcloud.ttl`: every class rolls up to core, and the `cp:Tool` ids equal the served tool names.
3. Load each prompt and skill: every tool it names is served, and every prompt carries data locations, guardrails and outputs.

## Commands

```text
uv run pytest tests/test_content_surface.py
```
