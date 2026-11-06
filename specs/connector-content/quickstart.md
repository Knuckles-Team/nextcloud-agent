# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Quickstart

The shortest proof builds the server from the connector entry point and checks the served tools against the packaged ontology, prompts and skills.

## Scenario

1. Build the server with `scion_sdk.connector.build_server("nextcloud_agent")` and list its tools.
2. Parse `nextcloud_agent/ontology/nextcloud.ttl`: every class rolls up to core. Read `nextcloud_agent/connector_contracts.py`: every served tool is a contract step, and every prompt and skill is linked.
3. Load each prompt and skill: every tool it names is served, and every prompt carries data locations, guardrails and outputs.

## Commands

```text
uv run pytest tests/test_content_surface.py
```
