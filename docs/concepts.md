# Concept Registry — nextcloud-agent

> **Prefix**: `CONCEPT:NC-*`
> **Version**: 0.15.0
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/agent-connector-sdk/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:NC-OS.governance.nc` | Calendar Management | MCP tool domain `calendar` — Action-routed dynamic tool registration |
| `CONCEPT:NC-OS.governance.nc-2` | Contact Management | MCP tool domain `contacts` — Action-routed dynamic tool registration |
| `CONCEPT:NC-OS.governance.nc-3` | File Management | MCP tool domain `files` — Action-routed dynamic tool registration |
| `CONCEPT:NC-OS.governance.nc-4` | Sharing & Collaboration | MCP tool domain `sharing` — Action-routed dynamic tool registration |
| `CONCEPT:NC-OS.governance.nc-5` | User & Identity Management | MCP tool domain `user` — Action-routed dynamic tool registration |

## Cross-Project References (from agent-connector-sdk)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:AU-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | agent-connector-sdk |
| `CONCEPT:AU-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | agent-connector-sdk |
| `CONCEPT:AU-OS.config.secrets-authentication` | Prompt Injection Defense | agent-connector-sdk |
| `CONCEPT:AU-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | agent-connector-sdk |
| `CONCEPT:AU-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | agent-connector-sdk |
| `CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | agent-connector-sdk |
| `CONCEPT:AU-KG.query.object-graph-mapper` | Knowledge Graph Core | agent-connector-sdk |

## Synergy with agent-connector-sdk

This project integrates with `agent-connector-sdk` via `CONCEPT:AU-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `nextcloud_agent` MCP server registers its tools with the agent-connector-sdk FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all NC-* concepts.
