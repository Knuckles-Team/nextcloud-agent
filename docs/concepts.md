# Concept Registry — nextcloud-agent

> **Prefix**: `CONCEPT:NC-*`
> **Version**: 0.15.0
> **Bridge**: [`CONCEPT:ECO-4.0`](../../agent-utilities/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:NC-001` | Calendar Management | MCP tool domain `calendar` — Action-routed dynamic tool registration |
| `CONCEPT:NC-002` | Contact Management | MCP tool domain `contacts` — Action-routed dynamic tool registration |
| `CONCEPT:NC-003` | File Management | MCP tool domain `files` — Action-routed dynamic tool registration |
| `CONCEPT:NC-004` | Sharing & Collaboration | MCP tool domain `sharing` — Action-routed dynamic tool registration |
| `CONCEPT:NC-005` | User & Identity Management | MCP tool domain `user` — Action-routed dynamic tool registration |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:ECO-4.0` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:ORCH-1.2` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:OS-5.1` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:OS-5.2` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:OS-5.3` | Guardrail Engine | agent-utilities |
| `CONCEPT:OS-5.4` | Audit Logging | agent-utilities |
| `CONCEPT:KG-2.0` | Knowledge Graph Core | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:ECO-4.0` (Unified Toolkit Ingestion). The `nextcloud_agent` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all NC-* concepts.
