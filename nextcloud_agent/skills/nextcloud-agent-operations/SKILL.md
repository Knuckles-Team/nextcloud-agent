---
name: nextcloud-agent-operations
skill_type: skill
description: >-
  Operate nextcloud-agent through its governed MCP and Arbora capabilities, covering files, sharing, calendar, tasks, contacts, Talk, Deck, notes, office documents, activity, administration, asset publishing and ingestion. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.
---

# Nextcloud Agent Operations

Use the provider's governed MCP tools through Arbora delegation.

## Workflow

1. Establish the verified GraphSession and tenant before discovery or retrieval.
2. Discover the current tool surface (one `nextcloud_<operation>` tool per operation); never assume a stale tool name or schema.
3. Prefer read-only inspection first. For changes, present impact and use the provider's
   dry-run or preview mode when available.
4. Execute mutations as fenced WorkItems so retries remain idempotent and auditable.
5. Ingest source data only through the signed connector preset and ChangeEnvelope path.
6. Verify the durable result and its trace/evidence before reporting completion.

## Steps

1. Discover the live tools with arbora `find`(intent="nextcloud"). Verify the tool names start with `nextcloud_` and are not assumed from memory.
2. Read the matching procedure in `references/catalog.md`. Verify the workflow and parameters it lists match the request.
3. Inspect state first with a read tool such as `nextcloud_list_files`(path) or `nextcloud_get_user_info`(). Verify the target exists before changing it.
4. Run one change at a time after the person approves any destructive or externally visible action. Verify the result with the matching read tool such as `nextcloud_get_properties`(path).
5. Report only the evidence returned by the tools, such as the `nextcloud_list_activities`(limit=5) entries. Verify no credential, hostname or personal identifier is included.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| 401 or connection error from a `nextcloud_*` tool | `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` or the app password in `NEXTCLOUD_PASSWORD` is missing or wrong | Set the three variables for the server, and verify the connection by calling `nextcloud_get_user_info` |
| Destructive tool asks for confirmation | confirm=true was not passed because the person has not agreed | Ask the person to agree, pass confirm=true on that single call, and verify the result with the matching read tool |
| Tool name not found | The tool surface changed | Re-run `find`(intent="nextcloud") and use a current tool name |

## Safety contract

- Never persist credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Resolve TLS trust and verification from environment/configuration; never hardcode bypasses.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces policy-scoped and privacy-sanitized.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.
