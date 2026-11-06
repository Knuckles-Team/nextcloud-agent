# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py::test_default_content_surfaces` (planned) | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |
| T-002 | `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002` | `tests/test_content_surface.py::test_ontology_rolls_up_to_core_and_records_every_tool` | The ontology records the surface, contracts and data locations, rolled up to core. |
| T-003 | `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R003` | `tests/test_content_surface.py::test_prompts_are_agent_definitions_over_real_tools_and_skills` | Each prompt is an agent definition. |
| T-004 | `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R004` | `tests/test_content_surface.py::test_skills_and_contracts_name_only_served_tools` | A skill per domain plus the asset-publishing workflow. |

## Negative and boundary cases

Each proving test above includes the refusal or error case that its row names. A vendor outage or an unauthorised call must return the SDK typed error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.
