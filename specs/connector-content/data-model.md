# CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT — Data model

This file describes the ontology classes, individuals and content files that the rows of this spec add or reuse.

## Entities

### Pack classes

- Owner: `nextcloud_agent/ontology/nextcloud.ttl`, row `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002`.
- `:Share` under core `:FileShare`; `:FileVersion` and `:TrashItem` under `:File`; `:OfficeDocument`, `:ContactCard` and `:NextcloudNote` under `:Document`; `:NextcloudCalendar`, `:AddressBook`, `:DeckBoard` and `:DeckStack` under `:Collection`; `:DeckCard` and `:ToolUseContract` under `:Task`; `:TalkConversation` under `:MessageChannel`; `:TalkMessage` under `:ChatMessage`; `:ActivityEvent` and `:NextcloudNotification` under `:Event`; `:NextcloudApp` under `:Application`; `:NextcloudDataLocation` under `:StorageContainer`.
- Core classes reused, not redeclared: `:File`, `:Folder`, `:CalendarEvent`, `:PersonalTask`, `:IdentityUser`, `:IdentityGroup`, `:TaskStep`.

### Surface individuals (`nc:` namespace)

- `nc:connector` (`cp:Connector`), one `nc:feature_<domain>` (`cp:Feature`) per domain, one `nc:tool_<operation>` (`cp:Tool`) per served tool with `cp:operatesOn`, `cp:produces` and `cp:providesCapability`, one `nc:skill_<name>` (`cp:Skill`) and one `nc:prompt_<name>` (`cp:Prompt`) per packaged skill and prompt.
- Links: `cp:realizedBy` (feature to tool, skill, prompt), `:usesTool` (skill or prompt to tool), `:promptUsesSkill` (prompt to skill), `:skillFollowsContract` (skill to contract).

### Tool-use contracts

- `nc:contract_<id>` (`:ToolUseContract`) `:hasStep` ordered `:TaskStep` nodes; each step carries `:stepIndex`, `:usesTool`, `:stepParameters`, `:stepGuidance`, `:needsSkill`, and `:precedes` the next.

### Data locations

- `nc:location_<family>` (`:NextcloudDataLocation`) with `:locationTemplate`, `:accessProtocol`, `:holdsClass`, `:routedThrough nc:connector` and, when declared, `:ingestSource` naming a source of `nextcloud_agent/connector_sources.py`.

### Prompt files

- `nextcloud_agent/prompts/<name>.json`, row `CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R003`: the existing prompt schema with `instructions.data_locations`, `instructions.tool_use_contracts`, `instructions.guardrails` and `instructions.outputs`.
