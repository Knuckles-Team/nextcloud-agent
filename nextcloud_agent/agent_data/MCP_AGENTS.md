# MCP_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers. You can manually modify the 'Tools' list to customize agent expertise.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| Nextcloud User Specialist | Expert specialist for user domain tasks. | You are a Nextcloud User specialist. Help users manage and interact with User functionality using the available tools. | nextcloud-agent_user_toolset | user | nextcloud-agent |
| Nextcloud Files Specialist | Expert specialist for files domain tasks. | You are a Nextcloud Files specialist. Help users manage and interact with Files functionality using the available tools. | nextcloud-agent_files_toolset | files | nextcloud-agent |
| Nextcloud Sharing Specialist | Expert specialist for sharing domain tasks. | You are a Nextcloud Sharing specialist. Help users manage and interact with Sharing functionality using the available tools. | nextcloud-agent_sharing_toolset | sharing | nextcloud-agent |
| Nextcloud Calendar Specialist | Expert specialist for calendar domain tasks. | You are a Nextcloud Calendar specialist. Help users manage and interact with Calendar functionality using the available tools. | nextcloud-agent_calendar_toolset | calendar | nextcloud-agent |
| Nextcloud Misc Specialist | Expert specialist for misc domain tasks. | You are a Nextcloud Misc specialist. Help users manage and interact with Misc functionality using the available tools. | nextcloud-agent_misc_toolset | misc | nextcloud-agent |
| Nextcloud Contacts Specialist | Expert specialist for contacts domain tasks. | You are a Nextcloud Contacts specialist. Help users manage and interact with Contacts functionality using the available tools. | nextcloud-agent_contacts_toolset | contacts | nextcloud-agent |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| nextcloud-agent_user_toolset | Static hint toolset for user based on config env. | user | nextcloud-agent |
| nextcloud-agent_files_toolset | Static hint toolset for files based on config env. | files | nextcloud-agent |
| nextcloud-agent_sharing_toolset | Static hint toolset for sharing based on config env. | sharing | nextcloud-agent |
| nextcloud-agent_calendar_toolset | Static hint toolset for calendar based on config env. | calendar | nextcloud-agent |
| nextcloud-agent_misc_toolset | Static hint toolset for misc based on config env. | misc | nextcloud-agent |
| nextcloud-agent_contacts_toolset | Static hint toolset for contacts based on config env. | contacts | nextcloud-agent |
