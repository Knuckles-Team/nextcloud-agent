"""Nextcloud graph configuration — tag prompts and env var mappings.

This is the only file needed to enable graph mode for this agent.
Provides TAG_PROMPTS and TAG_ENV_VARS for create_graph_agent_server().
"""

# ── Tag → System Prompt Mapping ──────────────────────────────────────
TAG_PROMPTS: dict[str, str] = {
    "calendar": (
        "You are a Nextcloud Calendar specialist. Help users manage and interact with Calendar functionality using the available tools."
    ),
    "contacts": (
        "You are a Nextcloud Contacts specialist. Help users manage and interact with Contacts functionality using the available tools."
    ),
    "files": (
        "You are a Nextcloud Files specialist. Help users manage and interact with Files functionality using the available tools."
    ),
    "sharing": (
        "You are a Nextcloud Sharing specialist. Help users manage and interact with Sharing functionality using the available tools."
    ),
    "user": (
        "You are a Nextcloud User specialist. Help users manage and interact with User functionality using the available tools."
    ),
}


# ── Tag → Environment Variable Mapping ────────────────────────────────
TAG_ENV_VARS: dict[str, str] = {
    "calendar": "CALENDARTOOL",
    "contacts": "CONTACTSTOOL",
    "files": "FILESTOOL",
    "sharing": "SHARINGTOOL",
    "user": "USERTOOL",
}
