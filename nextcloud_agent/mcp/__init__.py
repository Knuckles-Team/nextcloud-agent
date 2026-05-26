"""MCP tool registration modules for nextcloud-agent.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from nextcloud_agent.mcp.mcp_calendar import register_calendar_tools
from nextcloud_agent.mcp.mcp_contacts import register_contacts_tools
from nextcloud_agent.mcp.mcp_files import register_files_tools
from nextcloud_agent.mcp.mcp_sharing import register_sharing_tools
from nextcloud_agent.mcp.mcp_user import register_user_tools

__all__ = [
    "register_calendar_tools",
    "register_contacts_tools",
    "register_files_tools",
    "register_sharing_tools",
    "register_user_tools",
]
