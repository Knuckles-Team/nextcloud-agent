from nextcloud_agent.api.api_client_activity import NextcloudActivity
from nextcloud_agent.api.api_client_admin import NextcloudAdmin
from nextcloud_agent.api.api_client_calendar import NextcloudCalendar
from nextcloud_agent.api.api_client_contacts import NextcloudContacts
from nextcloud_agent.api.api_client_deck import NextcloudDeck
from nextcloud_agent.api.api_client_files import NextcloudFiles
from nextcloud_agent.api.api_client_notes import NextcloudNotes
from nextcloud_agent.api.api_client_office import NextcloudOffice
from nextcloud_agent.api.api_client_shares import NextcloudShares
from nextcloud_agent.api.api_client_talk import NextcloudTalk


class Api(
    NextcloudFiles,
    NextcloudShares,
    NextcloudCalendar,
    NextcloudContacts,
    NextcloudTalk,
    NextcloudDeck,
    NextcloudNotes,
    NextcloudActivity,
    NextcloudAdmin,
    NextcloudOffice,
):
    """Unified Nextcloud API client combining the per-domain sub-clients."""
