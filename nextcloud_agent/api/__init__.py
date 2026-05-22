from nextcloud_agent.api.api_client_calendar import Api as CalendarApi
from nextcloud_agent.api.api_client_contacts import Api as ContactsApi
from nextcloud_agent.api.api_client_files import Api as FilesApi


class Api(FilesApi, CalendarApi, ContactsApi):
    """Unified Nextcloud API Client combining decomposed sub-clients."""
