"""Activity stream and notifications over OCS (CONN-NEXTCLOUD-AGENT-R012)."""

import requests

from nextcloud_agent.api.api_client_base import BaseApiClient


class NextcloudActivity(BaseApiClient):
    """The activity stream (who changed what) and the account's notifications."""

    def list_activities(
        self,
        limit: int = 50,
        since: int | None = None,
        object_type: str | None = None,
        object_id: int | None = None,
    ) -> list[dict]:
        """Read the activity stream, newest first; narrow to one object
        (``object_type="files"`` with a file id) when given.

        GET /ocs/v2.php/apps/activity/api/v2/activity
        """
        params: dict = {"limit": max(1, min(int(limit), 200))}
        if since:
            params["since"] = int(since)
        endpoint = "apps/activity/api/v2/activity"
        if object_type and object_id is not None:
            endpoint += "/filter"
            params.update({"object_type": object_type, "object_id": int(object_id)})
        try:
            return self.ocs_request("GET", endpoint, params=params) or []
        except requests.TooManyRedirects:
            # The activity API answers 304 Not Modified when nothing is newer.
            return []

    def list_notifications(self) -> list[dict]:
        """List the account's current notifications.

        GET /ocs/v2.php/apps/notifications/api/v2/notifications
        """
        return self.ocs_request("GET", "apps/notifications/api/v2/notifications") or []

    def dismiss_notification(self, notification_id: int) -> bool:
        """Dismiss one notification.

        DELETE /ocs/v2.php/apps/notifications/api/v2/notifications/{notification_id}
        """
        self.ocs_request(
            "DELETE", f"apps/notifications/api/v2/notifications/{int(notification_id)}"
        )
        return True
