"""Talk conversations and chat over the spreed OCS API (CONN-NEXTCLOUD-AGENT-R009)."""

from typing import Literal

from nextcloud_agent.api.api_client_base import BaseApiClient

_ROOM = "apps/spreed/api/v4/room"
_CHAT = "apps/spreed/api/v1/chat"
#: Talk conversation types by name.
ROOM_TYPES = {"one_to_one": 1, "group": 2, "public": 3}


def _token(token: str) -> str:
    if not token or not token.isalnum() or len(token) > 64:
        raise ValueError("conversation token is invalid")
    return token


class NextcloudTalk(BaseApiClient):
    """Talk: conversations, participants and chat messages."""

    def list_conversations(self) -> list[dict]:
        """List the conversations the account takes part in.

        GET /ocs/v2.php/apps/spreed/api/v4/room
        """
        return self.ocs_request("GET", _ROOM) or []

    def get_conversation(self, token: str) -> dict:
        """Read one conversation by its token.

        GET /ocs/v2.php/apps/spreed/api/v4/room/{token}
        """
        return self.ocs_request("GET", f"{_ROOM}/{_token(token)}")

    def create_conversation(
        self,
        name: str,
        room_type: Literal["group", "public", "one_to_one"] = "group",
        invite: str | None = None,
    ) -> dict:
        """Create one conversation and return it with its ``token``.

        POST /ocs/v2.php/apps/spreed/api/v4/room

        ``group`` is private (members only, no link); ``public`` can be joined
        by link; ``one_to_one`` needs ``invite`` (a user id).
        """
        data: dict = {"roomType": ROOM_TYPES[room_type], "roomName": name}
        if invite:
            data["invite"] = invite
            data["source"] = "users"
        return self.ocs_request("POST", _ROOM, data=data)

    def rename_conversation(self, token: str, name: str) -> bool:
        """Rename one conversation.

        PUT /ocs/v2.php/apps/spreed/api/v4/room/{token}
        """
        self.ocs_request("PUT", f"{_ROOM}/{_token(token)}", data={"roomName": name})
        return True

    def delete_conversation(self, token: str) -> bool:
        """Delete one conversation for everyone (moderators only).

        DELETE /ocs/v2.php/apps/spreed/api/v4/room/{token}
        """
        self.ocs_request("DELETE", f"{_ROOM}/{_token(token)}")
        return True

    def list_participants(self, token: str) -> list[dict]:
        """List the participants of one conversation.

        GET /ocs/v2.php/apps/spreed/api/v4/room/{token}/participants
        """
        return self.ocs_request("GET", f"{_ROOM}/{_token(token)}/participants") or []

    def add_participant(
        self,
        token: str,
        participant: str,
        source: Literal["users", "groups", "emails", "circles"] = "users",
    ) -> dict:
        """Add a user, group, email guest or team to one conversation.

        POST /ocs/v2.php/apps/spreed/api/v4/room/{token}/participants
        """
        return self.ocs_request(
            "POST",
            f"{_ROOM}/{_token(token)}/participants",
            data={"newParticipant": participant, "source": source},
        )

    def remove_participant(self, token: str, attendee_id: int) -> bool:
        """Remove one attendee (by attendee id from list_participants) from a conversation.

        DELETE /ocs/v2.php/apps/spreed/api/v4/room/{token}/attendees
        """
        self.ocs_request(
            "DELETE",
            f"{_ROOM}/{_token(token)}/attendees",
            data={"attendeeId": int(attendee_id)},
        )
        return True

    def list_messages(
        self, token: str, limit: int = 50, last_known_message_id: int | None = None
    ) -> list[dict]:
        """Read the chat history of one conversation, newest first.

        GET /ocs/v2.php/apps/spreed/api/v1/chat/{token}
        """
        params: dict = {"lookIntoFuture": 0, "limit": max(1, min(int(limit), 200))}
        if last_known_message_id:
            params["lastKnownMessageId"] = int(last_known_message_id)
        return self.ocs_request("GET", f"{_CHAT}/{_token(token)}", params=params) or []

    def send_message(
        self,
        token: str,
        message: str,
        reply_to: int | None = None,
        silent: bool = False,
    ) -> dict:
        """Post one chat message to a conversation.

        POST /ocs/v2.php/apps/spreed/api/v1/chat/{token}
        """
        data: dict = {"message": message}
        if reply_to:
            data["replyTo"] = int(reply_to)
        if silent:
            data["silent"] = "true"
        return self.ocs_request("POST", f"{_CHAT}/{_token(token)}", data=data)

    def delete_message(self, token: str, message_id: int) -> dict:
        """Delete one chat message.

        DELETE /ocs/v2.php/apps/spreed/api/v1/chat/{token}/{message_id}
        """
        return self.ocs_request("DELETE", f"{_CHAT}/{_token(token)}/{int(message_id)}")
