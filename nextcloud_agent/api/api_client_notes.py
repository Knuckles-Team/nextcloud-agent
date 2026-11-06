"""Notes over the Notes REST API (CONN-NEXTCLOUD-AGENT-R011)."""

from nextcloud_agent.api.api_client_base import BaseApiClient

_NOTES = "notes/api/v1/notes"


class NextcloudNotes(BaseApiClient):
    """Notes: markdown notes stored as files in the account's Notes folder."""

    def list_notes(self, category: str | None = None) -> list[dict]:
        """List the account's notes, optionally of one category.

        GET /index.php/apps/notes/api/v1/notes
        """
        params = {"category": category} if category else {}
        return self._app_json("GET", _NOTES, params=params) or []

    def get_note(self, note_id: int) -> dict:
        """Read one note with its content.

        GET /index.php/apps/notes/api/v1/notes/{note_id}
        """
        return self._app_json("GET", f"{_NOTES}/{int(note_id)}")

    def create_note(self, title: str, content: str = "", category: str = "") -> dict:
        """Create one note.

        POST /index.php/apps/notes/api/v1/notes
        """
        return self._app_json(
            "POST",
            _NOTES,
            json={"title": title, "content": content, "category": category},
        )

    def update_note(
        self,
        note_id: int,
        title: str | None = None,
        content: str | None = None,
        category: str | None = None,
        favorite: bool | None = None,
    ) -> dict:
        """Change one note's title, content, category or favorite flag.

        PUT /index.php/apps/notes/api/v1/notes/{note_id}
        """
        fields = {
            "title": title,
            "content": content,
            "category": category,
            "favorite": favorite,
        }
        body = {k: v for k, v in fields.items() if v is not None}
        if not body:
            raise ValueError("no note field to update")
        return self._app_json("PUT", f"{_NOTES}/{int(note_id)}", json=body)

    def delete_note(self, note_id: int) -> dict:
        """Delete one note.

        DELETE /index.php/apps/notes/api/v1/notes/{note_id}
        """
        return self._app_json("DELETE", f"{_NOTES}/{int(note_id)}")
