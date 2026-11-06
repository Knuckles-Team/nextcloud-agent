"""Deck boards, stacks and cards over the Deck REST API (CONN-NEXTCLOUD-AGENT-R010)."""

from nextcloud_agent.api.api_client_base import BaseApiClient

_DECK = "deck/api/v1.0/boards"


class NextcloudDeck(BaseApiClient):
    """Deck: kanban boards, their stacks (columns) and cards."""

    def list_boards(self) -> list[dict]:
        """List the Deck boards the account can see.

        GET /index.php/apps/deck/api/v1.0/boards
        """
        return self._app_json("GET", _DECK) or []

    def get_board(self, board_id: int) -> dict:
        """Read one board with its labels and members.

        GET /index.php/apps/deck/api/v1.0/boards/{board_id}
        """
        return self._app_json("GET", f"{_DECK}/{int(board_id)}")

    def create_board(self, title: str, color: str = "0082c9") -> dict:
        """Create one board; ``color`` is a hex RGB value without ``#``.

        POST /index.php/apps/deck/api/v1.0/boards
        """
        return self._app_json("POST", _DECK, json={"title": title, "color": color})

    def delete_board(self, board_id: int) -> dict:
        """Delete one board (Deck keeps it restorable for a while).

        DELETE /index.php/apps/deck/api/v1.0/boards/{board_id}
        """
        return self._app_json("DELETE", f"{_DECK}/{int(board_id)}")

    def list_stacks(self, board_id: int) -> list[dict]:
        """List the stacks (columns) of one board, each with its cards.

        GET /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks
        """
        return self._app_json("GET", f"{_DECK}/{int(board_id)}/stacks") or []

    def create_stack(self, board_id: int, title: str, order: int = 0) -> dict:
        """Create one stack (column) on a board.

        POST /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks
        """
        return self._app_json(
            "POST",
            f"{_DECK}/{int(board_id)}/stacks",
            json={"title": title, "order": int(order)},
        )

    def delete_stack(self, board_id: int, stack_id: int) -> dict:
        """Delete one stack and its cards.

        DELETE /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}
        """
        return self._app_json(
            "DELETE", f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}"
        )

    def create_card(
        self,
        board_id: int,
        stack_id: int,
        title: str,
        description: str = "",
        duedate: str | None = None,
        order: int = 0,
    ) -> dict:
        """Create one card in a stack; ``duedate`` is ISO 8601.

        POST /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}/cards
        """
        body: dict = {
            "title": title,
            "type": "plain",
            "order": int(order),
            "description": description,
        }
        if duedate:
            body["duedate"] = duedate
        return self._app_json(
            "POST", f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}/cards", json=body
        )

    def get_card(self, board_id: int, stack_id: int, card_id: int) -> dict:
        """Read one card.

        GET /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}/cards/{card_id}
        """
        return self._app_json(
            "GET",
            f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}/cards/{int(card_id)}",
        )

    def update_card(
        self,
        board_id: int,
        stack_id: int,
        card_id: int,
        title: str | None = None,
        description: str | None = None,
        duedate: str | None = None,
    ) -> dict:
        """Change one card's title, description or due date.

        PUT /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}/cards/{card_id}
        """
        card = self.get_card(board_id, stack_id, card_id)
        owner = card.get("owner")
        body = {
            "title": title if title is not None else card.get("title"),
            "description": description
            if description is not None
            else card.get("description", ""),
            "type": card.get("type", "plain"),
            "order": card.get("order", 0),
            "owner": owner.get("uid") if isinstance(owner, dict) else owner,
            "duedate": duedate if duedate is not None else card.get("duedate"),
        }
        return self._app_json(
            "PUT",
            f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}/cards/{int(card_id)}",
            json=body,
        )

    def move_card(
        self, board_id: int, card_id: int, target_stack_id: int, order: int = 0
    ) -> dict:
        """Move one card to a stack (column) of the board, at position ``order``.

        PUT /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{target_stack_id}/cards/{card_id}/reorder
        """
        return self._app_json(
            "PUT",
            f"{_DECK}/{int(board_id)}/stacks/{int(target_stack_id)}/cards/{int(card_id)}/reorder",
            json={"order": int(order)},
        )

    def assign_card_user(
        self, board_id: int, stack_id: int, card_id: int, user_id: str
    ) -> dict:
        """Assign one user to a card.

        PUT /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}/cards/{card_id}/assignUser
        """
        return self._app_json(
            "PUT",
            f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}/cards/{int(card_id)}/assignUser",
            json={"userId": user_id},
        )

    def delete_card(self, board_id: int, stack_id: int, card_id: int) -> dict:
        """Delete one card.

        DELETE /index.php/apps/deck/api/v1.0/boards/{board_id}/stacks/{stack_id}/cards/{card_id}
        """
        return self._app_json(
            "DELETE",
            f"{_DECK}/{int(board_id)}/stacks/{int(stack_id)}/cards/{int(card_id)}",
        )
