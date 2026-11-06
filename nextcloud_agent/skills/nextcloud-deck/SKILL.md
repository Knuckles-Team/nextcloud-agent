---
name: nextcloud-deck
skill_type: skill
description: >-
  Run Nextcloud Deck kanban boards: list, read, create and delete boards; list, create and delete stacks; create, read, update, move, assign and delete cards. Use when the agent must track work items through stages (to do, doing, done). Do NOT use for personal to-dos (nextcloud-tasks).
license: MIT
tags: [nextcloud, deck, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Deck

Run Nextcloud Deck kanban boards: list, read, create and delete boards; list, create and delete stacks; create, read, update, move, assign and delete cards.

## When to use
- Add work: `nextcloud_list_boards`, `nextcloud_list_stacks(board_id)`, `nextcloud_create_card(board_id, stack_id, title, duedate)`.
- Advance it: `nextcloud_move_card(board_id, card_id, target_stack_id)`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_boards` |  | List the Deck boards the account can see. |
| `nextcloud_get_board` | board_id | Read one board with its labels and members. |
| `nextcloud_create_board` | title, color? | Create one board; ``color`` is a hex RGB value without ``#``. |
| `nextcloud_delete_board` | board_id | Delete one board (Deck keeps it restorable for a while). |
| `nextcloud_list_stacks` | board_id | List the stacks (columns) of one board, each with its cards. |
| `nextcloud_create_stack` | board_id, title, order? | Create one stack (column) on a board. |
| `nextcloud_delete_stack` | board_id, stack_id | Delete one stack and its cards. |
| `nextcloud_create_card` | board_id, stack_id, title, description?, duedate?, order? | Create one card in a stack; ``duedate`` is ISO 8601. |
| `nextcloud_get_card` | board_id, stack_id, card_id | Read one card. |
| `nextcloud_update_card` | board_id, stack_id, card_id, title?, description?, duedate? | Change one card's title, description or due date. |
| `nextcloud_move_card` | board_id, card_id, target_stack_id, order? | Move one card to a stack (column) of the board, at position ``order``. |
| `nextcloud_assign_card_user` | board_id, stack_id, card_id, user_id | Assign one user to a card. |
| `nextcloud_delete_card` | board_id, stack_id, card_id | Delete one card. |

### Contract: Track a piece of work on a Deck board

1. `nextcloud_list_boards` — find the board.
2. `nextcloud_list_stacks`(board_id) — find the stack.
3. `nextcloud_create_card`(board_id, stack_id, title, duedate) — add the card.
4. `nextcloud_move_card`(board_id, card_id, target_stack_id) — advance it.

## Gotchas
- `nextcloud_update_card` changes text and due date only; moving between stacks is `nextcloud_move_card`.
- Due dates are ISO 8601 with an offset.
- A deleted board stays restorable in Deck for a while.
