---
name: nextcloud-contacts
skill_type: skill
description: >-
  Manage Nextcloud contacts over CardDAV: list address books, list contacts with names, emails, phones and organization, read, create, update and delete a contact. Use when the agent must look up or maintain customer, supplier or team contact details. Do NOT use for account administration (nextcloud-admin).
license: MIT
tags: [nextcloud, contacts, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Nextcloud Contacts

Manage Nextcloud contacts over CardDAV: list address books, list contacts with names, emails, phones and organization, read, create, update and delete a contact.

## When to use
- Look someone up: `nextcloud_list_contacts(address_book_url)` and match on `full_name` or `emails`.
- Add a customer: `nextcloud_create_contact(address_book_url, full_name, emails, phones, organization)`.

## Connection
Reach the tools through arbora (`find`, then `act op=fleet.call` with `server="nextcloud-mcp"`) or the
`nextcloud-mcp` MCP server directly. The server reads `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` and
`NEXTCLOUD_PASSWORD` (an app password); `NEXTCLOUD_ALLOWED_PRIVATE_HOSTS` admits a LAN host. Destructive
tools ask for confirmation (pass `confirm=true` only after the person agreed).

## Tools
| Tool | Arguments | Purpose |
|------|-----------|---------|
| `nextcloud_list_address_books` |  | List the account's address books. |
| `nextcloud_list_contacts` | address_book_url | List the contacts of one address book with name, emails, phones and organization. |
| `nextcloud_get_contact` | contact_url | Read one contact: parsed fields, etag and the raw vCard. |
| `nextcloud_create_contact` | address_book_url, full_name, emails?, phones?, organization?, title?, note? | Create one contact in an address book. |
| `nextcloud_update_contact` | contact_url, full_name?, emails?, phones?, organization?, title?, note? | Change one contact; omitted fields keep their value. Guarded by the etag. |
| `nextcloud_delete_contact` | contact_url | Delete one contact. |

## Gotchas
- Use the `Contacts` address book for business contacts; `Accounts` and `Recently contacted` are server-generated.
- An update replaces only the fields you pass and is guarded by the etag.
- Contacts are personal data: never copy them into public shares or chat.
