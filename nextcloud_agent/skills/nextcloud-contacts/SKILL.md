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

## Steps

1. List address books with `nextcloud_list_address_books`(). Verify the Contacts book and its url are present.
2. Search with `nextcloud_list_contacts`(address_book_url). Verify a match on full_name or emails, or confirm the person is absent.
3. Add a new person with `nextcloud_create_contact`(address_book_url, full_name, emails, phones, organization). Verify the response returns a contact url.
4. Change one with `nextcloud_update_contact`(contact_url, emails). Verify with `nextcloud_get_contact`(contact_url) that the field changed.
5. Remove only on request with `nextcloud_delete_contact`(contact_url). Verify the contact is gone from `nextcloud_list_contacts`(address_book_url).

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| 401 or connection error from a `nextcloud_*` tool | `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME` or the app password in `NEXTCLOUD_PASSWORD` is missing or wrong | Set the three variables for the server, and verify the connection by calling `nextcloud_get_user_info` |
| Update fails on a stale etag | The contact changed since it was read | Re-read it with `nextcloud_get_contact`(contact_url) and update again |
| Contact not found | The url came from another address book | Take the url from `nextcloud_list_contacts`(address_book_url) of the right book |

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
