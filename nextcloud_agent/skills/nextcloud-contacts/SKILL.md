---
name: nextcloud-contacts
description: "Generated skill for contacts operations. Contains 3 tools."
---

### Overview
This skill handles operations related to contacts.

### Available Tools
- `list_address_books`: List address books.
  - **Parameters**:
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `list_contacts`: List contacts in an address book.
  - **Parameters**:
    - `address_book_url` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `create_contact`: Create a new contact using raw vCard data.
  - **Parameters**:
    - `address_book_url` (str)
    - `vcard_data` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)

### Usage Instructions
1. Review the tool available in this skill.
2. Call the tool with the required parameters.

### Error Handling
- Ensure all required parameters are provided.
- Check return values for error messages.
