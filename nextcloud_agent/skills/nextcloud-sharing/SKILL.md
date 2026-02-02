---
name: nextcloud-sharing
description: "Generated skill for sharing operations. Contains 3 tools."
---

### Overview
This skill handles operations related to sharing.

### Available Tools
- `list_shares`: List all shares.
  - **Parameters**:
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `create_share`: Create a new share.
  - **Parameters**:
    - `path` (str)
    - `share_type` (int)
    - `permissions` (int)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `delete_share`: Delete a share.
  - **Parameters**:
    - `share_id` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)

### Usage Instructions
1. Review the tool available in this skill.
2. Call the tool with the required parameters.

### Error Handling
- Ensure all required parameters are provided.
- Check return values for error messages.
