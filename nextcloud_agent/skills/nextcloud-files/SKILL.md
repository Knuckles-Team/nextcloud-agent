---
name: nextcloud-files
description: "Generated skill for files operations. Contains 8 tools."
---

### Overview
This skill handles operations related to files.

### Available Tools
- `list_files`: List files and directories at a specific path in Nextcloud. Returns a formatted string list of contents.
  - **Parameters**:
    - `path` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `read_file`: Read the contents of a text file from Nextcloud.
  - **Parameters**:
    - `path` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `write_file`: Write text content to a file in Nextcloud.
  - **Parameters**:
    - `path` (str)
    - `content` (str)
    - `overwrite` (bool)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `create_folder`: Create a new directory in Nextcloud.
  - **Parameters**:
    - `path` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `delete_item`: Delete a file or directory in Nextcloud.
  - **Parameters**:
    - `path` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `move_item`: Move a file or directory to a new location.
  - **Parameters**:
    - `source` (str)
    - `destination` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `copy_item`: Copy a file or directory to a new location.
  - **Parameters**:
    - `source` (str)
    - `destination` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `get_properties`: Get detailed properties for a file or folder.
  - **Parameters**:
    - `path` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)

### Usage Instructions
1. Review the tool available in this skill.
2. Call the tool with the required parameters.

### Error Handling
- Ensure all required parameters are provided.
- Check return values for error messages.
