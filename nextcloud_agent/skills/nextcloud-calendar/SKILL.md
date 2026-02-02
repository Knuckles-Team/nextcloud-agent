---
name: nextcloud-calendar
description: "Generated skill for calendar operations. Contains 3 tools."
---

### Overview
This skill handles operations related to calendar.

### Available Tools
- `list_calendars`: List available calendars.
  - **Parameters**:
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `list_calendar_events`: List events in a calendar.
  - **Parameters**:
    - `calendar_url` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)
- `create_calendar_event`: No description provided.
  - **Parameters**:
    - `calendar_url` (str)
    - `summary` (str)
    - `start_time` (str)
    - `end_time` (str)
    - `description` (str)
    - `base_url` (str)
    - `username` (str)
    - `password` (str)

### Usage Instructions
1. Review the tool available in this skill.
2. Call the tool with the required parameters.

### Error Handling
- Ensure all required parameters are provided.
- Check return values for error messages.
