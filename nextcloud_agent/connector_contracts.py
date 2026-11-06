"""Tool-use contracts, surface links, data locations and business roles of nextcloud-mcp.

The SDK serves this declaration as the connector's surface ontology and checks it
against the served tools, prompts and skills (SCION-CONNECTOR-CONTROL-R916).
"""

# ruff: noqa: E501
from __future__ import annotations

from scion_sdk.surface_contracts import (
    BusinessRole,
    DataLocation,
    Refusal,
    SurfaceLink,
    ToolStep,
    ToolUseContract,
)

CONTRACTS: tuple[ToolUseContract, ...] = (
    ToolUseContract(
        "publish_asset",
        description="Publish a produced asset to the team folder and share it",
        steps=(
            ToolStep(
                "nextcloud_list_files",
                params=("path",),
                result=("name", "href", "file_id", "is_folder", "content_type"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_folder",
                params=("path",),
                result=("created",),
                refusals=(Refusal("already_exists", "the folder exists (HTTP 405)"),),
            ),
            ToolStep(
                "nextcloud_copy_item",
                params=("source_path", "dest_path"),
                result=("copied",),
                refusals=(
                    Refusal(
                        "already_exists",
                        "the destination exists and overwrite is false",
                    ),
                    Refusal("not_found", "the source path does not exist"),
                ),
            ),
            ToolStep(
                "nextcloud_list_shares",
                params=("path",),
                result=("id", "path", "share_type", "share_with", "url"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_share",
                params=("path", "share_type", "share_with", "permissions"),
                result=("id", "url", "token"),
                refusals=(
                    Refusal(
                        "share_with_required", "an internal share has no share_with"
                    ),
                    Refusal("forbidden", "the account may not share the path"),
                ),
            ),
            ToolStep(
                "nextcloud_create_share",
                params=("path", "share_type", "expire_date", "label"),
                result=("id", "url", "token"),
                refusals=(
                    Refusal(
                        "share_with_required", "an internal share has no share_with"
                    ),
                    Refusal("forbidden", "the account may not share the path"),
                ),
            ),
            ToolStep(
                "nextcloud_get_share",
                params=("share_id",),
                result=("id", "share_type", "permissions", "expiration", "url"),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "schedule_meeting",
        description="Schedule a meeting with draft invitations",
        steps=(
            ToolStep(
                "nextcloud_list_calendars",
                params=(),
                result=("url", "displayname", "components"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_events",
                params=("calendar_url", "start", "end"),
                result=("url", "uid", "summary", "start", "end", "etag"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_event",
                params=(
                    "calendar_url",
                    "summary",
                    "start",
                    "end",
                    "attendees",
                    "send_invitations",
                ),
                result=("url", "uid", "etag"),
                refusals=(
                    Refusal(
                        "already_exists",
                        "an object with the same uid exists (If-None-Match)",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_get_calendar_object",
                params=("object_url",),
                result=("summary", "etag", "ics"),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "team_announcement",
        description="Post an update to a private team conversation",
        steps=(
            ToolStep(
                "nextcloud_list_conversations",
                params=(),
                result=("token", "displayName", "type"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_send_message",
                params=("token", "message"),
                result=("id", "message"),
                refusals=(
                    Refusal("invalid_token", "the conversation token is malformed"),
                ),
            ),
            ToolStep(
                "nextcloud_list_messages",
                params=("token", "limit"),
                result=("id", "actorId", "message", "timestamp"),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "track_work",
        description="Track a piece of work on a Deck board",
        steps=(
            ToolStep(
                "nextcloud_list_boards",
                params=(),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_stacks",
                params=("board_id",),
                result=("id", "title", "cards"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_card",
                params=("board_id", "stack_id", "title", "duedate"),
                result=("id", "title", "stackId"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_move_card",
                params=("board_id", "card_id", "target_stack_id"),
                result=("id", "stackId"),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "collaborate_document",
        description="Create a document for the team to edit together",
        steps=(
            ToolStep(
                "nextcloud_create_document",
                params=("path",),
                result=("fileid", "basename", "mime"),
                refusals=(
                    Refusal("already_exists", "a file exists at the path (HTTP 403)"),
                ),
            ),
            ToolStep(
                "nextcloud_create_share",
                params=("path", "share_type", "share_with", "permissions"),
                result=("id", "url", "token"),
                refusals=(
                    Refusal(
                        "share_with_required", "an internal share has no share_with"
                    ),
                    Refusal("forbidden", "the account may not share the path"),
                ),
            ),
            ToolStep(
                "nextcloud_open_document",
                params=("path",),
                result=("url",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "onboard_member",
        description="Onboard a team member",
        steps=(
            ToolStep(
                "nextcloud_create_user",
                params=("user_id", "email", "display_name", "groups"),
                result=("id",),
                refusals=(
                    Refusal(
                        "credential_required", "neither password nor email was given"
                    ),
                    Refusal("already_exists", "the account id is taken"),
                ),
            ),
            ToolStep(
                "nextcloud_get_user",
                params=("user_id",),
                result=("id", "email", "enabled", "groups"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_add_participant",
                params=("token", "participant"),
                result=("type",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "manage_files",
        description="Every operation of Files: browse, upload, download, move, copy, delete, versions, search, trash",
        steps=(
            ToolStep(
                "nextcloud_list_files",
                params=("path",),
                result=("name", "href", "file_id", "is_folder", "content_type"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_read_file",
                params=("path",),
                result=("content_base64",),
                refusals=(Refusal("not_found", "the path does not exist"),),
            ),
            ToolStep(
                "nextcloud_write_file",
                params=("path", "content"),
                result=("written",),
                refusals=(
                    Refusal("already_exists", "overwrite is false and the file exists"),
                ),
            ),
            ToolStep(
                "nextcloud_create_folder",
                params=("path",),
                result=("created",),
                refusals=(Refusal("already_exists", "the folder exists (HTTP 405)"),),
            ),
            ToolStep(
                "nextcloud_delete_item",
                params=("path",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_move_item",
                params=("source_path", "dest_path"),
                result=("moved",),
                refusals=(
                    Refusal(
                        "already_exists",
                        "the destination exists and overwrite is false",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_copy_item",
                params=("source_path", "dest_path"),
                result=("copied",),
                refusals=(
                    Refusal(
                        "already_exists",
                        "the destination exists and overwrite is false",
                    ),
                    Refusal("not_found", "the source path does not exist"),
                ),
            ),
            ToolStep(
                "nextcloud_get_properties",
                params=("path",),
                result=("file_id", "content_type", "content_length", "etag"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_user_quota",
                params=(),
                result=("quota_used", "quota_available"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_versions",
                params=("path",),
                result=("version_id", "getlastmodified", "getcontentlength"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_restore_version",
                params=("path", "version_id"),
                result=("restored",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_search_files",
                params=("query",),
                result=("path", "fileid", "getcontenttype"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_trash",
                params=(),
                result=("trash_id", "trashbin_original_location"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_restore_trash_item",
                params=("trash_id",),
                result=("restored",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "manage_sharing",
        description="Every operation of Sharing: internal user/group/Talk shares and public links",
        steps=(
            ToolStep(
                "nextcloud_list_shares",
                params=("path",),
                result=("id", "path", "share_type", "share_with", "url"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_share",
                params=("share_id",),
                result=("id", "share_type", "permissions", "expiration", "url"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_share",
                params=("path",),
                result=("id", "url", "token"),
                refusals=(
                    Refusal(
                        "share_with_required", "an internal share has no share_with"
                    ),
                    Refusal("forbidden", "the account may not share the path"),
                ),
            ),
            ToolStep(
                "nextcloud_update_share",
                params=("share_id",),
                result=("id", "permissions", "expiration"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_share",
                params=("share_id",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_calendar",
        description="Every operation of Calendar: calendars and events with draft invitations (CalDAV)",
        steps=(
            ToolStep(
                "nextcloud_list_calendars",
                params=(),
                result=("url", "displayname", "components"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_calendar",
                params=("display_name",),
                result=("url", "slug"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_calendar",
                params=("calendar_url",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_events",
                params=("calendar_url",),
                result=("url", "uid", "summary", "start", "end", "etag"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_calendar_object",
                params=("object_url",),
                result=("summary", "etag", "ics"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_event",
                params=("calendar_url", "summary", "start"),
                result=("url", "uid", "etag"),
                refusals=(
                    Refusal(
                        "already_exists",
                        "an object with the same uid exists (If-None-Match)",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_update_event",
                params=("event_url",),
                result=("url", "etag"),
                refusals=(
                    Refusal(
                        "precondition_failed",
                        "the event changed since it was read (etag)",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_delete_calendar_object",
                params=("object_url",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_tasks",
        description="Every operation of Tasks: to-dos (VTODO) in task-capable calendars",
        steps=(
            ToolStep(
                "nextcloud_list_tasks",
                params=("calendar_url",),
                result=("url", "summary", "due", "status"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_task",
                params=("calendar_url", "summary"),
                result=("url", "uid"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_update_task",
                params=("task_url",),
                result=("url", "etag"),
                refusals=(
                    Refusal(
                        "precondition_failed",
                        "the task changed since it was read (etag)",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_contacts",
        description="Every operation of Contacts: address books and vCards (CardDAV)",
        steps=(
            ToolStep(
                "nextcloud_list_address_books",
                params=(),
                result=("url", "displayname"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_contacts",
                params=("address_book_url",),
                result=("url", "full_name", "emails", "phones"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_contact",
                params=("contact_url",),
                result=("full_name", "etag", "vcard"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_contact",
                params=("address_book_url", "full_name"),
                result=("url", "uid"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_update_contact",
                params=("contact_url",),
                result=("url", "etag"),
                refusals=(
                    Refusal(
                        "precondition_failed",
                        "the contact changed since it was read (etag)",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_delete_contact",
                params=("contact_url",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_talk",
        description="Every operation of Talk: conversations, participants and chat",
        steps=(
            ToolStep(
                "nextcloud_list_conversations",
                params=(),
                result=("token", "displayName", "type"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_conversation",
                params=("token",),
                result=("token", "displayName"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_conversation",
                params=("name",),
                result=("token", "type"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_rename_conversation",
                params=("token", "name"),
                result=("renamed",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_conversation",
                params=("token",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_participants",
                params=("token",),
                result=("attendeeId", "actorId", "participantType"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_add_participant",
                params=("token", "participant"),
                result=("type",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_remove_participant",
                params=("token", "attendee_id"),
                result=("removed",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_messages",
                params=("token",),
                result=("id", "actorId", "message", "timestamp"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_send_message",
                params=("token", "message"),
                result=("id", "message"),
                refusals=(
                    Refusal("invalid_token", "the conversation token is malformed"),
                ),
            ),
            ToolStep(
                "nextcloud_delete_message",
                params=("token", "message_id"),
                result=("id",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_deck",
        description="Every operation of Deck: kanban boards, stacks and cards",
        steps=(
            ToolStep(
                "nextcloud_list_boards",
                params=(),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_board",
                params=("board_id",),
                result=("id", "title", "labels"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_board",
                params=("title",),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_board",
                params=("board_id",),
                result=("id",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_stacks",
                params=("board_id",),
                result=("id", "title", "cards"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_stack",
                params=("board_id", "title"),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_stack",
                params=("board_id", "stack_id"),
                result=("id",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_create_card",
                params=("board_id", "stack_id", "title"),
                result=("id", "title", "stackId"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_card",
                params=("board_id", "stack_id", "card_id"),
                result=("id", "title", "duedate"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_update_card",
                params=("board_id", "stack_id", "card_id"),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_move_card",
                params=("board_id", "card_id", "target_stack_id"),
                result=("id", "stackId"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_assign_card_user",
                params=("board_id", "stack_id", "card_id", "user_id"),
                result=("participant",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_card",
                params=("board_id", "stack_id", "card_id"),
                result=("id",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_notes",
        description="Every operation of Notes: markdown notes",
        steps=(
            ToolStep(
                "nextcloud_list_notes",
                params=("category",),
                result=("id", "title", "category"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_note",
                params=("note_id",),
                result=("id", "title", "content"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_note",
                params=("title",),
                result=("id", "title"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_update_note",
                params=("note_id",),
                result=("id", "content"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_note",
                params=("note_id",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_office",
        description="Every operation of Office: create and collaboratively edit documents, spreadsheets, presentations and markdown",
        steps=(
            ToolStep(
                "nextcloud_list_document_templates",
                params=(),
                result=("app", "extension", "templates"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_document",
                params=("path",),
                result=("fileid", "basename", "mime"),
                refusals=(
                    Refusal("already_exists", "a file exists at the path (HTTP 403)"),
                ),
            ),
            ToolStep(
                "nextcloud_open_document",
                params=("path",),
                result=("url",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "manage_activity",
        description="Every operation of Activity and notifications",
        steps=(
            ToolStep(
                "nextcloud_list_activities",
                params=("limit",),
                result=("activity_id", "subject", "datetime", "object_name"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_notifications",
                params=(),
                result=("notification_id", "subject"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_dismiss_notification",
                params=("notification_id",),
                result=("dismissed",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "manage_admin",
        description="Every operation of Administration: accounts, groups and apps",
        steps=(
            ToolStep(
                "nextcloud_get_user_info",
                params=(),
                result=("id", "displayname", "email", "groups"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_list_users",
                params=("search",),
                result=("user_ids",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_user",
                params=("user_id",),
                result=("id", "email", "enabled", "groups"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_user",
                params=("user_id",),
                result=("id",),
                refusals=(
                    Refusal(
                        "credential_required", "neither password nor email was given"
                    ),
                    Refusal("already_exists", "the account id is taken"),
                ),
            ),
            ToolStep(
                "nextcloud_update_user",
                params=("user_id", "key", "value"),
                result=("updated",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_disable_user",
                params=("user_id",),
                result=("disabled",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_enable_user",
                params=("user_id",),
                result=("enabled",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_user",
                params=("user_id",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_groups",
                params=("search",),
                result=("group_ids",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_create_group",
                params=("group_id",),
                result=("created",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_delete_group",
                params=("group_id",),
                result=("deleted",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_group_members",
                params=("group_id",),
                result=("user_ids",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_add_user_to_group",
                params=("user_id", "group_id"),
                result=("added",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_remove_user_from_group",
                params=("user_id", "group_id"),
                result=("removed",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
            ToolStep(
                "nextcloud_list_apps",
                params=("status",),
                result=("app_ids",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_get_app",
                params=("app_id",),
                result=("id", "version"),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_enable_app",
                params=("app_id",),
                result=("enabled",),
                refusals=(),
            ),
            ToolStep(
                "nextcloud_disable_app",
                params=("app_id",),
                result=("disabled",),
                refusals=(
                    Refusal(
                        "not_confirmed",
                        "the caller did not confirm the destructive call",
                    ),
                ),
            ),
        ),
    ),
)

LINKS: tuple[SurfaceLink, ...] = (
    SurfaceLink(
        "publish_asset",
        tools=(
            "nextcloud_copy_item",
            "nextcloud_create_folder",
            "nextcloud_create_share",
            "nextcloud_get_share",
            "nextcloud_list_files",
            "nextcloud_list_shares",
        ),
        prompts=("operator", "asset_publisher"),
        skills=("nextcloud-publish-asset",),
    ),
    SurfaceLink(
        "schedule_meeting",
        tools=(
            "nextcloud_create_event",
            "nextcloud_get_calendar_object",
            "nextcloud_list_calendars",
            "nextcloud_list_events",
        ),
        prompts=("operator", "office_coordinator"),
        skills=("nextcloud-calendar",),
    ),
    SurfaceLink(
        "team_announcement",
        tools=(
            "nextcloud_list_conversations",
            "nextcloud_list_messages",
            "nextcloud_send_message",
        ),
        prompts=("operator", "asset_publisher", "office_coordinator"),
        skills=("nextcloud-talk",),
    ),
    SurfaceLink(
        "track_work",
        tools=(
            "nextcloud_create_card",
            "nextcloud_list_boards",
            "nextcloud_list_stacks",
            "nextcloud_move_card",
        ),
        prompts=("operator", "project_coordinator"),
        skills=("nextcloud-deck",),
    ),
    SurfaceLink(
        "collaborate_document",
        tools=(
            "nextcloud_create_document",
            "nextcloud_create_share",
            "nextcloud_open_document",
        ),
        prompts=("operator", "project_coordinator"),
        skills=("nextcloud-office",),
    ),
    SurfaceLink(
        "onboard_member",
        tools=(
            "nextcloud_add_participant",
            "nextcloud_create_user",
            "nextcloud_get_user",
        ),
        prompts=("operator", "workspace_admin"),
        skills=("nextcloud-admin",),
    ),
    SurfaceLink(
        "manage_files",
        tools=(
            "nextcloud_list_files",
            "nextcloud_read_file",
            "nextcloud_write_file",
            "nextcloud_create_folder",
            "nextcloud_delete_item",
            "nextcloud_move_item",
            "nextcloud_copy_item",
            "nextcloud_get_properties",
            "nextcloud_get_user_quota",
            "nextcloud_list_versions",
            "nextcloud_restore_version",
            "nextcloud_search_files",
            "nextcloud_list_trash",
            "nextcloud_restore_trash_item",
        ),
        prompts=("operator", "asset_publisher", "file_steward"),
        skills=("nextcloud-files",),
    ),
    SurfaceLink(
        "manage_sharing",
        tools=(
            "nextcloud_list_shares",
            "nextcloud_get_share",
            "nextcloud_create_share",
            "nextcloud_update_share",
            "nextcloud_delete_share",
        ),
        prompts=("operator", "asset_publisher", "workspace_admin", "file_steward"),
        skills=("nextcloud-shares",),
    ),
    SurfaceLink(
        "manage_calendar",
        tools=(
            "nextcloud_list_calendars",
            "nextcloud_create_calendar",
            "nextcloud_delete_calendar",
            "nextcloud_list_events",
            "nextcloud_get_calendar_object",
            "nextcloud_create_event",
            "nextcloud_update_event",
            "nextcloud_delete_calendar_object",
        ),
        prompts=("operator", "office_coordinator"),
        skills=("nextcloud-calendar",),
    ),
    SurfaceLink(
        "manage_tasks",
        tools=(
            "nextcloud_list_tasks",
            "nextcloud_create_task",
            "nextcloud_update_task",
        ),
        prompts=("operator", "office_coordinator", "project_coordinator"),
        skills=("nextcloud-tasks",),
    ),
    SurfaceLink(
        "manage_contacts",
        tools=(
            "nextcloud_list_address_books",
            "nextcloud_list_contacts",
            "nextcloud_get_contact",
            "nextcloud_create_contact",
            "nextcloud_update_contact",
            "nextcloud_delete_contact",
        ),
        prompts=("operator", "office_coordinator"),
        skills=("nextcloud-contacts",),
    ),
    SurfaceLink(
        "manage_talk",
        tools=(
            "nextcloud_list_conversations",
            "nextcloud_get_conversation",
            "nextcloud_create_conversation",
            "nextcloud_rename_conversation",
            "nextcloud_delete_conversation",
            "nextcloud_list_participants",
            "nextcloud_add_participant",
            "nextcloud_remove_participant",
            "nextcloud_list_messages",
            "nextcloud_send_message",
            "nextcloud_delete_message",
        ),
        prompts=("operator", "asset_publisher", "office_coordinator"),
        skills=("nextcloud-talk",),
    ),
    SurfaceLink(
        "manage_deck",
        tools=(
            "nextcloud_list_boards",
            "nextcloud_get_board",
            "nextcloud_create_board",
            "nextcloud_delete_board",
            "nextcloud_list_stacks",
            "nextcloud_create_stack",
            "nextcloud_delete_stack",
            "nextcloud_create_card",
            "nextcloud_get_card",
            "nextcloud_update_card",
            "nextcloud_move_card",
            "nextcloud_assign_card_user",
            "nextcloud_delete_card",
        ),
        prompts=("operator", "project_coordinator"),
        skills=("nextcloud-deck",),
    ),
    SurfaceLink(
        "manage_notes",
        tools=(
            "nextcloud_list_notes",
            "nextcloud_get_note",
            "nextcloud_create_note",
            "nextcloud_update_note",
            "nextcloud_delete_note",
        ),
        prompts=("operator", "office_coordinator", "project_coordinator"),
        skills=("nextcloud-notes",),
    ),
    SurfaceLink(
        "manage_office",
        tools=(
            "nextcloud_list_document_templates",
            "nextcloud_create_document",
            "nextcloud_open_document",
        ),
        prompts=("operator", "project_coordinator"),
        skills=("nextcloud-office",),
    ),
    SurfaceLink(
        "manage_activity",
        tools=(
            "nextcloud_list_activities",
            "nextcloud_list_notifications",
            "nextcloud_dismiss_notification",
        ),
        prompts=("operator", "project_coordinator", "workspace_admin"),
        skills=("nextcloud-activity",),
    ),
    SurfaceLink(
        "manage_admin",
        tools=(
            "nextcloud_get_user_info",
            "nextcloud_list_users",
            "nextcloud_get_user",
            "nextcloud_create_user",
            "nextcloud_update_user",
            "nextcloud_disable_user",
            "nextcloud_enable_user",
            "nextcloud_delete_user",
            "nextcloud_list_groups",
            "nextcloud_create_group",
            "nextcloud_delete_group",
            "nextcloud_list_group_members",
            "nextcloud_add_user_to_group",
            "nextcloud_remove_user_from_group",
            "nextcloud_list_apps",
            "nextcloud_get_app",
            "nextcloud_enable_app",
            "nextcloud_disable_app",
        ),
        prompts=("operator", "workspace_admin"),
        skills=("nextcloud-admin",),
    ),
    SurfaceLink(
        "ingest_nextcloud",
        tools=(
            "nextcloud_list_files",
            "nextcloud_list_shares",
            "nextcloud_list_notes",
            "nextcloud_list_boards",
            "nextcloud_list_conversations",
            "nextcloud_list_calendars",
            "nextcloud_list_address_books",
        ),
        prompts=("operator", "file_steward"),
        skills=("nextcloud-ingest",),
    ),
    SurfaceLink(
        "operate_connector",
        tools=("nextcloud_get_user_info", "nextcloud_list_apps"),
        prompts=("operator",),
        skills=("nextcloud-agent-operations", "nextcloud-agent-development"),
    ),
)

LOCATIONS: tuple[DataLocation, ...] = (
    DataLocation(
        "files",
        holds=("File", "Folder", "OfficeDocument"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_files",
        store="WebDAV {NEXTCLOUD_URL}/remote.php/dav/files/{user}/ (Account files)",
        route_param="path",
        federation_key="file_id",
    ),
    DataLocation(
        "versions",
        holds=("FileVersion",),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_versions",
        store="WebDAV {NEXTCLOUD_URL}/remote.php/dav/versions/{user}/versions/{fileid}/ (File versions)",
        route_param="path",
        federation_key="version_id",
    ),
    DataLocation(
        "trash",
        holds=("TrashItem",),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_trash",
        store="WebDAV {NEXTCLOUD_URL}/remote.php/dav/trashbin/{user}/trash/ (Trash bin)",
        route_param="",
        federation_key="trash_id",
    ),
    DataLocation(
        "shares",
        holds=("Share",),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_shares",
        store="OCS {NEXTCLOUD_URL}/ocs/v2.php/apps/files_sharing/api/v1/shares (Share grants)",
        route_param="path",
        federation_key="id",
    ),
    DataLocation(
        "calendars",
        holds=("NextcloudCalendar", "CalendarEvent", "PersonalTask"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_events",
        store="CalDAV {NEXTCLOUD_URL}/remote.php/dav/calendars/{user}/ (Calendars, events and tasks)",
        route_param="calendar_url",
        federation_key="uid",
    ),
    DataLocation(
        "contacts",
        holds=("AddressBook", "ContactCard"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_contacts",
        store="CardDAV {NEXTCLOUD_URL}/remote.php/dav/addressbooks/users/{user}/ (Address books)",
        route_param="address_book_url",
        federation_key="uid",
    ),
    DataLocation(
        "talk",
        holds=("TalkConversation", "TalkMessage"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_messages",
        store="OCS {NEXTCLOUD_URL}/ocs/v2.php/apps/spreed/api/ (Talk conversations and chat)",
        route_param="token",
        federation_key="id",
    ),
    DataLocation(
        "deck",
        holds=("DeckBoard", "DeckStack", "DeckCard"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_stacks",
        store="REST {NEXTCLOUD_URL}/index.php/apps/deck/api/v1.0/boards (Deck boards)",
        route_param="board_id",
        federation_key="id",
    ),
    DataLocation(
        "notes",
        holds=("NextcloudNote",),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_notes",
        store="REST {NEXTCLOUD_URL}/index.php/apps/notes/api/v1/notes (Notes (files in the Notes folder))",
        route_param="category",
        federation_key="id",
    ),
    DataLocation(
        "activity",
        holds=("ActivityEvent", "NextcloudNotification"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_activities",
        store="OCS {NEXTCLOUD_URL}/ocs/v2.php/apps/activity/api/v2/activity (Activity stream and notifications)",
        route_param="object_id",
        federation_key="activity_id",
    ),
    DataLocation(
        "accounts",
        holds=("IdentityUser", "IdentityGroup", "NextcloudApp"),
        instance="nextcloud (NEXTCLOUD_URL, connected account)",
        read_tool="nextcloud_list_users",
        store="OCS {NEXTCLOUD_URL}/ocs/v2.php/cloud/ (Accounts, groups and apps)",
        route_param="search",
        federation_key="id",
    ),
)

ROLES: tuple[BusinessRole, ...] = (
    BusinessRole(
        "asset-publisher",
        tasks=("publish_asset", "team_announcement", "manage_files", "manage_sharing"),
        description="Publishes produced creative assets to team folders and shares them.",
    ),
    BusinessRole(
        "office-coordinator",
        tasks=(
            "schedule_meeting",
            "team_announcement",
            "manage_calendar",
            "manage_tasks",
            "manage_contacts",
            "manage_notes",
            "manage_talk",
        ),
        description="Runs meetings, to-dos, contacts and notes.",
    ),
    BusinessRole(
        "project-coordinator",
        tasks=(
            "track_work",
            "collaborate_document",
            "manage_deck",
            "manage_tasks",
            "manage_office",
            "manage_notes",
            "manage_activity",
        ),
        description="Runs project boards, documents and change tracking.",
    ),
    BusinessRole(
        "workspace-admin",
        tasks=("onboard_member", "manage_admin", "manage_sharing", "manage_activity"),
        description="Administers accounts, groups, apps and access.",
    ),
    BusinessRole(
        "file-steward",
        tasks=("manage_files", "manage_sharing", "ingest_nextcloud"),
        description="Keeps the file tree organized, versioned and safely shared.",
    ),
)
