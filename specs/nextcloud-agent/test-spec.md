# CONN-NEXTCLOUD-AGENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-NEXTCLOUD-AGENT-R001` | `tests/test_files_webdav.py::test_file_methods_route_through_sdk_webdav_backend` | Every file request passes through the SDK client, and a listing keeps the three Nextcloud fields. |
| T-002 | `CONN-NEXTCLOUD-AGENT-R002` | `tests/test_files_webdav.py::test_backend_uses_files_root_and_private_host_setting` | The base URL ends with the files root, and a private host refuses unless listed. |
| T-003 | `CONN-NEXTCLOUD-AGENT-R003` | `tests/test_files_webdav.py::test_sdk_pin_carries_storage_backend` | The backend class imports. |
| T-004 | `CONN-NEXTCLOUD-AGENT-R004` | `tests/test_business_surface.py::test_versions_search_and_trash_use_the_dav_endpoints` | versions sort newest first, restore targets `restore/target`, the search body carries the name and MIME filters. |
| T-005 | `CONN-NEXTCLOUD-AGENT-R005` | `tests/test_business_surface.py::test_shares_cover_public_links_and_internal_grants` | a public link returns its URL, an internal share without `share_with` refuses, `sendMail` is false by default. |
| T-006 | `CONN-NEXTCLOUD-AGENT-R006` | `tests/test_business_surface.py::test_event_invites_stay_drafts_unless_asked` | a new event is written with `If-None-Match: *`, attendees carry `SCHEDULE-AGENT=CLIENT`, and `send_invitations` removes it. |
| T-007 | `CONN-NEXTCLOUD-AGENT-R007` | `tests/test_business_surface.py::test_tasks_are_vtodos_and_complete_with_etag` | completing a task writes `STATUS:COMPLETED` with `If-Match`. |
| T-008 | `CONN-NEXTCLOUD-AGENT-R008` | `tests/test_business_surface.py::test_contacts_round_trip_as_vcards` | the created vCard carries the fields; a delete outside the address books refuses. |
| T-009 | `CONN-NEXTCLOUD-AGENT-R009` | `tests/test_business_surface.py::test_talk_conversations_and_messages` | a group conversation posts `roomType` 2, a message returns its id, an invalid token refuses. |
| T-010 | `CONN-NEXTCLOUD-AGENT-R010` | `tests/test_business_surface.py::test_deck_move_uses_the_target_stack_route` | a move calls `/stacks/{target}/cards/{id}/reorder`. |
| T-011 | `CONN-NEXTCLOUD-AGENT-R011` | `tests/test_business_surface.py::test_notes_crud_uses_the_notes_api` | create returns the note; an empty update refuses. |
| T-012 | `CONN-NEXTCLOUD-AGENT-R012` | `tests/test_business_surface.py::test_activity_not_modified_reads_as_empty` | A 304 answer from the activity API reads as an empty list. |
| T-013 | `CONN-NEXTCLOUD-AGENT-R013` | `tests/test_business_surface.py::test_admin_creates_users_into_groups` | a user without password or email refuses; groups post as `groups[]`; an id with `/` refuses. |
| T-014 | `CONN-NEXTCLOUD-AGENT-R014` | `tests/test_business_surface.py::test_office_documents_open_in_the_matching_editor` | a `.docx` opens through richdocuments, a `.md` through Text. |
| T-015 | `CONN-NEXTCLOUD-AGENT-R015` | `tests/test_business_surface.py::test_every_operation_is_its_own_typed_tool` | the business tools are listed, none takes `action`, no duplicate alias remains. |

## Negative and boundary cases

A document type declaration in a listing refuses. A write with `overwrite` false onto an existing file returns the conflict error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.

Live proof against the deployed instance (2026-11-06) is recorded in the operations validation ledger; each row's operation ran create, verify and delete on `zz-validation-20261106` objects.
