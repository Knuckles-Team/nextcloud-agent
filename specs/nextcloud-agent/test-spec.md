# CONN-NEXTCLOUD-AGENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-NEXTCLOUD-AGENT-R001` | `tests/test_files_webdav.py::test_file_methods_route_through_sdk_webdav_backend` (planned) | Every file request passes through the SDK client, and a listing keeps the three Nextcloud fields. |
| T-002 | `CONN-NEXTCLOUD-AGENT-R002` | `tests/test_files_webdav.py::test_backend_uses_files_root_and_private_host_setting` (planned) | The base URL ends with the files root, and a private host refuses unless listed. |
| T-003 | `CONN-NEXTCLOUD-AGENT-R003` | `tests/test_files_webdav.py::test_sdk_pin_carries_storage_backend` (planned) | The backend class imports. |

## Negative and boundary cases

A document type declaration in a listing refuses. A write with `overwrite` false onto an existing file returns the conflict error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.
