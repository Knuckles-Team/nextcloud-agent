# Storage backends seam contract

Provider: scion-sdk SCION-CENTRALIZED-CAPABILITIES-R042.5
Consumer: nextcloud-agent CONN-NEXTCLOUD-AGENT-R001

Provider spec: [SCION-CENTRALIZED-CAPABILITIES](https://github.com/Knuckles-Team/scion-sdk/blob/main/specs/centralized-capabilities/requirements.md).

## Wire shape

The seam is a Python import. The connector builds `WebDavBackend(base_url, username=..., password=..., allowed_private_hosts=...)` from `scion_sdk.storage`. It calls the path methods `propfind`, `read_path`, `write_path`, `make_collection`, `delete_path`, `copy_path` and `move_path`. `propfind(path, depth, extra_props)` returns entries with `href`, `name`, `is_collection`, `size`, `etag`, `last_modified`, `content_type` and a `props` map of the extra properties.

## Refusal codes

`STORAGE_NOT_FOUND`, `STORAGE_EXISTS`, `STORAGE_NAME_INVALID`, `STORAGE_TOO_LARGE`, `STORAGE_BACKEND_FAILED`, `STORAGE_EGRESS_REFUSED`. The connector maps not-found and exists to its current error text.

## Authoritative types

- `WebDavBackend` and `DavEntry` in `scion_sdk/storage/webdav.py` in scion-sdk.

## Test vectors

- `tests/storage/test_webdav_backend.py` in scion-sdk.
- `tests/test_files_webdav.py` in this repository.
