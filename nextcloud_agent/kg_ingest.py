"""Native epistemic-graph ingestion for Nextcloud structure (typed graph nodes).

CONCEPT:AU-KG.ingest.enterprise-source-extractor. The structure-node twin of
``kg_media`` (which stores the raw file blobs + extracted documents): this module maps
Nextcloud WebDAV/OCS records — folder listings, shares, calendar events — into **typed
OWL nodes** (``:File``, ``:Folder``, ``:Share``, ``:CalendarEvent``) + links
(``:inFolder``, ``:sharesResource``, ``:hasBlob``) via the shared
``agent_utilities.knowledge_graph.memory.native_ingest.ingest_entities`` primitive.

The required native transaction primitive fails closed when the authoritative engine is
unavailable; partial writes are never acknowledged. Node ids follow
``nextcloud:<class>:<externalId>`` and each ``node_type`` matches a class the package's
``ontology`` ``.ttl`` federates.
"""

from __future__ import annotations

import posixpath
from typing import Any

from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_entities as _native_ingest_entities,
)

_SOURCE = "nextcloud-agent"
_DOMAIN = "nextcloud"


def _ingest(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    ingest_entities: Any | None = None,
) -> dict[str, int]:
    """Route typed nodes/edges through the shared native primitive (injectable for tests)."""
    writer = ingest_entities or _native_ingest_entities
    return writer(entities, relationships, source=_SOURCE, domain=_DOMAIN)


def ingest_listing(
    entries: list[dict[str, Any]],
    *,
    parent_path: str = "",
    ingest_entities: Any | None = None,
) -> dict[str, int] | None:
    """Map a WebDAV folder listing → ``:File`` / ``:Folder`` nodes (+ ``:inFolder`` links).

    ``entries``: the dicts returned by ``Api.list_files`` (``name``, ``href``,
    ``is_folder``, ``file_id``, ``content_type``, ``content_length``, ``last_modified``).
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    parent_clean = parent_path.strip("/")
    parent_id = f"nextcloud:folder:{parent_clean}" if parent_clean else None
    if parent_id is not None:
        entities.append(
            {
                "id": parent_id,
                "node_type": "Folder",
                "name": posixpath.basename(parent_clean) or parent_clean,
                "path": parent_clean,
            }
        )

    for entry in entries or []:
        name = entry.get("name")
        if not name:
            continue
        is_folder = bool(entry.get("is_folder"))
        rel = posixpath.join(parent_clean, name) if parent_clean else name
        fid = entry.get("file_id") or rel
        if is_folder:
            node = {
                "id": f"nextcloud:folder:{fid}",
                "node_type": "Folder",
                "name": name,
                "path": rel,
                "modifiedAt": entry.get("last_modified"),
                "file_id": entry.get("file_id"),
            }
        else:
            size = entry.get("content_length")
            try:
                size = int(size) if size is not None else None
            except (TypeError, ValueError):
                size = None
            node = {
                "id": f"nextcloud:file:{fid}",
                "node_type": "File",
                "name": name,
                "path": rel,
                "mimeType": entry.get("content_type") or None,
                "sizeBytes": size,
                "modifiedAt": entry.get("last_modified"),
                "file_id": entry.get("file_id"),
                "etag": entry.get("etag"),
            }
        entities.append(node)
        if parent_id is not None:
            relationships.append(
                {"source": node["id"], "target": parent_id, "relationship": "inFolder"}
            )

    return _ingest(entities, relationships, ingest_entities=ingest_entities)


def ingest_shares(
    shares: list[dict[str, Any]] | dict[str, Any],
    *,
    ingest_entities: Any | None = None,
) -> dict[str, int] | None:
    """Map OCS shares → ``:Share`` nodes (+ ``:sharesResource`` to the shared path)."""
    if isinstance(shares, dict):
        shares = [shares]
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for share in shares or []:
        sid = share.get("id")
        if sid is None:
            continue
        path = (share.get("path") or "").strip("/")
        share_node = {
            "id": f"nextcloud:share:{sid}",
            "node_type": "Share",
            "name": share.get("file_target") or share.get("path"),
            "path": path or None,
            "shareType": share.get("share_type"),
            "permissions": share.get("permissions"),
            "sharedWith": share.get("share_with"),
            "url": share.get("url"),
        }
        entities.append(share_node)
        if path:
            # Link to the shared File/Folder node (by path-derived id when known).
            target = f"nextcloud:file:{path}"
            entities.append(
                {"id": target, "node_type": "File", "name": posixpath.basename(path), "path": path}
            )
            relationships.append(
                {"source": share_node["id"], "target": target, "relationship": "sharesResource"}
            )
    return _ingest(entities, relationships, ingest_entities=ingest_entities)


def ingest_calendar_events(
    events: list[dict[str, Any]],
    *,
    calendar: str | None = None,
    ingest_entities: Any | None = None,
) -> dict[str, int] | None:
    """Map CalDAV events → ``:CalendarEvent`` nodes."""
    entities: list[dict[str, Any]] = []
    for event in events or []:
        href = event.get("href") or event.get("url")
        if not href:
            continue
        entities.append(
            {
                "id": f"nextcloud:event:{href}",
                "node_type": "CalendarEvent",
                "name": event.get("name") or event.get("summary"),
                "url": event.get("url"),
                "calendar": calendar,
            }
        )
    return _ingest(entities, None, ingest_entities=ingest_entities)
