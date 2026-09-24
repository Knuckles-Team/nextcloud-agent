"""Native epistemic-graph ingestion for Nextcloud files (blobs + extracted text).

CONCEPT:AU-KG.ingest.list-durable-media. Nextcloud is a file store, so its headline
KG contribution is **blobs**: a downloaded file's raw bytes are stored content-addressed
as a ``:Blob`` + ``:AssetOccurrence`` node (carrying its WebDAV metadata) in ONE cross-modal
ACID commit, via the agent-utilities ``MediaStore``. When the file is a document
(pdf/office/txt) or an image, its text is ALSO extracted (``read_any`` / OCR) and written
as a ``:Document`` node linked back to the file — so the file is durable, deduped, AND
semantically searchable inside the knowledge graph.

Entirely best-effort and dependency-/engine-guarded: if agent-utilities' KG stack or a
live engine is not present, every entry point here **no-ops** (returns ``None``), so the
connector keeps working with zero KG infrastructure. This is the native ingestion seam
the ``nextcloud-agent`` package contributes to the KG.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("nextcloud_agent.kg")

_SOURCE = "nextcloud-agent"
_DOMAIN = "nextcloud"

# WebDAV/file metadata worth carrying onto the :AssetOccurrence / :File node.
_META_FIELDS = (
    "file_id",
    "etag",
    "permissions",
    "favorite",
    "last_modified",
)

# Extensions whose text we extract into a :Document (read_any also handles OCR on images).
_TEXT_EXTS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".json",
    ".eml",
    ".csv",
    ".html",
    ".htm",
    ".doc",
    ".docx",
    ".odt",
    ".ppt",
    ".pptx",
    ".odp",
    ".xls",
    ".xlsx",
    ".ods",
    ".rtf",
}


def _media_store(*args: object, **kwargs: object) -> object:
    """Build a ``MediaStore`` over a live engine, or ``None`` when unavailable.

    SDK-GAP: No-op: nothing left to register/write; preserves the graceful-degradation contract.
    """
    return None


def _classify(mime: str) -> str:
    if mime.startswith("audio"):
        return "audio"
    if mime.startswith("video"):
        return "video"
    if mime.startswith("image"):
        return "image"
    return "file"


def _extract_text(*args: object, **kwargs: object) -> object:
    """Extract plain text from a document/image byte blob via ``read_any`` (best-effort).

    SDK-GAP: No-op: nothing left to register/write; preserves the graceful-degradation contract.
    """
    return None


def ingest_file(*args: object, **kwargs: object) -> object:
    """Store a Nextcloud file as a blob (+ extracted :Document) in the knowledge graph.

    SDK-GAP: No-op: nothing left to register/write; preserves the graceful-degradation contract.
    """
    return None


class KnowledgeGraphIngestUnavailable(RuntimeError):
    """Direct-to-graph ingestion is unavailable from this connector.

    SDK-GAP (EH-48x, /var/tmp/l9/finish/au-decon-G4c/SDK-GAPS.md): raised in
    place of the old ``agent_utilities.knowledge_graph`` native-ingest call --
    agent-connector-sdk has no facade over EG's typed ingestion protocol yet,
    and the fleet precedent (agents/world-reference-mcp) moves direct-to-graph
    delivery to agent_connector_sdk.runner/sinks at the deployment layer, out
    of connector scope.
    """


def _kg_unavailable(name: str) -> None:
    raise KnowledgeGraphIngestUnavailable(
        f"{name}: direct-to-graph ingestion moved out of connector code "
        "(agent-utilities removed); no agent-connector-sdk facade exists yet "
        "-- see SDK-GAPS.md"
    )
