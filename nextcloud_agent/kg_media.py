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
import mimetypes
import os
import tempfile
from typing import Any

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


def _media_store() -> Any | None:
    """Build a ``MediaStore`` over a live engine, or ``None`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.memory.native_ingest import media_store

        return media_store()
    except Exception as e:  # noqa: BLE001 — agent-utilities KG stack absent
        logger.debug("Operation failed: error_type=%s", type(e).__name__)
        return None


def _classify(mime: str) -> str:
    if mime.startswith("audio"):
        return "audio"
    if mime.startswith("video"):
        return "video"
    if mime.startswith("image"):
        return "image"
    return "file"


def _extract_text(data: bytes, remote_path: str, mime: str) -> str | None:
    """Extract plain text from a document/image byte blob via ``read_any`` (best-effort)."""
    ext = os.path.splitext(remote_path)[1].lower()
    is_image = mime.startswith("image")
    if ext not in _TEXT_EXTS and not is_image:
        return None
    try:
        from agent_utilities.knowledge_graph.extraction.readers import read_any
    except Exception as e:  # noqa: BLE001 — extraction stack absent
        logger.debug(
            "KG text extraction unavailable: error_type=%s", type(e).__name__
        )
        return None
    tmp_path = None
    try:
        suffix = ext or mimetypes.guess_extension(mime) or ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp_path = fh.name
        text = read_any(tmp_path, mime=mime)  # never raises; "" when it can't read
    except Exception as e:  # noqa: BLE001 — extraction failure is non-fatal
        logger.debug("KG text extraction failed: error_type=%s", type(e).__name__)
        text = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return text or None


def ingest_file(
    path_or_bytes: str | bytes,
    *,
    remote_path: str,
    mime: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = _SOURCE,
    media_store: Any | None = None,
    ingest_documents: Any | None = None,
) -> dict[str, Any] | None:
    """Store a Nextcloud file as a blob (+ extracted :Document) in the knowledge graph.

    ``path_or_bytes``: a local filesystem path OR the raw file bytes (WebDAV download).
    ``remote_path``: the Nextcloud server-relative path (used for id, name, and MIME sniff).
    Returns ``{asset_id, digest, size_bytes, media_type, doc_id?}`` on success, or ``None``
    when there is no engine / no data / the store failed (never raises). ``media_store`` /
    ``ingest_documents`` may be injected (tests); otherwise resolved on demand.
    """
    # Resolve the bytes.
    if isinstance(path_or_bytes, bytes):
        data: bytes | None = path_or_bytes
    elif isinstance(path_or_bytes, str):
        if not path_or_bytes or not os.path.exists(path_or_bytes):
            return None
        try:
            with open(path_or_bytes, "rb") as fh:
                data = fh.read()
        except OSError as e:
            logger.warning("Operation failed: error_type=%s", type(e).__name__)
            return None
    else:
        return None
    if not data:
        return None

    store = media_store if media_store is not None else _media_store()
    if store is None:
        return None

    metadata = metadata or {}
    mime = mime or mimetypes.guess_type(remote_path)[0] or "application/octet-stream"
    media_type = _classify(mime)
    name = os.path.basename(remote_path.rstrip("/")) or remote_path

    extra = {k: metadata[k] for k in _META_FIELDS if metadata.get(k) is not None}
    extra["remote_path"] = remote_path

    try:
        stored = store.store_media(
            data,
            media_type=media_type,
            mime_type=mime,
            source=source,
            name=name,
            extra=extra,
        )
    except Exception as e:  # noqa: BLE001 — engine/store failure is non-fatal
        logger.warning("Operation failed: error_type=%s", type(e).__name__)
        return None
    if stored is None:
        return None

    result: dict[str, Any] = {
        "asset_id": stored.asset_id,
        "digest": stored.digest,
        "size_bytes": len(data),
        "media_type": media_type,
    }
    logger.info(
        "KG media ingest: stored %s (%s bytes) as asset %s digest %s",
        name,
        len(data),
        stored.asset_id,
        stored.digest[:16],
    )

    # Also extract + ingest the file's text as a linked :Document (searchable).
    text = _extract_text(data, remote_path, mime)
    if text and text.strip():
        doc_id = f"nextcloud:document:{stored.digest}"
        doc = {
            "id": doc_id,
            "title": name,
            "text": text,
            "source_uri": f"nextcloud://{remote_path.lstrip('/')}",
            "mimeType": mime,
            "path": remote_path,
            "hasBlob": stored.asset_id,
        }
        writer = ingest_documents
        if writer is None:
            try:
                from agent_utilities.knowledge_graph.memory.native_ingest import (
                    ingest_documents as writer,  # type: ignore[no-redef]
                )
            except Exception as e:  # noqa: BLE001 — document stack absent
                logger.debug("Operation failed: error_type=%s", type(e).__name__)
                writer = None
        if writer is not None:
            try:
                written = writer([doc], source=source, domain=_DOMAIN)
                if written:
                    result["doc_id"] = doc_id
            except Exception as e:  # noqa: BLE001 — document write is non-fatal
                logger.debug("Operation failed: error_type=%s", type(e).__name__)

    return result
