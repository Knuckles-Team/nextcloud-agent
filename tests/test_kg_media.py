"""Native epistemic-graph ingestion — Wire-First live-path coverage for nextcloud files.

Exercises ``nextcloud_agent.kg_media.ingest_file`` with a fake ``MediaStore`` and a fake
document writer (no engine required) and asserts a file is stored as a blob and, when it
is a document/image, its extracted text is written as a linked :Document.
CONCEPT:AU-KG.ingest.list-durable-media.
"""

from __future__ import annotations

from dataclasses import dataclass

from nextcloud_agent.kg_media import ingest_file


@dataclass
class _Stored:
    asset_id: str
    digest: str


class _FakeMediaStore:
    """Captures the store_media call the way the real MediaStore is invoked."""

    def __init__(self):
        self.calls = []

    def store_media(self, data, **kw):
        self.calls.append((data, kw))
        return _Stored(asset_id="media:cafebabe", digest="cafebabe")


class _FakeDocWriter:
    """Captures ingest_documents(documents, source=, domain=) calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, documents, *, source, domain):
        self.calls.append((documents, source, domain))
        return {"nodes": len(documents), "edges": 0}


def test_ingest_file_stores_blob_and_extracted_document(monkeypatch):
    store = _FakeMediaStore()
    docs = _FakeDocWriter()

    # Fake the text extractor so no real reader dependency is needed.
    monkeypatch.setattr(
        "nextcloud_agent.kg_media._extract_text",
        lambda data, remote_path, mime: "extracted body text",
    )

    res = ingest_file(
        b"%PDF-1.7 fake pdf bytes",
        remote_path="Documents/report.pdf",
        metadata={"file_id": "123", "etag": "abc"},
        media_store=store,
        ingest_documents=docs,
    )

    assert res is not None
    assert res["asset_id"] == "media:cafebabe"
    assert res["digest"] == "cafebabe"
    assert res["media_type"] == "file"
    assert res["size_bytes"] == len(b"%PDF-1.7 fake pdf bytes")
    assert res["doc_id"] == "nextcloud:document:cafebabe"

    # Blob stored with the raw bytes + propagated WebDAV metadata.
    assert len(store.calls) == 1
    data, kw = store.calls[0]
    assert data == b"%PDF-1.7 fake pdf bytes"
    assert kw["source"] == "nextcloud-agent"
    assert kw["mime_type"] == "application/pdf"
    assert kw["name"] == "report.pdf"
    assert kw["extra"]["file_id"] == "123"
    assert kw["extra"]["remote_path"] == "Documents/report.pdf"

    # Extracted text written as a linked :Document.
    assert len(docs.calls) == 1
    documents, source, domain = docs.calls[0]
    assert source == "nextcloud-agent"
    assert domain == "nextcloud"
    doc = documents[0]
    assert doc["id"] == "nextcloud:document:cafebabe"
    assert doc["text"] == "extracted body text"
    assert doc["source_uri"] == "nextcloud://Documents/report.pdf"
    assert doc["hasBlob"] == "media:cafebabe"


def test_ingest_file_binary_stores_blob_without_document(monkeypatch):
    """A non-document binary (e.g. a zip) is stored as a blob, no :Document."""
    store = _FakeMediaStore()
    docs = _FakeDocWriter()
    monkeypatch.setattr(
        "nextcloud_agent.kg_media._extract_text",
        lambda data, remote_path, mime: None,
    )

    res = ingest_file(
        b"PK\x03\x04zip",
        remote_path="Archives/backup.zip",
        media_store=store,
        ingest_documents=docs,
    )
    assert res is not None
    assert "doc_id" not in res
    assert len(store.calls) == 1
    assert len(docs.calls) == 0


def test_ingest_file_reads_local_path(tmp_path):
    f = tmp_path / "note.txt"
    f.write_bytes(b"hello world")
    store = _FakeMediaStore()

    res = ingest_file(
        str(f),
        remote_path="note.txt",
        media_store=store,
        ingest_documents=_FakeDocWriter(),
    )
    assert res is not None
    data, kw = store.calls[0]
    assert data == b"hello world"
    assert kw["mime_type"] == "text/plain"


def test_ingest_file_noops_without_engine():
    """No injected store + no reachable engine -> clean no-op (never raises)."""
    assert ingest_file(b"data", remote_path="x.bin") is None


def test_ingest_file_noops_on_missing_path():
    assert (
        ingest_file(
            "/no/such/file.pdf", remote_path="x.pdf", media_store=_FakeMediaStore()
        )
        is None
    )


def test_ingest_file_noops_on_empty_bytes():
    assert ingest_file(b"", remote_path="x.bin", media_store=_FakeMediaStore()) is None
