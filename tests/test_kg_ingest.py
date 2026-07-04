"""Structure-node ingestion coverage for nextcloud_agent.kg_ingest (typed OWL nodes)."""

from __future__ import annotations

from nextcloud_agent.kg_ingest import (
    ingest_calendar_events,
    ingest_listing,
    ingest_shares,
)


class _FakeEntityWriter:
    """Captures ingest_entities(entities, relationships, source=, domain=) calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, entities, relationships=None, *, source, domain):
        self.calls.append((entities, relationships, source, domain))
        return {"nodes": len(entities), "edges": len(relationships or [])}


def test_ingest_listing_maps_files_and_folders():
    writer = _FakeEntityWriter()
    entries = [
        {
            "name": "report.pdf",
            "is_folder": False,
            "file_id": "10",
            "content_type": "application/pdf",
            "content_length": "2048",
            "last_modified": "Mon, 01 Jan 2026 00:00:00 GMT",
        },
        {"name": "sub", "is_folder": True, "file_id": "11"},
    ]
    res = ingest_listing(entries, parent_path="Documents", ingest_entities=writer)
    assert res == {"nodes": 3, "edges": 2}  # parent folder + file + subfolder, 2 inFolder

    entities, relationships, source, domain = writer.calls[0]
    assert source == "nextcloud-agent"
    assert domain == "nextcloud"
    file_node = next(e for e in entities if e["type"] == "File")
    assert file_node["id"] == "nextcloud:file:10"
    assert file_node["mimeType"] == "application/pdf"
    assert file_node["sizeBytes"] == 2048
    assert file_node["path"] == "Documents/report.pdf"
    assert {r["type"] for r in relationships} == {"inFolder"}


def test_ingest_shares_maps_share_and_resource():
    writer = _FakeEntityWriter()
    res = ingest_shares(
        [{"id": "42", "path": "/Documents/report.pdf", "share_type": 3, "share_with": None}],
        ingest_entities=writer,
    )
    assert res is not None
    entities, relationships, _, _ = writer.calls[0]
    share = next(e for e in entities if e["type"] == "Share")
    assert share["id"] == "nextcloud:share:42"
    assert relationships[0]["type"] == "sharesResource"


def test_ingest_calendar_events_maps_events():
    writer = _FakeEntityWriter()
    res = ingest_calendar_events(
        [{"href": "/cal/e1.ics", "name": "e1.ics"}],
        calendar="personal",
        ingest_entities=writer,
    )
    assert res == {"nodes": 1, "edges": 0}
    entities, _, _, _ = writer.calls[0]
    assert entities[0]["type"] == "CalendarEvent"
    assert entities[0]["calendar"] == "personal"


def test_ingest_empty_noops():
    assert ingest_listing([], ingest_entities=_FakeEntityWriter()) is None
