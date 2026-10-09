"""Structure-node ingestion coverage for nextcloud_agent.kg_ingest (typed OWL nodes),
via agent_connector_sdk — exercised against a fake transport one level below the SDK's
own ``KnowledgeIngest`` facade, so these tests still run the SDK's real
request-building contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from agent_connector_sdk.ingest import IngestError, KnowledgeIngest

from nextcloud_agent.kg_ingest import (
    ingest_calendar_events,
    ingest_listing,
    ingest_shares,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def source_status(self, connector, stream):
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data):
        raise AssertionError("nextcloud-agent structure ingestion carries no media")


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


def _node_type(record: Any) -> str:
    """A real generated ``SourceRecord`` has no bare ``node_type`` field — it's the
    last segment of ``mapping_reference``
    (``manifest:<connector>#schema_mappings/<NodeType>``)."""
    return record.mapping_reference.rsplit("/", 1)[-1]


def _rel_name(relationship: Any) -> str:
    """Likewise, a ``SourceRelationship``'s kind is the last segment of
    ``relation_reference`` (``manifest:<connector>#resources/<NodeType>/relations/<kind>``)."""
    return relationship.relation_reference.rsplit("/", 1)[-1]


async def test_ingest_listing_maps_files_and_folders(ingest):
    service, transport = ingest
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
    res = await ingest_listing(entries, parent_path="Documents", ingest=service)
    assert res == {"nodes": 3, "edges": 2}  # parent folder + file + subfolder, 2 inFolder

    records = {r.record_id: r for r in transport.requests[0].records}
    file_node = records["nextcloud:file:10"]
    assert _node_type(file_node) == "File"
    assert file_node.payload["mimeType"] == "application/pdf"
    assert file_node.payload["sizeBytes"] == 2048
    # The SDK's PersistencePrivacyGuard (IngestBinding(sanitize=True), the default)
    # redacts location-shaped fields -- "path" among them -- before they leave the
    # process. A real, deliberate improvement this migration picks up for free: file
    # paths no longer land in the knowledge graph in plaintext.
    assert file_node.payload["path"] == "[REDACTED_LOCATION]"
    assert file_node.record_id == "nextcloud:file:10"  # structural id field: untouched
    rels = {_rel_name(r) for r in transport.requests[0].relationships}
    assert rels == {"inFolder"}


async def test_ingest_shares_maps_share_and_resource(ingest):
    service, transport = ingest
    res = await ingest_shares(
        [{"id": "42", "path": "/Documents/report.pdf", "share_type": 3, "share_with": None}],
        ingest=service,
    )
    assert res is not None
    records = {r.record_id: r for r in transport.requests[0].records}
    share = records["nextcloud:share:42"]
    assert _node_type(share) == "Share"
    assert _rel_name(transport.requests[0].relationships[0]) == "sharesResource"


async def test_ingest_calendar_events_maps_events(ingest):
    service, transport = ingest
    res = await ingest_calendar_events(
        [{"href": "/cal/e1.ics", "name": "e1.ics"}],
        calendar="personal",
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    record = transport.requests[0].records[0]
    assert _node_type(record) == "CalendarEvent"
    assert record.payload["calendar"] == "personal"


async def test_empty_native_ingest_is_rejected(ingest):
    service, _transport = ingest
    with pytest.raises(IngestError, match="at least one entity"):
        await ingest_listing([], ingest=service)


async def test_ingest_shares_and_events_empty_is_a_noop(ingest):
    service, transport = ingest
    assert await ingest_shares([], ingest=service) is None
    assert await ingest_calendar_events([], ingest=service) is None
    assert transport.requests == []
