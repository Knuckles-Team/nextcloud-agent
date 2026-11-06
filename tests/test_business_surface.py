"""CONN-NEXTCLOUD-AGENT R004-R015: the business surface, one typed tool per operation."""

import asyncio
import json

import pytest

from nextcloud_agent.api_client import NextcloudAPI


class _Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body if isinstance(body, bytes) else body.encode()
        self.headers = headers or {}
        self.text = self._body.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        yield self._body

    def close(self):
        pass


def _ocs(data):
    return json.dumps({"ocs": {"meta": {"status": "ok", "statuscode": 200}, "data": data}})


class _Session:
    """Records each request and answers from a route table keyed by (method, path suffix)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for (verb, suffix), answer in self.routes.items():
            if verb == method and url.split("?")[0].endswith(suffix):
                return answer() if callable(answer) else answer
        return _Resp(404)

    def close(self):
        pass


def _client(routes):
    client = NextcloudAPI(base_url="https://cloud.example", username="svc", password="pw")
    client._session = _Session(routes)
    return client


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R015")
def test_every_operation_is_its_own_typed_tool():
    from scion_sdk.connector import build_server

    _, server = build_server("nextcloud_agent", command_args=["--transport", "stdio"])
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    for name in (
        "nextcloud_list_versions",
        "nextcloud_search_files",
        "nextcloud_create_share",
        "nextcloud_update_event",
        "nextcloud_create_task",
        "nextcloud_update_contact",
        "nextcloud_send_message",
        "nextcloud_move_card",
        "nextcloud_create_note",
        "nextcloud_list_activities",
        "nextcloud_create_user",
        "nextcloud_create_document",
    ):
        assert name in names, name
    assert not any("action" in (t.parameters or {}).get("properties", {}) for t in tools)
    assert not any(n.endswith(("_resource", "_contents", "_directory")) for n in names)


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R004")
def test_versions_search_and_trash_use_the_dav_endpoints():
    listing = (
        '<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:response>'
        "<d:href>/remote.php/dav/files/svc/a.txt</d:href><d:propstat><d:prop>"
        "<oc:fileid>9</oc:fileid></d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat>"
        "</d:response></d:multistatus>"
    )
    versions = (
        '<d:multistatus xmlns:d="DAV:"><d:response><d:href>/v/9/</d:href></d:response>'
        "<d:response><d:href>/remote.php/dav/versions/svc/versions/9/100</d:href><d:propstat>"
        "<d:prop><d:getcontentlength>2</d:getcontentlength></d:prop>"
        "<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
        "<d:response><d:href>/remote.php/dav/versions/svc/versions/9/200</d:href></d:response>"
        "</d:multistatus>"
    )
    client = _client(
        {
            ("PROPFIND", "/versions/svc/versions/9"): lambda: _Resp(207, versions),
            ("SEARCH", "/remote.php/dav/"): lambda: _Resp(207, listing),
            ("MOVE", "/versions/svc/versions/9/100"): _Resp(201),
        }
    )

    class _Entry:
        props = {"fileid": "9"}
        is_collection, name, path, last_modified, etag, content_type, size = (
            False, "a.txt", "a.txt", "", "", "", 2,
        )

    client._files_dav = type("D", (), {"propfind": lambda *a, **k: [_Entry()], "close": lambda s: None})()
    rows = client.list_versions("a.txt")
    assert [r["version_id"] for r in rows] == ["200", "100"]
    assert client.restore_version("a.txt", "100") is True
    move = [c for c in client._session.calls if c[0] == "MOVE"][0]
    assert move[2]["headers"]["Destination"].endswith("/versions/svc/restore/target")
    found = client.search_files("a", mime_prefix="image/")
    assert found[0]["path"] == "a.txt"
    body = [c for c in client._session.calls if c[0] == "SEARCH"][0][2]["data"]
    assert "%a%" in body and "image/%" in body and "/files/svc" in body


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R005")
def test_shares_cover_public_links_and_internal_grants():
    client = _client({("POST", "/shares"): lambda: _Resp(200, _ocs({"id": 5, "url": "https://x/s/abc"}))})
    share = client.create_share("Team/a.png", "public_link", password="pw1", expire_date="2026-12-01")
    assert share["url"].endswith("/s/abc")
    sent = client._session.calls[-1][2]["data"]
    assert sent["shareType"] == 3 and sent["expireDate"] == "2026-12-01" and sent["sendMail"] == "false"
    with pytest.raises(ValueError):
        client.create_share("Team", "group")
    client.create_share("Team", "group", share_with="staff", permissions=31)
    assert client._session.calls[-1][2]["data"]["shareWith"] == "staff"
    with pytest.raises(ValueError):
        client.update_share("../1", note="x")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R006")
def test_event_invites_stay_drafts_unless_asked():
    stored = {}

    def put():
        return _Resp(201, headers={"ETag": '"e1"'})

    client = _client(
        {
            ("PUT", ".ics"): put,
            ("GET", "/cloud/user"): lambda: _Resp(200, _ocs({"email": "me@example.com"})),
        }
    )
    cal = "https://cloud.example/remote.php/dav/calendars/svc/work/"
    result = client.create_event(cal, "Kickoff", "2026-11-20T15:00:00Z", attendees=["a@example.com"])
    method, url, kwargs = [c for c in client._session.calls if c[0] == "PUT"][-1]
    ics = kwargs["data"].decode()
    stored["ics"] = ics
    assert url.startswith(cal) and result["uid"] in url
    assert "SCHEDULE-AGENT=CLIENT" in ics.replace("\r\n ", "")
    assert kwargs["headers"]["If-None-Match"] == "*"
    client.create_event(cal, "Real", "2026-11-20T15:00:00Z", attendees=["a@example.com"], send_invitations=True)
    assert "SCHEDULE-AGENT" not in client._session.calls[-1][2]["data"].decode().replace("\r\n ", "")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R007")
def test_tasks_are_vtodos_and_complete_with_etag():
    todo = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:x\r\nBEGIN:VTODO\r\nUID:t1\r\n"
        "SUMMARY:Ship\r\nSTATUS:NEEDS-ACTION\r\nDTSTAMP:20261106T000000Z\r\nEND:VTODO\r\nEND:VCALENDAR\r\n"
    )
    client = _client(
        {
            ("GET", "t1.ics"): lambda: _Resp(200, todo, {"ETag": '"v1"'}),
            ("PUT", "t1.ics"): lambda: _Resp(204, headers={"ETag": '"v2"'}),
        }
    )
    url = "https://cloud.example/remote.php/dav/calendars/svc/tasks/t1.ics"
    client.update_task(url, completed=True)
    _, _, kwargs = client._session.calls[-1]
    assert kwargs["headers"]["If-Match"] == '"v1"'
    assert "STATUS:COMPLETED" in kwargs["data"].decode()


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R008")
def test_contacts_round_trip_as_vcards():
    client = _client({("PUT", ".vcf"): lambda: _Resp(201, headers={"ETag": '"c1"'})})
    book = "https://cloud.example/remote.php/dav/addressbooks/users/svc/contacts/"
    made = client.create_contact(book, "Ada Lovelace", emails=["ada@example.com"], organization="Analytical")
    vcf = client._session.calls[-1][2]["data"]
    assert "FN:Ada Lovelace" in vcf and "ada@example.com" in vcf and made["uid"] in made["url"]
    with pytest.raises(ValueError):
        client.delete_contact("https://cloud.example/remote.php/dav/files/svc/x.vcf")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R009")
def test_talk_conversations_and_messages():
    client = _client(
        {
            ("POST", "/api/v4/room"): lambda: _Resp(200, _ocs({"token": "abc123", "type": 2})),
            ("POST", "/api/v1/chat/abc123"): lambda: _Resp(201, _ocs({"id": 7})),
        }
    )
    room = client.create_conversation("ops", "group")
    assert client._session.calls[-1][2]["data"]["roomType"] == 2
    assert client.send_message(room["token"], "hi")["id"] == 7
    with pytest.raises(ValueError):
        client.send_message("../x", "hi")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R010")
def test_deck_move_uses_the_target_stack_route():
    client = _client({("PUT", "/stacks/9/cards/3/reorder"): lambda: _Resp(200, "{}")})
    client.move_card(1, 3, 9)
    assert client._session.calls[-1][1].endswith("/boards/1/stacks/9/cards/3/reorder")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R011")
def test_notes_crud_uses_the_notes_api():
    client = _client({("POST", "/notes/api/v1/notes"): lambda: _Resp(200, '{"id": 4}')})
    assert client.create_note("t", "body")["id"] == 4
    with pytest.raises(ValueError):
        client.update_note(4)


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R012")
def test_activity_not_modified_reads_as_empty():
    import requests

    client = _client({})

    def raise_redirect(*a, **k):
        raise requests.TooManyRedirects("304")

    client._session.request = raise_redirect
    assert client.list_activities() == []


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R013")
def test_admin_creates_users_into_groups():
    client = _client({("POST", "/cloud/users"): lambda: _Resp(200, _ocs({"id": "zz"}))})
    with pytest.raises(ValueError):
        client.create_user("zz")
    client.create_user("zz", email="zz@example.com", groups=["staff"])
    assert client._session.calls[-1][2]["data"]["groups[]"] == ["staff"]
    with pytest.raises(ValueError):
        client.get_user("a/b")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R014")
def test_office_documents_open_in_the_matching_editor():
    client = _client(
        {
            ("POST", "/templates/create"): lambda: _Resp(200, _ocs({"fileid": 3})),
            ("POST", "/richdocuments/api/v1/document"): lambda: _Resp(200, _ocs({"url": "https://x/direct/t"})),
            ("POST", "/directEditing/open"): lambda: _Resp(200, _ocs({"url": "https://x/de/t"})),
        }
    )
    client.create_document("Team/plan.docx")
    assert client._session.calls[-1][2]["data"]["filePath"] == "/Team/plan.docx"
    client.get_properties = lambda path: {"file_id": "3"}
    assert client.open_document("Team/plan.docx")["url"].endswith("/direct/t")
    assert client.open_document("Team/notes.md")["url"].endswith("/de/t")
