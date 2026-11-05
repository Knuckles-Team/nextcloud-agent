"""CONN-NEXTCLOUD-AGENT: the file tools run through the SDK WebDAV backend."""

from urllib.parse import unquote, urlsplit

import httpx
import pytest
from scion_sdk.storage import WebDavBackend

from nextcloud_agent.api_client import NextcloudAPI

ROOT = "/remote.php/dav/files/svc"


class _Dav:
    """A minimal in-memory WebDAV double."""

    def __init__(self):
        self.files, self.cols, self.requests = {}, {""}, []

    def _path(self, url):
        return unquote(urlsplit(str(url)).path)[len(ROOT) :].strip("/")

    def _resp(self, path):
        href = f"{ROOT}/{path}".rstrip("/") + ("/" if path in self.cols else "")
        if path in self.cols:
            props = "<d:resourcetype><d:collection/></d:resourcetype>"
        else:
            props = f"<d:resourcetype/><d:getcontentlength>{len(self.files[path])}</d:getcontentlength>"
        props += "<oc:permissions>RGDNVW</oc:permissions><oc:favorite>1</oc:favorite><oc:fileid>7</oc:fileid>"
        return f"<d:response><d:href>{href}</d:href><d:propstat><d:prop>{props}</d:prop></d:propstat></d:response>"

    def handle(self, request):
        path, method = self._path(request.url), request.method
        self.requests.append((method, path))
        if method == "PROPFIND":
            if path not in self.cols and path not in self.files:
                return httpx.Response(404)
            items = (
                [path]
                + sorted(
                    p
                    for p in (*self.files, *self.cols)
                    if p and p.rsplit("/", 1)[0] == path and p != path
                )
                if path
                else [path]
                + sorted(p for p in (*self.files, *self.cols) if p and "/" not in p)
            )
            body = (
                '<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">'
                + "".join(self._resp(p) for p in items)
                + "</d:multistatus>"
            )
            return httpx.Response(207, content=body.encode())
        if method == "MKCOL":
            if path in self.cols:
                return httpx.Response(405)
            self.cols.add(path)
            return httpx.Response(201)
        if method == "PUT":
            if request.headers.get("If-None-Match") == "*" and path in self.files:
                return httpx.Response(412)
            self.files[path] = request.read()
            return httpx.Response(201)
        if method == "GET":
            return (
                httpx.Response(200, content=self.files[path])
                if path in self.files
                else httpx.Response(404)
            )
        if method == "DELETE":
            return (
                httpx.Response(204)
                if self.files.pop(path, None) is not None
                else httpx.Response(404)
            )
        if method in ("COPY", "MOVE"):
            target = self._path(request.headers["Destination"])
            self.files[target] = self.files[path]
            if method == "MOVE":
                del self.files[path]
            return httpx.Response(201)
        return httpx.Response(405)


def _client(dav):
    client = NextcloudAPI(
        base_url="https://cloud.example", username="svc", password="pw"
    )
    client._files_dav = WebDavBackend(
        client.webdav_base,
        username="svc",
        password="pw",
        transport=httpx.MockTransport(dav.handle),
    )
    return client


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R001")
def test_file_methods_route_through_sdk_webdav_backend():
    dav = _Dav()
    client = _client(dav)
    assert client.write_file("Docs/a.txt", "hello") is True
    assert [m for m, _ in dav.requests][:2] == ["MKCOL", "PUT"]
    with pytest.raises(FileExistsError):
        client.write_file("Docs/a.txt", "again", overwrite=False)
    assert client.read_file("Docs/a.txt") == b"hello"
    listing = {e["name"]: e for e in client.list_files("Docs")}
    entry = listing["a.txt"]
    assert (
        entry["permissions"] == "RGDNVW"
        and entry["favorite"] == "1"
        and entry["file_id"] == "7"
    )
    assert entry["is_folder"] is False and entry["content_length"] == "5"
    client.copy_item("Docs/a.txt", "Docs/b.txt")
    client.move_item("Docs/b.txt", "Docs/c.txt")
    client.delete_item("Docs/c.txt")
    assert set(dav.files) == {"Docs/a.txt"}
    with pytest.raises(FileNotFoundError):
        client.read_file("Docs/missing.txt")
    with pytest.raises(ValueError):
        client.read_file("../escape")


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R002")
def test_backend_uses_files_root_and_private_host_setting(monkeypatch):
    client = NextcloudAPI(
        base_url="https://cloud.example", username="svc", password="pw"
    )
    assert client.webdav_base.endswith("/remote.php/dav/files/svc")
    monkeypatch.delenv("NEXTCLOUD_ALLOWED_PRIVATE_HOSTS", raising=False)
    private = NextcloudAPI(base_url="https://10.9.8.7", username="svc", password="pw")
    with pytest.raises(httpx.TransportError, match="rejected"):
        private.read_file("a.txt")
    monkeypatch.setenv("NEXTCLOUD_ALLOWED_PRIVATE_HOSTS", "10.9.8.7")
    allowed = NextcloudAPI(base_url="https://10.9.8.7", username="svc", password="pw")
    assert allowed.files_dav is not None


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-R003")
def test_sdk_pin_carries_storage_backend():
    from scion_sdk.storage import WebDavBackend as Backend

    assert Backend.backend_type == "webdav"
