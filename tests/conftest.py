import sys
from unittest.mock import MagicMock, patch

import pytest

# Hermetically mock tree_sitter and all potential tree_sitter language modules before any imports happen
sys.modules["tree_sitter"] = MagicMock()
for lang in [
    "javascript",
    "python",
    "typescript",
    "tsx",
    "go",
    "rust",
    "cpp",
    "c",
    "bash",
    "html",
    "css",
    "json",
]:
    sys.modules[f"tree_sitter_{lang}"] = MagicMock()


@pytest.fixture
def mock_session():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value
        session.status_code = 200
        session.put_status_code = None
        session.post_status_code = None
        session.delete_status_code = None
        session.request_status_code = None
        session.response_content = None
        session.response_json = None

        def mock_request(method, url, *args, **kwargs):
            response = MagicMock()

            # Resolve dynamic status code
            status_code = getattr(session, "status_code", 200)
            if (
                method == "PUT"
                and getattr(session, "put_status_code", None) is not None
            ):
                status_code = session.put_status_code
            elif (
                method == "POST"
                and getattr(session, "post_status_code", None) is not None
            ):
                status_code = session.post_status_code
            elif (
                method == "DELETE"
                and getattr(session, "delete_status_code", None) is not None
            ):
                status_code = session.delete_status_code
            elif getattr(session, "request_status_code", None) is not None:
                status_code = session.request_status_code

            response.status_code = status_code

            # Resolve dynamic response content
            if getattr(session, "response_content", None) is not None:
                response.content = session.response_content
            else:
                # Safe string parsing of request data
                data = kwargs.get("data")
                data_str = ""
                if data:
                    if isinstance(data, bytes):
                        data_str = data.decode("utf-8", errors="ignore")
                    else:
                        data_str = str(data)

                # Custom XML payloads based on URLs and bodies
                if "calendars" in url:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                        <d:response>
                            <d:href>/remote.php/dav/calendars/testuser/work/</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:displayname>Work Calendar</d:displayname>
                                    <d:resourcetype>
                                        <c:calendar/>
                                    </d:resourcetype>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                        <d:response>
                            <d:href>/remote.php/dav/calendars/testuser/work/event1.ics</d:href>
                        </d:response>
                    </d:multistatus>"""
                elif "addressbooks" in url:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
                        <d:response>
                            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:displayname>Contacts</d:displayname>
                                    <d:resourcetype>
                                        <c:addressbook/>
                                    </d:resourcetype>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                        <d:response>
                            <d:href>/remote.php/dav/addressbooks/users/testuser/contacts/contact1.vcf</d:href>
                        </d:response>
                    </d:multistatus>"""
                elif "quota" in data_str:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:">
                        <d:response>
                            <d:propstat>
                                <d:prop>
                                    <d:quota-available-bytes>5000000000</d:quota-available-bytes>
                                    <d:quota-used-bytes>1000000000</d:quota-used-bytes>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                    </d:multistatus>"""
                else:
                    response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
                    <d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
                        <d:response>
                            <d:href>/remote.php/dav/files/testuser/somefile.txt</d:href>
                            <d:propstat>
                                <d:prop>
                                    <d:getlastmodified>Fri, 22 May 2026 12:00:00 GMT</d:getlastmodified>
                                    <d:getcontentlength>1234</d:getcontentlength>
                                    <d:getcontenttype>text/plain</d:getcontenttype>
                                    <oc:permissions>WDNVR</oc:permissions>
                                    <d:resourcetype/>
                                    <d:getetag>"abcdef"</d:getetag>
                                    <oc:favorite>1</oc:favorite>
                                    <oc:fileid>12345</oc:fileid>
                                </d:prop>
                            </d:propstat>
                        </d:response>
                    </d:multistatus>"""

            # Resolve dynamic response JSON
            if getattr(session, "response_json", None) is not None:
                response.json.return_value = session.response_json
            else:
                response.json.return_value = {
                    "ocs": {
                        "meta": {"status": "ok", "statuscode": 100, "message": "OK"},
                        "data": {"id": "1", "share_type": 3, "token": "abc"},
                    }
                }
            response.text = (
                '{"ocs": {"meta": {"status": "ok", "statuscode": 100}, "data": {}}}'
            )
            return response

        session.request.side_effect = mock_request
        session.get.side_effect = lambda url, *a, **k: mock_request("GET", url, *a, **k)
        session.post.side_effect = lambda url, *a, **k: mock_request(
            "POST", url, *a, **k
        )
        session.put.side_effect = lambda url, *a, **k: mock_request("PUT", url, *a, **k)
        session.delete.side_effect = lambda url, *a, **k: mock_request(
            "DELETE", url, *a, **k
        )
        yield session
