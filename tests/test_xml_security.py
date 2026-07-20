"""Security regression coverage for untrusted WebDAV XML."""

import pytest

from nextcloud_agent.api import api_client_base, xml_security


def test_dtd_and_external_entity_are_rejected():
    payload = (
        b'<!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///ignored">]><root>&xxe;</root>'
    )
    with pytest.raises(xml_security.XmlSecurityError, match="invalid"):
        xml_security.parse_untrusted_xml(payload)


def test_structure_depth_is_bounded(monkeypatch):
    monkeypatch.setattr(xml_security, "MAX_XML_DEPTH", 3)
    with pytest.raises(xml_security.XmlSecurityError, match="structure"):
        xml_security.parse_untrusted_xml(b"<a><b><c><d/></c></b></a>")


def test_response_size_is_bounded(monkeypatch):
    monkeypatch.setattr(xml_security, "MAX_XML_BYTES", 8)
    with pytest.raises(xml_security.XmlSecurityError, match="size"):
        xml_security.parse_untrusted_xml(b"<root>toolarge</root>")


def test_streamed_response_is_bounded_and_closed(monkeypatch):
    class Response:
        headers = {}
        closed = False

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def iter_content(chunk_size):
            assert chunk_size > 0
            yield b"12345"
            yield b"67890"

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(api_client_base, "MAX_XML_BYTES", 8)
    with pytest.raises(xml_security.XmlSecurityError, match="size"):
        api_client_base.BaseApiClient._read_xml_response(response)
    assert response.closed


def test_service_origin_rejects_credentials_and_plaintext_remote_hosts():
    with pytest.raises(ValueError, match="invalid"):
        api_client_base._origin("https://user:secret@example.test/")
    with pytest.raises(ValueError, match="loopback"):
        api_client_base._origin("http://example.test/")


def test_client_rejects_cross_origin_references_and_path_traversal():
    client = api_client_base.BaseApiClient(
        "https://cloud.example.test", "account", "credential"
    )
    assert client._session.trust_env is False
    with pytest.raises(ValueError, match="origin"):
        client._get_absolute_url("https://attacker.example.test/dav")
    with pytest.raises(ValueError, match="path"):
        client._get_full_url("folder/../secret")
