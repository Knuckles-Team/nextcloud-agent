"""Bounded XML parsing for untrusted WebDAV and CalDAV responses."""

from __future__ import annotations

from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

MAX_XML_BYTES = 8 * 1024 * 1024
MAX_XML_DEPTH = 64
MAX_XML_ELEMENTS = 100_000


class XmlSecurityError(ValueError):
    """A remote XML document crossed a parser security boundary."""


def parse_untrusted_xml(value: str | bytes | bytearray):
    """Parse bounded XML with entities, DTDs, and external references disabled."""

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    else:
        raise XmlSecurityError("remote XML response has an invalid representation")
    if not payload or len(payload) > MAX_XML_BYTES:
        raise XmlSecurityError("remote XML response exceeds its safe size boundary")
    try:
        root = DefusedET.fromstring(
            payload,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except (DefusedET.ParseError, DefusedXmlException, ValueError):
        raise XmlSecurityError("remote XML response is invalid") from None

    count = 0
    stack = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > MAX_XML_ELEMENTS or depth > MAX_XML_DEPTH:
            raise XmlSecurityError("remote XML response exceeds its structure boundary")
        stack.extend((child, depth + 1) for child in element)
    return root
