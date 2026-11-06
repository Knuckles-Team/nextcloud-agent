import ipaddress
import json
import logging
from collections.abc import Iterator
from typing import Any, Protocol
from urllib.parse import quote, urljoin, urlsplit

import requests
from scion_sdk.config import csv_values, setting
from scion_sdk.storage import WebDavBackend
from scion_sdk.tls.profile import ResolvedTLSProfile
from scion_sdk.tls.resolve import resolve_tls_profile

from nextcloud_agent.api.xml_security import (
    MAX_XML_BYTES,
    XmlSecurityError,
)

logger = logging.getLogger(__name__)


class _StreamedResponse(Protocol):
    """Minimal streamed-response contract ``_read_xml_response`` needs.

    Satisfied by a real ``requests.Response`` and by the lightweight fakes
    tests use in place of one.
    """

    headers: Any

    def raise_for_status(self) -> Any: ...
    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...
    def close(self) -> Any: ...


def _allowed_private_hosts() -> set[str]:
    """Hosts the operator admits for private addresses and plain http (R002)."""
    raw = setting("NEXTCLOUD_ALLOWED_PRIVATE_HOSTS", "")
    return {host.lower().rstrip(".") for host in csv_values(raw)}


def _origin(url: str, *, require_clean_base: bool = False) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except (TypeError, ValueError):
        raise ValueError("configured service URL is invalid") from None
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (require_clean_base and parsed.query)
        or len(url) > 8_192
        or not (1 <= port <= 65_535)
    ):
        raise ValueError("configured service URL is invalid")
    if scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback and host not in _allowed_private_hosts():
            raise ValueError(
                "unencrypted service URL is restricted to loopback and "
                "NEXTCLOUD_ALLOWED_PRIVATE_HOSTS"
            )
    return scheme, host, port


class _SameOriginSession(requests.Session):
    """Requests session with a fixed origin, finite timeout, and no redirects."""

    def __init__(self, expected_origin: tuple[str, str, int]) -> None:
        super().__init__()
        self._expected_origin = expected_origin
        self.trust_env = False

    def request(
        self,
        method: str | bytes,
        url: str | bytes,
        params: Any = None,
        data: Any = None,
        headers: Any = None,
        cookies: Any = None,
        files: Any = None,
        auth: Any = None,
        timeout: Any = None,
        allow_redirects: bool = True,
        proxies: Any = None,
        hooks: Any = None,
        stream: Any = None,
        verify: Any = None,
        cert: Any = None,
        json: Any = None,
    ) -> requests.Response:
        del (
            allow_redirects
        )  # accepted for Session compatibility; redirects are always refused
        if _origin(str(url)) != self._expected_origin:
            raise ValueError("remote request crossed the configured service origin")
        kwargs: dict[str, Any] = {
            "params": params,
            "data": data,
            "headers": headers,
            "cookies": cookies,
            "files": files,
            "auth": auth,
            "timeout": timeout if timeout is not None else (10, 30),
            "allow_redirects": False,
            "proxies": proxies,
            "hooks": hooks,
            "stream": stream,
            "verify": verify,
            "cert": cert,
            "json": json,
        }
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise requests.TooManyRedirects("remote redirect was rejected")
        return response


class BaseApiClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        tls_profile: ResolvedTLSProfile | None = None,
    ):
        """Initialize the Nextcloud WebDAV API client."""
        if (
            not isinstance(username, str)
            or not username
            or len(username) > 256
            or any(ord(character) < 32 for character in username)
        ):
            raise ValueError("configured account identifier is invalid")
        if (
            not isinstance(password, str)
            or not password
            or len(password) > 65_536
            or any(character in password for character in "\x00\r\n")
        ):
            raise ValueError("configured credential is invalid")
        configured_origin = _origin(base_url, require_clean_base=True)
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._owns_tls_profile = tls_profile is None
        self.tls_profile = tls_profile or resolve_tls_profile("nextcloud")
        self._session = _SameOriginSession(configured_origin)
        self._session.auth = (username, password)
        self.tls_profile.configure_requests_session(self._session)
        # WebDAV responses can contain attacker-controlled absolute references.
        # Keep ambient proxy variables disabled; an explicitly named TLS profile
        # can still install its governed proxy directly on the session.
        self._session.trust_env = False
        self._session.headers.update({"OCS-APIRequest": "true"})

        self.webdav_base = (
            f"{self.base_url}/remote.php/dav/files/{quote(self.username, safe='')}"
        )
        self.caldav_base = (
            f"{self.base_url}/remote.php/dav/calendars/{quote(self.username, safe='')}"
        )
        self.carddav_base = f"{self.base_url}/remote.php/dav/addressbooks/users/{quote(self.username, safe='')}"
        self.ocs_base = f"{self.base_url}/ocs/v2.php"
        self._files_dav: WebDavBackend | None = None

    @property
    def files_dav(self) -> WebDavBackend:
        """The SDK WebDAV backend on the files root (CONN-NEXTCLOUD-AGENT-R002)."""
        if self._files_dav is None:
            hosts = tuple(sorted(_allowed_private_hosts()))
            self._files_dav = WebDavBackend(
                self.webdav_base,
                username=self.username,
                password=self.password,
                tls=self.tls_profile,
                allowed_private_hosts=hosts,
            )
        return self._files_dav

    def close(self) -> None:
        """Close transport state and erase any profile materialized for this client."""
        if self._files_dav is not None:
            self._files_dav.close()
        self._session.close()
        if self._owns_tls_profile:
            self.tls_profile.cleanup()

    def _get_full_url(self, path: str) -> str:
        """Helper to construct full WebDAV URL for a path."""
        clean_path = path.strip("/")
        if not clean_path:
            return self.webdav_base + "/"
        if (
            len(clean_path) > 4_096
            or "\\" in clean_path
            or any(ord(character) < 32 for character in clean_path)
        ):
            raise ValueError("remote path is invalid")
        segments = clean_path.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise ValueError("remote path is invalid")
        return f"{self.webdav_base}/{'/'.join(quote(segment, safe='') for segment in segments)}"

    @staticmethod
    def _read_xml_response(response: _StreamedResponse) -> bytes:
        """Read one streamed XML response under the parser's byte boundary."""

        try:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared:
                try:
                    if int(declared) > MAX_XML_BYTES:
                        raise XmlSecurityError(
                            "remote XML response exceeds its safe size boundary"
                        )
                except ValueError:
                    raise XmlSecurityError(
                        "remote XML response length is invalid"
                    ) from None
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_XML_BYTES:
                    raise XmlSecurityError(
                        "remote XML response exceeds its safe size boundary"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            response.close()

    @staticmethod
    def _read_json_response(response: requests.Response) -> dict:
        """Read one streamed API response without retaining unbounded content."""

        max_bytes = 8 * 1024 * 1024
        try:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                raw.extend(chunk)
                if len(raw) > max_bytes:
                    raise RuntimeError("remote API response exceeded its safe boundary")
            try:
                value = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError):
                raise RuntimeError("remote API response was invalid") from None
            if not isinstance(value, dict):
                raise RuntimeError("remote API response shape was invalid")
            return value
        finally:
            response.close()

    def _get_absolute_url(self, href: str) -> str:
        """Convert a WebDAV href to a full URL."""
        absolute = urljoin(self.base_url + "/", href)
        if _origin(absolute) != _origin(self.base_url):
            raise ValueError("remote reference crossed the configured service origin")
        return absolute

    def ocs_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make a request to the OCS API."""
        if (
            not isinstance(endpoint, str)
            or len(endpoint) > 2_048
            or ".." in endpoint
            or any(character in endpoint for character in "\\\r\n#")
        ):
            raise ValueError("remote API path is invalid")
        url = f"{self.ocs_base}/{endpoint.lstrip('/')}"
        kwargs.setdefault("headers", {})
        kwargs["headers"].update({"OCS-APIRequest": "true"})
        kwargs.setdefault("params", {})
        kwargs["params"].update({"format": "json"})
        kwargs["stream"] = True

        response = self._session.request(method, url, **kwargs)
        data = self._read_json_response(response)
        meta = data.get("ocs", {}).get("meta", {})
        if meta.get("status") != "ok" and meta.get("statuscode") not in [100, 200]:
            raise RuntimeError("remote API returned an application-level error")

        return data.get("ocs", {}).get("data", {})

    def _app_json(self, method: str, endpoint: str, **kwargs) -> Any:
        """Call one app REST endpoint under ``/index.php/apps`` (Deck, Notes) as JSON."""
        if (
            not isinstance(endpoint, str)
            or len(endpoint) > 2_048
            or ".." in endpoint
            or any(character in endpoint for character in "\\\r\n#")
        ):
            raise ValueError("remote API path is invalid")
        url = f"{self.base_url}/index.php/apps/{endpoint.lstrip('/')}"
        headers = {"OCS-APIRequest": "true", "Accept": "application/json"}
        headers.update(kwargs.pop("headers", {}) or {})
        response = self._session.request(
            method, url, headers=headers, stream=True, **kwargs
        )
        max_bytes = 8 * 1024 * 1024
        try:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                raw.extend(chunk)
                if len(raw) > max_bytes:
                    raise RuntimeError("remote API response exceeded its safe boundary")
        finally:
            response.close()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            raise RuntimeError("remote API response was invalid") from None

    def _dav_request(
        self,
        method: str,
        url: str,
        body: str | bytes | None = None,
        headers: dict | None = None,
    ) -> bytes:
        """Send one DAV request to a URL on the configured origin; return the body."""
        response = self._session.request(
            method,
            self._get_absolute_url(url),
            data=body,
            headers=headers or {},
            stream=True,
            timeout=(10, 60),
        )
        return self._read_xml_response(response)

    def _dav_send(
        self,
        method: str,
        url: str,
        body: str | bytes | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        """Send one DAV write (PUT, DELETE, MOVE, COPY) and raise on an HTTP error."""
        response = self._session.request(
            method,
            self._get_absolute_url(url),
            data=body,
            headers=headers or {},
            timeout=(10, 120),
        )
        response.raise_for_status()
        return response
