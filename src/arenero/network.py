import ipaddress
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import SplitResult, unquote, urljoin, urlsplit, urlunsplit

import urllib3
from urllib3.util import Timeout

from ._deadline import remaining as _shell_remaining

_CONTROLLED_HEADERS = {
    "connection",
    "content-length",
    "host",
    "proxy-authorization",
    "transfer-encoding",
}


class NetworkError(ValueError):
    """Report a rejected or failed network request."""


@dataclass(frozen=True)
class NetworkConfig:
    """Set the public network policy for the virtual shell.

    Attributes:
        allowed_url_prefixes: Exact URL origins and path prefixes that can be used.
            An empty tuple allows all public HTTP and HTTPS URLs.
        enabled: Whether the virtual curl command is available.
        allowed_methods: Uppercase HTTP methods that can be used.
        timeout_seconds: Maximum connect and between-read idle time. Elapsed
            redirect time reduces the next request timeout.
        max_response_bytes: Maximum bytes accepted in the final response body.
        max_request_bytes: Maximum bytes accepted in a request body.
        max_redirects: Maximum followed redirects.
        deny_non_global_addresses: Reject private, local, and reserved IP addresses.
        trusted_header_provider: Optional host callback that returns headers for a
            checked URL. These headers override headers supplied by curl.
    """

    allowed_url_prefixes: tuple[str, ...] = ()
    enabled: bool = True
    allowed_methods: frozenset[str] = frozenset({"GET", "HEAD"})
    timeout_seconds: float = 10.0
    max_response_bytes: int = 1_000_000
    max_request_bytes: int = 100_000
    max_redirects: int = 5
    deny_non_global_addresses: bool = True
    trusted_header_provider: Callable[[str], Mapping[str, str]] | None = None

    def __post_init__(self) -> None:
        """Validate the network policy values."""
        for prefix in self.allowed_url_prefixes:
            parsed = _parse_url(prefix, "allowed URL prefix")
            if parsed.query:
                raise ValueError("allowed URL prefixes must not include a query")
        if not self.allowed_methods or any(
            not method.isalpha() or method != method.upper()
            for method in self.allowed_methods
        ):
            raise ValueError("allowed_methods must contain uppercase HTTP methods")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if self.max_response_bytes <= 0 or self.max_request_bytes < 0:
            raise ValueError("network byte limits must be valid positive limits")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be at least 0")
        if self.trusted_header_provider is not None and not callable(
            self.trusted_header_provider
        ):
            raise ValueError("trusted_header_provider must be callable")


@dataclass(frozen=True)
class NetworkRequest:
    """Describe one bounded HTTP request.

    Attributes:
        url: Absolute HTTP or HTTPS URL.
        method: Uppercase HTTP method.
        headers: Request header name and value pairs.
        body: Optional request body bytes.
    """

    url: str
    method: str = "GET"
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes | None = None


@dataclass(frozen=True)
class NetworkResponse:
    """Describe one completed HTTP response.

    Attributes:
        url: Final URL after redirects.
        status: HTTP status code.
        reason: HTTP reason text.
        headers: Response header name and value pairs.
        body: Response body bytes.
    """

    url: str
    status: int
    reason: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


def fetch_url(
    request: NetworkRequest,
    config: NetworkConfig,
    *,
    follow_redirects: bool = False,
    timeout_seconds: float | None = None,
) -> NetworkResponse:
    """Run one policy-checked HTTP request with pinned DNS resolution.

    Args:
        request: URL, method, headers, and optional body.
        config: Host network policy and hard limits.
        follow_redirects: Follow redirects when true.
        timeout_seconds: Optional lower connection and read timeout requested by
            the caller.

    Returns:
        Final response with a bounded body.

    Raises:
        NetworkError: If policy rejects the request or the request fails.
    """
    method = request.method.upper()
    body = request.body
    headers = dict(request.headers)
    timeout = config.timeout_seconds
    if timeout_seconds is not None:
        if timeout_seconds <= 0:
            raise NetworkError("curl timeout must be greater than 0")
        timeout = min(timeout, timeout_seconds)
    deadline = time.monotonic() + timeout
    url = request.url

    for redirect_count in range(config.max_redirects + 1):
        parsed = _require_allowed_url(url, config)
        request_headers = _trusted_headers(url, headers, config)
        _validate_request(method, request_headers, body, config)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NetworkError("network request timed out")
        response = _fetch_once(
            url,
            method,
            request_headers,
            body,
            parsed,
            config,
            remaining,
        )
        location = _header(response.headers, "location")
        is_redirect = response.status in {301, 302, 303, 307, 308} and location
        if not follow_redirects or not is_redirect:
            return response
        if redirect_count == config.max_redirects:
            raise NetworkError("network redirect limit exceeded")

        next_url = urljoin(url, location)
        current_origin = _origin(parsed)
        next_parsed = _require_allowed_url(next_url, config)
        if _origin(next_parsed) != current_origin:
            headers = {
                name: value
                for name, value in headers.items()
                if name.lower() not in {"authorization", "cookie"}
            }
        if response.status == 303 or (
            response.status in {301, 302} and method not in {"GET", "HEAD"}
        ):
            method = "GET"
            body = None
        url = next_url

    raise NetworkError("network redirect limit exceeded")


def _trusted_headers(
    url: str,
    request_headers: dict[str, str],
    config: NetworkConfig,
) -> dict[str, str]:
    """Add trusted host headers after the URL policy check.

    Args:
        url: Checked request URL.
        request_headers: Headers supplied by the curl caller.
        config: Active network policy.

    Returns:
        Combined headers. Trusted values replace caller values without regard to
        header-name case.

    Raises:
        NetworkError: If the trusted provider fails or returns invalid data.
    """
    if config.trusted_header_provider is None:
        return request_headers
    try:
        trusted = dict(config.trusted_header_provider(url))
    except Exception as exc:
        raise NetworkError(f"trusted header provider failed: {exc}") from exc
    combined = dict(request_headers)
    caller_names = {name.lower(): name for name in combined}
    for name, value in trusted.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise NetworkError("trusted headers must contain string names and values")
        previous = caller_names.get(name.lower())
        if previous is not None:
            del combined[previous]
        combined[name] = value
    return combined


def _validate_request(
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    config: NetworkConfig,
) -> None:
    """Validate method, headers, and request body against the policy.

    Args:
        method: Uppercase HTTP method.
        headers: Request header mapping.
        body: Optional request body.
        config: Active network policy.

    Raises:
        NetworkError: If a request value is not safe or allowed.
    """
    if method not in config.allowed_methods:
        raise NetworkError(f"HTTP method is not allowed: {method}")
    if body is not None and len(body) > config.max_request_bytes:
        raise NetworkError("request body exceeds the network limit")
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not name
            or any(character in name for character in "\r\n:")
        ):
            raise NetworkError("invalid HTTP header name")
        if name.lower() in _CONTROLLED_HEADERS:
            raise NetworkError(f"HTTP header is controlled by Arenero: {name}")
        if "\r" in value or "\n" in value:
            raise NetworkError("invalid HTTP header value")


def _fetch_once(
    url: str,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    parsed: SplitResult,
    config: NetworkConfig,
    timeout_seconds: float,
) -> NetworkResponse:
    """Fetch one URL from one DNS-pinned address without redirects.

    Args:
        url: Allowed absolute URL.
        method: Allowed uppercase HTTP method.
        headers: Validated request headers.
        body: Optional bounded request body.
        parsed: Parsed URL from ``urlsplit``.
        config: Active network policy.
        timeout_seconds: Remaining request time.

    Returns:
        One response with a bounded body.

    Raises:
        NetworkError: If DNS, transport, TLS, or response limits fail.
    """
    host = parsed.hostname
    if host is None:
        raise NetworkError("URL must include a host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolve_addresses(host, port, config.deny_non_global_addresses)
    shell_remaining = _shell_remaining()
    if shell_remaining is not None:
        timeout_seconds = min(timeout_seconds, shell_remaining)
    address = addresses[0]
    target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    request_headers = dict(headers)
    request_headers["Host"] = _host_header(host, port, parsed.scheme)
    request_headers.setdefault("User-Agent", "arenero/0.1")
    timeout = Timeout(total=timeout_seconds)

    if parsed.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            address,
            port=port,
            timeout=timeout,
            retries=False,
            cert_reqs="CERT_REQUIRED",
            assert_hostname=host,
            server_hostname=host,
        )
    else:
        pool = urllib3.HTTPConnectionPool(
            address,
            port=port,
            timeout=timeout,
            retries=False,
        )

    response = None
    try:
        response = pool.urlopen(
            method,
            target,
            body=body,
            headers=request_headers,
            redirect=False,
            retries=False,
            preload_content=False,
            decode_content=False,
        )
        declared_length = response.headers.get("content-length")
        if (
            declared_length is not None
            and int(declared_length) > config.max_response_bytes
        ):
            raise NetworkError("response body exceeds the network limit")
        response_body = response.read(config.max_response_bytes + 1)
        _shell_remaining()
        if len(response_body) > config.max_response_bytes:
            raise NetworkError("response body exceeds the network limit")
        return NetworkResponse(
            url=url,
            status=response.status,
            reason=response.reason or "",
            headers=tuple(response.headers.items()),
            body=response_body,
        )
    except NetworkError:
        raise
    except (OSError, ValueError, urllib3.exceptions.HTTPError) as exc:
        raise NetworkError(f"network request failed: {exc}") from exc
    finally:
        if response is not None:
            response.release_conn()
        pool.close()


def _require_allowed_url(url: str, config: NetworkConfig) -> SplitResult:
    """Parse a URL and require an allowed exact origin and path prefix.

    Args:
        url: URL to validate.
        config: Active network policy.

    Returns:
        Parsed URL value.

    Raises:
        NetworkError: If the URL is outside the allowlist.
    """
    parsed = _parse_url(url, "URL")
    if not config.allowed_url_prefixes:
        return parsed
    request_path = _safe_path(parsed.path)
    for prefix in config.allowed_url_prefixes:
        allowed = _parse_url(prefix, "allowed URL prefix")
        allowed_path = _safe_path(allowed.path)
        if _origin(parsed) == _origin(allowed) and _path_has_prefix(
            request_path, allowed_path
        ):
            return parsed
    raise NetworkError("URL is not in allowed_url_prefixes")


def _parse_url(url: str, label: str) -> SplitResult:
    """Parse and validate the common URL structure.

    Args:
        url: Absolute URL or allowed prefix.
        label: Name used in error messages.

    Returns:
        Parsed URL value.

    Raises:
        ValueError: If the URL structure is invalid.
    """
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"{label} must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not include user information")
    if parsed.fragment:
        raise ValueError(f"{label} must not include a fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{label} has an invalid port")
    _safe_path(parsed.path)
    return parsed


def _safe_path(path: str) -> str:
    """Return a decoded path after rejecting ambiguous path forms.

    Args:
        path: URL path text.

    Returns:
        Decoded absolute path.

    Raises:
        ValueError: If the path can have an ambiguous server meaning.
    """
    lowered = path.lower()
    if "\\" in path or "%2f" in lowered or "%5c" in lowered:
        raise ValueError("URL path contains an ambiguous separator")
    decoded = unquote(path or "/")
    if "//" in decoded or any(ord(character) < 32 for character in decoded):
        raise ValueError("URL path contains an ambiguous or control character")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError("URL path must not contain dot segments")
    return decoded if decoded.startswith("/") else f"/{decoded}"


def _path_has_prefix(path: str, prefix: str) -> bool:
    """Test a segment-aware URL path prefix.

    Args:
        path: Request path.
        prefix: Allowed path prefix.

    Returns:
        True when the path is equal to or below the prefix.
    """
    if prefix == "/":
        return True
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(f"{prefix}/")


def _origin(parsed: SplitResult) -> tuple[str, str, int]:
    """Return a normalized scheme, host, and effective port.

    Args:
        parsed: Parsed URL value.

    Returns:
        Normalized URL origin tuple.
    """
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = parsed.hostname
    if host is None:
        raise NetworkError("URL hostname is required")
    return parsed.scheme, host.lower(), port


def _resolve_addresses(host: str, port: int, deny_non_global: bool) -> list[str]:
    """Resolve a host and reject unsafe addresses when requested.

    Args:
        host: URL hostname.
        port: Effective URL port.
        deny_non_global: Reject all non-global IP addresses when true.

    Returns:
        Unique resolved IP address strings.

    Raises:
        NetworkError: If resolution fails or returns a rejected address.
    """
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise NetworkError(f"DNS resolution failed: {exc}") from exc
    addresses = list(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses:
        raise NetworkError("DNS resolution returned no addresses")
    if deny_non_global:
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
            if not ip.is_global:
                raise NetworkError(f"network address is not public: {address}")
    return addresses


def _host_header(host: str, port: int, scheme: str) -> str:
    """Build an HTTP Host header with a non-default port when needed.

    Args:
        host: URL hostname.
        port: Effective URL port.
        scheme: HTTP or HTTPS scheme.

    Returns:
        Host header value.
    """
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    return rendered_host if port == default_port else f"{rendered_host}:{port}"


def _header(headers: tuple[tuple[str, str], ...], name: str) -> str | None:
    """Find one case-insensitive response header.

    Args:
        headers: Response header pairs.
        name: Header name to find.

    Returns:
        Header value, or ``None`` when absent.
    """
    lowered = name.lower()
    return next((value for key, value in headers if key.lower() == lowered), None)
