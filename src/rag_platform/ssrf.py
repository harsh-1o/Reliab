"""SSRF Protection Layer for Outbound HTTP Adapters.

Validates URLs and resolved IP addresses against loopback, private (RFC1918),
link-local, cloud-metadata, multicast, broadcast, and non-whitelisted destinations.
Protects against DNS rebinding and redirect-based SSRF bypasses.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any, Callable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Disallowed IPv4 networks
DISALLOWED_IPV4_NETWORKS = [
    ipaddress.IPv4Network("0.0.0.0/8"),          # Current network (only valid as source)
    ipaddress.IPv4Network("10.0.0.0/8"),         # RFC1918 Private
    ipaddress.IPv4Network("100.64.0.0/10"),      # Shared Address Space (Carrier-Grade NAT)
    ipaddress.IPv4Network("127.0.0.0/8"),        # Loopback
    ipaddress.IPv4Network("169.254.0.0/16"),     # Link-Local / Cloud Metadata (169.254.169.254)
    ipaddress.IPv4Network("172.16.0.0/12"),      # RFC1918 Private
    ipaddress.IPv4Network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.IPv4Network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.IPv4Network("192.168.0.0/16"),     # RFC1918 Private
    ipaddress.IPv4Network("198.18.0.0/15"),      # Benchmarking
    ipaddress.IPv4Network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.IPv4Network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.IPv4Network("224.0.0.0/4"),        # Multicast
    ipaddress.IPv4Network("240.0.0.0/4"),        # Reserved for Future Use
    ipaddress.IPv4Network("255.255.255.255/32"), # Broadcast
]

# Disallowed IPv6 networks
DISALLOWED_IPV6_NETWORKS = [
    ipaddress.IPv6Network("::1/128"),            # Loopback
    ipaddress.IPv6Network("::/128"),             # Unspecified
    ipaddress.IPv6Network("::ffff:0:0/96"),      # IPv4-mapped (checked via mapped IPv4)
    ipaddress.IPv6Network("64:ff9b::/96"),       # IPv4/IPv6 translation
    ipaddress.IPv6Network("100::/64"),           # Discard-Only
    ipaddress.IPv6Network("2001::/23"),          # IETF Protocol Assignments
    ipaddress.IPv6Network("2001:db8::/32"),      # Documentation
    ipaddress.IPv6Network("fc00::/7"),           # Unique Local (ULA - Private)
    ipaddress.IPv6Network("fe80::/10"),          # Link-Local Unicast
    ipaddress.IPv6Network("ff00::/8"),           # Multicast
]

DISALLOWED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "metadata.google.internal",
    "instance-data",
}


class SSRFProtectionError(ValueError):
    """Raised when an outbound URL violates SSRF security policies."""
    pass


def get_allowed_hosts() -> set[str] | None:
    """Read allowed outbound hosts from HTTP_RAG_ALLOWED_HOSTS env var if configured."""
    raw = os.environ.get("HTTP_RAG_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    hosts = set()
    for h in raw.split(","):
        cleaned = h.strip().lower()
        if cleaned:
            # Strip port if user configured host:port
            if ":" in cleaned:
                cleaned = cleaned.split(":", 1)[0]
            hosts.add(cleaned)
    return hosts if hosts else None


# Static DNS mapping for air-gapped deployments, offline test suites, and mock resolution
STATIC_DNS_MAP: dict[str, list[str]] = {
    # Default public test fixtures (IANA / RFC reserved public documentation IPs)
    "example.com": ["93.184.216.34"],
    "rag.example.com": ["93.184.216.34"],
    "staging-rag.example.com": ["93.184.216.35"],
    "httpbin.org": ["54.237.133.81"],
}


def register_static_dns(hostname: str, ips: list[str] | str) -> None:
    """Register a static DNS entry for air-gapped environments or offline testing.

    Ensures that offline test runners and air-gapped deployments can resolve public
    service hostnames without requiring external DNS socket calls.
    All resolved IPs remain strictly subject to is_ip_allowed() security checks.
    """
    clean_host = hostname.strip().lower()
    ip_list = [ips] if isinstance(ips, str) else list(ips)
    STATIC_DNS_MAP[clean_host] = ip_list


def clear_static_dns(hostname: str | None = None) -> None:
    """Clear static DNS entries."""
    if hostname:
        STATIC_DNS_MAP.pop(hostname.strip().lower(), None)
    else:
        STATIC_DNS_MAP.clear()


def get_static_dns_map_from_env() -> dict[str, list[str]]:
    """Parse static DNS mapping from SSRF_STATIC_DNS_MAP environment variable.

    Supports JSON formatting: '{"api.internal": ["93.184.216.34"]}'
    or comma-separated key=value: 'api.internal=93.184.216.34,api2.internal=93.184.216.35'
    """
    raw = os.environ.get("SSRF_STATIC_DNS_MAP", "").strip()
    if not raw:
        return {}
    try:
        import json
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                k.strip().lower(): [v] if isinstance(v, str) else list(v)
                for k, v in data.items()
            }
    except Exception:
        pass

    result: dict[str, list[str]] = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip().lower()] = [v.strip()]
        elif ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip().lower()] = [v.strip()]
    return result


def is_ip_allowed(ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if an IP address is a safe, routable public address."""
    if isinstance(ip, str):
        try:
            ip = ipaddress.ip_address(ip)
        except ValueError:
            return False

    # Check for IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_ip_allowed(ip.ipv4_mapped)

    # General python ipaddress classification
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    ):
        return False

    # Specific network block verification
    if isinstance(ip, ipaddress.IPv4Address):
        for net in DISALLOWED_IPV4_NETWORKS:
            if ip in net:
                return False
    elif isinstance(ip, ipaddress.IPv6Address):
        for net in DISALLOWED_IPV6_NETWORKS:
            if ip in net:
                return False

    return True


def validate_url_ssrf(
    url: str,
    allowed_hosts: set[str] | list[str] | None = None,
    allow_private_ips: bool = False,
    dns_resolver: Callable[[str, int], list[str]] | None = None,
) -> list[str]:
    """Validate a URL against SSRF attack vectors immediately before outbound request.

    Performs:
    1. URL parsing and scheme restriction (http/https only).
    2. Host validation and allowed_hosts enforcement.
    3. DNS resolution of hostname (via custom resolver, air-gapped static DNS map, or DNS).
    4. IP validation rejecting private, loopback, link-local, and reserved ranges.

    Returns:
        List of resolved, validated IP address strings.

    Raises:
        SSRFProtectionError: If any security check fails.
    """
    if not url or not isinstance(url, str):
        raise SSRFProtectionError("URL must be a non-empty string")

    try:
        parsed = urlsplit(url.strip())
    except Exception as exc:
        raise SSRFProtectionError(f"Malformed URL: {exc}") from exc

    # 1. Scheme check
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SSRFProtectionError(
            f"Unsupported URL scheme '{scheme}'. Only 'http' and 'https' are allowed."
        )

    # 2. Hostname extraction
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise SSRFProtectionError("URL has no hostname")

    # Reject known loopback/metadata hostnames immediately unless explicitly bypassed for test harnesses
    if not allow_private_ips:
        if hostname in DISALLOWED_HOSTNAMES or hostname.endswith(".localhost"):
            raise SSRFProtectionError(f"Access to host '{hostname}' is blocked by SSRF policy")

    # Check configured allowlist
    configured_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()
    if configured_allowlist is not None:
        allowed_set = {h.lower() for h in configured_allowlist}
        if hostname not in allowed_set:
            raise SSRFProtectionError(
                f"Host '{hostname}' is not in the allowed hosts list: {sorted(allowed_set)}"
            )

    # 3. DNS Resolution & IP validation
    port = parsed.port or (443 if scheme == "https" else 80)

    # Direct IP literal check
    try:
        literal_ip = ipaddress.ip_address(hostname)
        if not allow_private_ips and not is_ip_allowed(literal_ip):
            raise SSRFProtectionError(
                f"IP address '{hostname}' is in a private/reserved range and is blocked"
            )
        return [str(literal_ip)]
    except ValueError:
        pass  # Hostname is a domain name, proceed to DNS resolution

    raw_ips: list[str] = []
    if dns_resolver is not None:
        try:
            raw_ips = dns_resolver(hostname, port)
        except Exception as exc:
            raise SSRFProtectionError(f"Custom DNS resolver failed for '{hostname}': {exc}") from exc
    elif hostname in STATIC_DNS_MAP:
        raw_ips = STATIC_DNS_MAP[hostname]
    else:
        env_map = get_static_dns_map_from_env()
        if hostname in env_map:
            raw_ips = env_map[hostname]
        else:
            try:
                # Resolve all addresses (protects against multi-A DNS rebinding / mixed public-private records)
                addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
                raw_ips = [sockaddr[0] for _, _, _, _, sockaddr in addr_info]
            except socket.gaierror as exc:
                raise SSRFProtectionError(f"DNS resolution failed for host '{hostname}': {exc}") from exc

    if not raw_ips:
        raise SSRFProtectionError(f"No DNS records found for host '{hostname}'")

    resolved_ips: list[str] = []
    for ip_str in raw_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SSRFProtectionError(f"Unparseable resolved IP '{ip_str}' for host '{hostname}'")

        if not allow_private_ips and not is_ip_allowed(ip_obj):
            raise SSRFProtectionError(
                f"Host '{hostname}' resolved to disallowed private/reserved IP: {ip_str}"
            )
        resolved_ips.append(ip_str)

    return resolved_ips


# --- DNS Rebinding Protection via Transport-Level IP Pinning ---
import httpx
from httpcore._backends.auto import AutoBackend


class SSRFPinningNetworkBackend(AutoBackend):
    """Network backend that connects directly to pre-validated IP addresses, preventing DNS rebinding.

    Guarantees that the socket connects to an already-validated IP address and never
    performs a second unbound DNS lookup that could resolve to a private/loopback/metadata IP.
    Preserves TLS SNI and HTTP Host headers for correct virtual hosting and SSL certificate validation.
    """

    def __init__(
        self,
        ip_pins: dict[str, str] | None = None,
        allowed_hosts: set[str] | list[str] | None = None,
        allow_private_ips: bool = False,
        dns_resolver: Callable[[str, int], list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.ip_pins: dict[str, str] = {k.lower(): v for k, v in (ip_pins or {}).items()}
        self.allowed_hosts = allowed_hosts
        self.allow_private_ips = allow_private_ips
        self.dns_resolver = dns_resolver

    def pin_host(self, host: str, ip: str) -> None:
        self.ip_pins[host.strip().lower()] = ip

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        clean_host = host.strip().lower()
        target_ip = self.ip_pins.get(clean_host)

        if target_ip is None:
            # Validate URL/host and resolve IP immediately at socket creation time
            url_to_validate = f"http://{clean_host}:{port}"
            valid_ips = validate_url_ssrf(
                url_to_validate,
                allowed_hosts=self.allowed_hosts,
                allow_private_ips=self.allow_private_ips,
                dns_resolver=self.dns_resolver,
            )
            target_ip = valid_ips[0]
            self.ip_pins[clean_host] = target_ip
        else:
            # Enforce that the pinned IP itself is still strictly safe
            if not self.allow_private_ips:
                try:
                    ip_obj = ipaddress.ip_address(target_ip)
                    if not is_ip_allowed(ip_obj):
                        raise SSRFProtectionError(
                            f"Pinned IP '{target_ip}' for host '{host}' is disallowed private/reserved IP"
                        )
                except ValueError as exc:
                    raise SSRFProtectionError(f"Invalid pinned IP '{target_ip}': {exc}") from exc

        await self._init_backend()
        # Connect TCP socket directly to target_ip, while httpcore/httpx retains server_hostname for TLS SNI
        return await self._backend.connect_tcp(
            target_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class SSRFProtectedTransport(httpx.AsyncHTTPTransport):
    """Custom httpx AsyncHTTPTransport configured with SSRFPinningNetworkBackend.

    Guarantees that socket-level connections use pinned, pre-validated IP addresses.
    """

    def __init__(
        self,
        ip_pins: dict[str, str] | None = None,
        allowed_hosts: set[str] | list[str] | None = None,
        allow_private_ips: bool = False,
        dns_resolver: Callable[[str, int], list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pinning_backend = SSRFPinningNetworkBackend(
            ip_pins=ip_pins,
            allowed_hosts=allowed_hosts,
            allow_private_ips=allow_private_ips,
            dns_resolver=dns_resolver,
        )
        self._pool._network_backend = self.pinning_backend

    def pin_host(self, host: str, ip: str) -> None:
        self.pinning_backend.pin_host(host, ip)

