"""Caller identity extraction.

For unauthenticated callers we use the client IP, honoring trusted-proxy
headers (``X-Forwarded-For`` / ``X-Real-IP``). For authenticated callers we
use a SHA-256 of the token so the raw secret never appears in logs or in
the rate-limit backend keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from fastapi import Request


class IdentitySource(str, Enum):
    IP = "ip"
    TOKEN = "token"  # noqa: S105 — enum value, not a credential


@dataclass(frozen=True, slots=True)
class Identity:
    """A stable identifier used for rate limiting and metrics."""

    key: str
    source: IdentitySource
    raw_ip: str | None = None
    is_authenticated: bool = False

    def short(self) -> str:
        """Return a short, log-safe representation."""
        if self.source is IdentitySource.TOKEN:
            return f"token:{self.key[:8]}"
        return f"ip:{self.key}"


def _client_ip_from_request(request: Request, trusted_proxies: list[str]) -> str:
    """Resolve the client IP, optionally honoring forwarded headers."""
    direct = request.client.host if request.client else "0.0.0.0"  # noqa: S104
    if not trusted_proxies and direct not in trusted_proxies:
        return direct
    headers = request.headers
    fwd = headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip() or direct
    real = headers.get("x-real-ip")
    if real:
        return real.strip() or direct
    return direct


def extract_identity(
    request: Request,
    *,
    authenticated_token: str | None,
    trusted_proxies: list[str],
) -> Identity:
    """Build an :class:`Identity` for the current request."""
    if authenticated_token:
        digest = hashlib.sha256(authenticated_token.encode("utf-8")).hexdigest()
        return Identity(
            key=f"tok_{digest}",
            source=IdentitySource.TOKEN,
            is_authenticated=True,
        )
    ip = _client_ip_from_request(request, trusted_proxies)
    return Identity(key=f"ip_{ip}", source=IdentitySource.IP, raw_ip=ip)
