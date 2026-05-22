"""Authentication helpers using constant-time comparison."""

from __future__ import annotations

import hmac

_BEARER_PREFIX = "bearer "


def extract_bearer(authorization: str | None) -> str | None:
    """Return the raw token from an Authorization header.

    Accepts both ``Bearer <token>`` and the bare token form. Whitespace is
    stripped. ``None``/empty/``"Bearer "``-only values yield ``None``.
    """
    if not authorization:
        return None
    value = authorization.strip()
    if not value:
        return None
    lower = value.lower()
    if lower.startswith(_BEARER_PREFIX):
        rest = value[len(_BEARER_PREFIX):].strip()
        return rest or None
    if lower == _BEARER_PREFIX.strip():
        return None
    return value


def compare_token(provided: str | None, expected: str | None) -> bool:
    """Compare a provided token against the expected secret.

    Always uses :func:`hmac.compare_digest` to avoid timing side-channels.
    Returns ``False`` whenever either value is missing.
    """
    if not provided or not expected:
        return False
    try:
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
    except (AttributeError, TypeError):
        return False
