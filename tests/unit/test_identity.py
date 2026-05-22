"""Identity extraction."""

from __future__ import annotations

from types import SimpleNamespace

from lightweight_embeddings.security.identity import (
    Identity,
    IdentitySource,
    extract_identity,
)


def _fake_request(host: str, headers: dict[str, str] | None = None):
    headers = headers or {}
    # Lowercase keys to match Starlette's CIMultiDict behaviour.
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers={k.lower(): v for k, v in headers.items()},
    )


def test_anonymous_identity_uses_ip():
    req = _fake_request("1.2.3.4")
    ident = extract_identity(req, authenticated_token=None, trusted_proxies=[])
    assert isinstance(ident, Identity)
    assert ident.source is IdentitySource.IP
    assert ident.is_authenticated is False
    assert ident.key.endswith("1.2.3.4")


def test_authenticated_identity_hashes_token():
    req = _fake_request("1.2.3.4")
    ident = extract_identity(req, authenticated_token="secret", trusted_proxies=[])
    assert ident.source is IdentitySource.TOKEN
    assert ident.is_authenticated is True
    assert "secret" not in ident.key
    assert ident.key.startswith("tok_")
