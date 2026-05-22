"""Bearer parsing + constant-time comparison."""

from __future__ import annotations

from lightweight_embeddings.security.auth import compare_token, extract_bearer


def test_extract_bearer_variants():
    assert extract_bearer("Bearer abc") == "abc"
    assert extract_bearer("bearer abc") == "abc"
    assert extract_bearer("BEARER abc") == "abc"
    assert extract_bearer("abc") == "abc"
    assert extract_bearer("  Bearer   xyz  ") == "xyz"


def test_extract_bearer_empty():
    assert extract_bearer(None) is None
    assert extract_bearer("") is None
    assert extract_bearer("   ") is None
    assert extract_bearer("Bearer ") is None


def test_compare_token_constant_time():
    assert compare_token("secret", "secret") is True
    assert compare_token("secret", "wrong") is False
    assert compare_token(None, "secret") is False
    assert compare_token("secret", None) is False
    assert compare_token("", "") is False
