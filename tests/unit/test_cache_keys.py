"""Cache key correctness: deterministic, model-namespaced, low collision."""

from __future__ import annotations

from lightweight_embeddings.core.cache import EmbeddingCache, make_cache_key


def test_make_cache_key_is_deterministic():
    a = make_cache_key("model-x", "hello")
    b = make_cache_key("model-x", "hello")
    assert a == b
    assert len(a) == 32  # 128-bit hex


def test_make_cache_key_separates_models():
    assert make_cache_key("a", "x") != make_cache_key("b", "x")


def test_make_cache_key_separates_texts():
    assert make_cache_key("m", "foo") != make_cache_key("m", "bar")


def test_make_cache_key_is_namespace_safe():
    # Ensure model+text concatenation cannot collide via clever input.
    a = make_cache_key("ab", "c")
    b = make_cache_key("a", "bc")
    assert a != b


def test_embedding_cache_basic():
    import numpy as np

    cache = EmbeddingCache(maxsize=4)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("k1", v)
    out = cache.get("k1")
    assert out is not None
    assert out.shape == (3,)
    assert out.dtype == np.float32

    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["hits"] == 1


def test_embedding_cache_disabled_when_zero():
    cache = EmbeddingCache(maxsize=0)
    assert not cache.enabled
    cache.set("k", b"\x00")  # type: ignore[arg-type]
    assert cache.get("k") is None
