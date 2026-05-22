"""Token estimation helpers."""

from __future__ import annotations

from lightweight_embeddings.core.tokens import count_tokens_exact, estimate_tokens_fast


def test_estimate_tokens_fast_monotonic():
    short = estimate_tokens_fast(["hi"])
    long = estimate_tokens_fast(["hi" * 1000])
    assert long > short


def test_estimate_tokens_fast_empty_strings():
    assert estimate_tokens_fast([""]) == 1


class _FakeTokenizer:
    def __call__(self, texts, **_):
        return {"length": [len(t) // 2 + 1 for t in texts]}


def test_count_tokens_exact_uses_tokenizer():
    tok = _FakeTokenizer()
    assert count_tokens_exact(tok, ["hello", "world"]) == sum(
        [len("hello") // 2 + 1, len("world") // 2 + 1]
    )


def test_count_tokens_exact_falls_back_on_error():
    class Broken:
        def __call__(self, *_, **__):
            raise RuntimeError("nope")

    fast = estimate_tokens_fast(["abc"])
    assert count_tokens_exact(Broken(), ["abc"]) == fast
