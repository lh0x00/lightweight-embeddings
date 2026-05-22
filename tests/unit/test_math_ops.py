"""Numerical stability of softmax / cosine."""

from __future__ import annotations

import numpy as np

from lightweight_embeddings.core import math_ops


def test_softmax_sums_to_one():
    a = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, -10.0]])
    out = math_ops.softmax(a)
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_softmax_handles_large_values():
    a = np.array([1e6, 1e6 + 1.0])
    out = math_ops.softmax(a)
    assert np.isfinite(out).all()


def test_cosine_similarity_identity():
    a = np.eye(3, dtype=np.float32)
    sim = math_ops.cosine_similarity(a, a)
    assert np.allclose(sim, np.eye(3), atol=1e-6)


def test_cosine_similarity_zero_vectors_safe():
    a = np.zeros((1, 4), dtype=np.float32)
    b = np.zeros((1, 4), dtype=np.float32)
    sim = math_ops.cosine_similarity(a, b)
    # Should not be NaN — clamp to zero norms keeps result finite.
    assert np.isfinite(sim).all()


def test_normalize_unit_norm():
    a = np.random.RandomState(0).randn(5, 7).astype(np.float32)
    out = math_ops.normalize(a, axis=1)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
