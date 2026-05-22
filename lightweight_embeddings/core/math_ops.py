"""Numerically stable similarity and softmax helpers."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``matrix`` with each row L2-normalized.

    Rows whose norm is below ``_EPS`` are returned unchanged (avoids
    NaN/exploding values for degenerate zero vectors).
    """
    norm = np.linalg.norm(matrix, axis=axis, keepdims=True)
    safe = np.where(norm < _EPS, 1.0, norm)
    return matrix / safe


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` and rows of ``b``.

    Inputs may be unnormalized. Output shape is ``(len(a), len(b))``.
    """
    a_n = normalize(a, axis=1)
    b_n = normalize(b, axis=1)
    return a_n @ b_n.T


def cosine_similarity_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same as :func:`cosine_similarity` but assumes inputs are already
    L2-normalized — saves two normalizations on the hot path."""
    return a @ b.T


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)
