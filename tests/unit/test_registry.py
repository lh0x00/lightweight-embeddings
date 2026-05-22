"""Model registry behaviour."""

from __future__ import annotations

import pytest

from lightweight_embeddings.core.registry import (
    REGISTRY,
    ModelKind,
    UnknownModelError,
    detect_model_kind,
)


def test_known_text_model():
    spec = REGISTRY.get("multilingual-e5-small")
    assert spec.kind is ModelKind.TEXT


def test_known_image_model():
    spec = REGISTRY.get("siglip-base-patch16-256-multilingual")
    assert spec.kind is ModelKind.IMAGE


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        REGISTRY.get("nonexistent-model")


def test_detect_model_kind():
    assert detect_model_kind("bge-m3") is ModelKind.TEXT
    assert detect_model_kind("siglip-base-patch16-256-multilingual") is ModelKind.IMAGE


def test_truncate_dim_matryoshka_ok():
    spec = REGISTRY.get("embeddinggemma-300m")
    assert spec.truncate_dim(256) == 256
    assert spec.truncate_dim(spec.embedding_dim) is None  # native dim → no truncate
    assert spec.truncate_dim(None) is None


def test_truncate_dim_matryoshka_invalid():
    spec = REGISTRY.get("embeddinggemma-300m")
    with pytest.raises(ValueError):
        spec.truncate_dim(123)


def test_truncate_dim_unsupported():
    spec = REGISTRY.get("multilingual-e5-small")
    with pytest.raises(ValueError):
        spec.truncate_dim(128)
