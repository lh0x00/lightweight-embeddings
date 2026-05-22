"""Core domain logic: model registry, embedding service, math ops, cache."""

from .registry import (
    REGISTRY,
    ModelKind,
    ModelRegistry,
    ModelSpec,
    detect_model_kind,
)

__all__ = [
    "REGISTRY",
    "ModelKind",
    "ModelRegistry",
    "ModelSpec",
    "detect_model_kind",
]
