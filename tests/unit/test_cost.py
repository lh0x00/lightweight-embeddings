"""Compute-unit cost estimation correctness."""

from __future__ import annotations

from lightweight_embeddings.core.registry import REGISTRY, ModelKind
from lightweight_embeddings.security.cost import estimate_request_cost


def test_text_cost_scales_with_tokens():
    spec = REGISTRY.get("multilingual-e5-small")
    cu_low = estimate_request_cost(
        spec=spec, operation="embeddings", n_items=1, estimated_tokens=100
    )
    cu_high = estimate_request_cost(
        spec=spec, operation="embeddings", n_items=1, estimated_tokens=10_000
    )
    assert cu_high > cu_low


def test_text_cost_scales_with_model_weight():
    light = REGISTRY.get("multilingual-e5-small")
    heavy = REGISTRY.get("bge-m3")
    cu_light = estimate_request_cost(
        spec=light, operation="embeddings", n_items=1, estimated_tokens=1000
    )
    cu_heavy = estimate_request_cost(
        spec=heavy, operation="embeddings", n_items=1, estimated_tokens=1000
    )
    assert cu_heavy > cu_light


def test_image_cost_per_item():
    spec = REGISTRY.get("siglip-base-patch16-256-multilingual")
    assert spec.kind is ModelKind.IMAGE
    cu_one = estimate_request_cost(
        spec=spec, operation="embeddings", n_items=1, estimated_tokens=0
    )
    cu_many = estimate_request_cost(
        spec=spec, operation="embeddings", n_items=4, estimated_tokens=0
    )
    assert cu_many == 4 * cu_one
