"""Compute Unit (CU) cost estimation for quota accounting.

A request's cost is::

    CU = model_weight × tokens_or_pixels × op_weight

where:

* ``model_weight`` is fetched from the model registry's ``cost_weight``.
* For text models, the unit is *kilo-tokens* (so an input of 1000 tokens
  on ``e5-small`` costs ``1.0 × 1.0 = 1.0`` CU).
* For image models, the unit is the number of images (1 image ≈ 1 unit).
* ``op_weight`` is ``1.0`` for ``embeddings`` and ``1.0`` for ``rank`` —
  ranking already scales with the total token count of queries+candidates.
"""

from __future__ import annotations

from typing import Literal

from ..core.registry import REGISTRY, ModelKind, ModelSpec

OperationKind = Literal["embeddings", "rank"]

# Convenience export for code that wants the static weight without the spec.
MODEL_COST_WEIGHT: dict[str, float] = {spec.name: spec.cost_weight for spec in REGISTRY}


def estimate_request_cost(
    *,
    spec: ModelSpec,
    operation: OperationKind,
    n_items: int,
    estimated_tokens: int,
) -> float:
    """Return the CU cost for a request."""
    if spec.kind is ModelKind.IMAGE:
        return spec.cost_weight * max(1, n_items)
    # text: 1k token = 1 unit. Both operations cost the same per token; the
    # ranking endpoint already inflates ``estimated_tokens`` because it sums
    # queries+candidates.
    units = max(1.0, estimated_tokens / 1000.0)
    return spec.cost_weight * units
