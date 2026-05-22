"""GET /v1/models — list registered models (OpenAI-compatible)."""

from __future__ import annotations

from fastapi import APIRouter

from ...core.registry import REGISTRY
from .. import deps
from ..schemas import ModelInfo, ModelsListResponse

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse, tags=["models"])
async def list_models(service: deps.ServiceDep):
    loaded = set(service.loaded_models())
    items = [
        ModelInfo(
            id=spec.name,
            kind=spec.kind.value,
            family=spec.family,
            embedding_dim=spec.embedding_dim,
            max_seq_length=spec.max_seq_length,
            matryoshka_dims=list(spec.matryoshka_dims),
            cost_weight=spec.cost_weight,
            loaded=spec.name in loaded,
        )
        for spec in REGISTRY
    ]
    return ModelsListResponse(data=items)
