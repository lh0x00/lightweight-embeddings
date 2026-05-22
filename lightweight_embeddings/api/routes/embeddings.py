"""POST /v1/embeddings."""

from __future__ import annotations

import base64
import logging
from typing import Annotated

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ...core.registry import ModelKind
from ...security.cost import estimate_request_cost
from .. import deps
from ..schemas import EmbeddingDataItem, EmbeddingRequest, EmbeddingResponse, Usage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(
    body: EmbeddingRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: deps.SettingsDep,
    service: deps.ServiceDep,
    analytics: deps.AnalyticsDep,
    rate_limiter: deps.RateLimiterDep,
    concurrency: deps.ConcurrencyDep,
    shedder: deps.ShedderDep,
    auth_token: Annotated[str | None, Depends(deps.authenticated_token)],
):
    ctx = await deps.enforce_request_limits(
        request=request,
        model_name=body.model,
        inputs=body.input,
        operation="embeddings",
        settings=settings,
        auth_token=auth_token,
        rate_limiter=rate_limiter,
        shedder=shedder,
    )

    async with deps.acquired_slot(
        concurrency=concurrency,
        identity_key=ctx.identity.key,
        tier=ctx.tier,
        model=ctx.spec.name,
    ):
        embeddings, spec = await service.generate_embeddings(
            body.model,
            body.input,
            normalize=True,
            dimensions=body.dimensions,
        )

    if spec.kind is ModelKind.TEXT:
        actual_tokens = service.count_tokens(spec.name, body.input)
    else:
        actual_tokens = len(embeddings)

    actual_cu = estimate_request_cost(
        spec=spec,
        operation="embeddings",
        n_items=len(embeddings),
        estimated_tokens=actual_tokens,
    )
    await deps.commit_actual_cost(
        rate_limiter=rate_limiter,
        ctx=ctx,
        actual_tokens=actual_tokens,
        actual_cu=actual_cu,
    )

    data = [
        EmbeddingDataItem(index=i, embedding=_serialize_vector(vec, body.encoding_format))
        for i, vec in enumerate(embeddings)
    ]
    response = EmbeddingResponse(
        data=data,
        model=spec.name,
        usage=Usage(prompt_tokens=actual_tokens, total_tokens=actual_tokens),
    )

    background_tasks.add_task(_record_analytics, analytics, spec.name, actual_tokens)
    request.state.rate_limit_headers = ctx.decision.headers()
    return response


def _serialize_vector(vec: np.ndarray, fmt: str) -> list[float] | str:
    arr = vec.astype(np.float32, copy=False)
    if fmt == "base64":
        return base64.b64encode(arr.tobytes()).decode("ascii")
    return arr.tolist()


async def _record_analytics(analytics, model_id: str, tokens: int) -> None:
    try:
        await analytics.access(model_id, tokens)
    except Exception:  # pragma: no cover
        logger.exception("analytics.access failed")
