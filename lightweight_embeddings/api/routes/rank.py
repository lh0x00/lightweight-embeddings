"""POST /v1/rank."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ...core.registry import ModelKind
from ...security.cost import estimate_request_cost
from .. import deps
from ..schemas import RankRequest, RankResponse, Usage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rank", response_model=RankResponse, tags=["rank"])
async def rank_candidates(
    body: RankRequest,
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
    # Combine queries+candidates for the cost gate.
    combined_inputs: list[str] = []
    if isinstance(body.queries, list):
        combined_inputs.extend(body.queries)
    else:
        combined_inputs.append(body.queries)
    combined_inputs.extend(body.candidates)

    ctx = await deps.enforce_request_limits(
        request=request,
        model_name=body.model,
        inputs=combined_inputs,
        operation="rank",
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
        result = await service.rank(body.model, body.queries, body.candidates)

    actual_tokens = int(result["usage"]["total_tokens"])
    actual_cu = estimate_request_cost(
        spec=ctx.spec,
        operation="rank",
        n_items=len(combined_inputs),
        estimated_tokens=actual_tokens
        if ctx.spec.kind is ModelKind.TEXT
        else len(combined_inputs),
    )
    await deps.commit_actual_cost(
        rate_limiter=rate_limiter,
        ctx=ctx,
        actual_tokens=actual_tokens,
        actual_cu=actual_cu,
    )

    response = RankResponse(
        probabilities=result["probabilities"],
        cosine_similarities=result["cosine_similarities"],
        usage=Usage(prompt_tokens=actual_tokens, total_tokens=actual_tokens),
    )
    background_tasks.add_task(_record_analytics, analytics, ctx.spec.name, actual_tokens)
    request.state.rate_limit_headers = ctx.decision.headers()
    return response


async def _record_analytics(analytics, model_id: str, tokens: int) -> None:
    try:
        await analytics.access(model_id, tokens)
    except Exception:  # pragma: no cover
        logger.exception("analytics.access failed")
