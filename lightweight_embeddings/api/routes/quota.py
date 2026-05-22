"""GET /v1/quota — show the caller's current quota state."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ...security.identity import extract_identity
from ...security.tier import resolve_tier
from .. import deps
from ..schemas import QuotaResponse

router = APIRouter()


@router.get("/quota", response_model=QuotaResponse, tags=["quota"])
async def get_quota(
    request: Request,
    settings: deps.SettingsDep,
    rate_limiter: deps.RateLimiterDep,
    auth_token: Annotated[str | None, Depends(deps.authenticated_token)],
):
    identity = extract_identity(
        request,
        authenticated_token=auth_token,
        trusted_proxies=settings.trusted_proxies,
    )
    tier = resolve_tier(identity, settings)

    # No-op consume to surface current state without spending budget.
    decision = await rate_limiter.consume(
        identity.key, tier, cost_req=0.0, cost_cu=0.0, cost_tokens=0.0
    )
    snapshot = decision.snapshot
    req_day = snapshot.get("req_day", (tier.rpd, tier.rpd))
    req_min = snapshot.get("req_minute", (tier.rpm, tier.rpm))
    cu_day = snapshot.get("cu_day", (tier.daily_cu, tier.daily_cu))

    return QuotaResponse(
        tier=tier.name.value,
        daily_request_limit=int(req_day[0]),
        daily_request_remaining=max(0, int(req_day[1])),
        daily_cu_limit=float(cu_day[0]),
        daily_cu_remaining=max(0.0, float(cu_day[1])),
        minute_request_limit=int(req_min[0]),
        minute_request_remaining=max(0, int(req_min[1])),
        concurrency_limit=tier.concurrency,
    )
