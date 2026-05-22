"""GET /v1/stats — auth-gated, in-process."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...analytics.periods import current_period_keys
from .. import deps
from ..schemas import StatsBucket, StatsResponse

router = APIRouter()


@router.get("/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats(
    analytics: deps.AnalyticsDep,
    _: Annotated[str, Depends(deps.require_token)],
):
    keys = current_period_keys()
    raw = await analytics.stats()

    def bucket(kind: str) -> StatsBucket:
        kind_data = raw.get(kind, {})
        return StatsBucket(
            total=kind_data.get(keys.total, {}),
            daily=kind_data.get(keys.day, {}),
            weekly=kind_data.get(keys.week, {}),
            monthly=kind_data.get(keys.month, {}),
            yearly=kind_data.get(keys.year, {}),
        )

    return StatsResponse(access=bucket("access"), tokens=bucket("tokens"))
