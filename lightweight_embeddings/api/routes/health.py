"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from .. import deps
from ..schemas import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def liveness():
    return HealthResponse(version=__version__)


@router.get("/readyz", response_model=ReadyResponse, tags=["health"])
async def readiness(
    service: deps.ServiceDep,
    shedder: deps.ShedderDep,
    memguard: deps.MemGuardDep,
):
    loaded = service.loaded_models()
    cpu = shedder.cpu_percent
    rss = memguard.percent()
    if not loaded:
        status = "loading"
    elif memguard.state() == "panic":
        status = "degraded"
    else:
        status = "ok"
    return ReadyResponse(
        status=status,  # type: ignore[arg-type]
        version=__version__,
        models_loaded=loaded,
        device=service.device,
        cpu_percent=cpu,
        memory_percent=rss,
    )
