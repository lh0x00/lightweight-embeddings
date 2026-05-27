"""Reusable FastAPI dependencies.

* :func:`get_settings` — cached settings instance.
* :func:`get_service` — :class:`EmbeddingsService` from app state.
* :func:`get_analytics`, :func:`get_rate_limiter`, ... — likewise.
* :func:`enforce_request_limits` — combined gate (auth + shedder +
  rate-limit + concurrency + tier-specific input caps). Routes use this in
  place of hand-rolled checks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from ..analytics import AnalyticsService, NullAnalyticsService
from ..core.registry import REGISTRY, ModelKind, ModelSpec
from ..core.service import EmbeddingsService
from ..core.tokens import estimate_tokens_fast
from ..security.auth import compare_token, extract_bearer
from ..security.concurrency import ConcurrencyLimiter
from ..security.cost import estimate_request_cost
from ..security.identity import Identity, extract_identity
from ..security.memguard import MemoryGuard
from ..security.ratelimit import Decision, RateLimiter
from ..security.shedder import AdaptiveShedder
from ..security.tier import Tier, resolve_tier
from ..settings import Settings
from ..settings import get_settings as _get_settings_cached

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return _get_settings_cached()


def get_service(request: Request) -> EmbeddingsService:
    service: EmbeddingsService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "service not ready")
    return service


def get_analytics(request: Request) -> AnalyticsService | NullAnalyticsService:
    return request.app.state.analytics


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_concurrency(request: Request) -> ConcurrencyLimiter:
    return request.app.state.concurrency


def get_shedder(request: Request) -> AdaptiveShedder:
    return request.app.state.shedder


def get_memguard(request: Request) -> MemoryGuard:
    return request.app.state.memguard


# --------------------------------------------------------------------------- #
# Auth                                                                        #
# --------------------------------------------------------------------------- #


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[EmbeddingsService, Depends(get_service)]
AnalyticsDep = Annotated[AnalyticsService | NullAnalyticsService, Depends(get_analytics)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
ConcurrencyDep = Annotated[ConcurrencyLimiter, Depends(get_concurrency)]
ShedderDep = Annotated[AdaptiveShedder, Depends(get_shedder)]
MemGuardDep = Annotated[MemoryGuard, Depends(get_memguard)]


def authenticated_token(
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """Return the validated bearer token, or ``None`` if not set."""
    expected = settings.access_token.get_secret_value() if settings.access_token else None
    if expected is None:
        # Auth disabled — every request is considered "anonymous".
        return None
    provided = extract_bearer(authorization)
    if compare_token(provided, expected):
        return provided
    return None


def require_token(
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    expected = settings.access_token.get_secret_value() if settings.access_token else None
    if expected is None:
        raise HTTPException(503, "authentication is not configured")
    provided = extract_bearer(authorization)
    if not compare_token(provided, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Combined enforcement                                                        #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RequestContext:
    """Bag passed from :func:`enforce_request_limits` to route handlers."""

    identity: Identity
    tier: Tier
    spec: ModelSpec
    estimated_tokens: int
    estimated_cu: float
    decision: Decision

    @property
    def request_id(self) -> str | None:  # convenience
        return None


def _coerce_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    raise HTTPException(422, "input must be a string or list of strings")


def _validate_input_size(items: list[str], tier: Tier, settings: Settings) -> int:
    max_items = min(tier.max_items, settings.max_items_per_request)
    if len(items) > max_items:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"too many items: {len(items)} > {max_items}",
        )
    total_chars = sum(len(s) for s in items)
    max_chars = min(tier.max_chars, settings.max_total_chars)
    if total_chars > max_chars:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"input too large: {total_chars} chars > {max_chars}",
        )
    return total_chars


async def enforce_request_limits(
    *,
    request: Request,
    model_name: str,
    inputs: Any,
    operation: str,
    settings: Settings,
    auth_token: str | None,
    rate_limiter: RateLimiter,
    shedder: AdaptiveShedder,
) -> RequestContext:
    """Apply tier limits and rate-limit, returning a :class:`RequestContext`.

    The actual concurrency acquisition and embedding generation are performed
    by the route using :func:`acquired_slot` and the service.
    """
    identity = extract_identity(
        request,
        authenticated_token=auth_token,
        trusted_proxies=settings.trusted_proxies,
    )
    tier = resolve_tier(identity, settings)

    # Adaptive shedder — fail fast, no useful work yet performed.
    if shedder.should_shed(tier.name):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "service temporarily overloaded",
            headers={"Retry-After": "5"},
        )

    spec = REGISTRY.get(model_name)
    items = _coerce_inputs(inputs)
    if not items:
        raise HTTPException(422, "input must contain at least one item")

    _validate_input_size(items, tier, settings)

    # Image inputs are bounded separately (per-request count + size handled
    # later in the service).
    if spec.kind is ModelKind.IMAGE and len(items) > settings.image_max_per_request:
        raise HTTPException(
            413,
            f"too many images: {len(items)} > {settings.image_max_per_request}",
        )

    # Cost pre-flight (1 "token" per image for the daily token bucket).
    estimated_tokens = (
        estimate_tokens_fast(items) if spec.kind is ModelKind.TEXT else len(items)
    )

    estimated_cu = estimate_request_cost(
        spec=spec,
        operation=operation,  # type: ignore[arg-type]
        n_items=len(items),
        estimated_tokens=estimated_tokens,
    )

    decision = await rate_limiter.consume(
        identity.key,
        tier,
        cost_req=1.0,
        cost_cu=estimated_cu,
        cost_tokens=float(estimated_tokens),
    )
    if not decision.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"rate limit exceeded ({decision.limited_window})",
            headers=decision.headers(),
        )

    return RequestContext(
        identity=identity,
        tier=tier,
        spec=spec,
        estimated_tokens=estimated_tokens,
        estimated_cu=estimated_cu,
        decision=decision,
    )


@asynccontextmanager
async def acquired_slot(
    *,
    concurrency: ConcurrencyLimiter,
    identity_key: str,
    tier: Tier,
    model: str,
):
    """Acquire all concurrency layers for the duration of the block."""
    async with concurrency.acquire(
        identity_key=identity_key,
        identity_capacity=tier.concurrency,
        model=model,
    ):
        yield


async def commit_actual_cost(
    *,
    rate_limiter: RateLimiter,
    ctx: RequestContext,
    actual_tokens: int,
    actual_cu: float,
) -> None:
    """Refund the difference if the actual cost was lower than estimated."""
    refund_cu = max(0.0, ctx.estimated_cu - actual_cu)
    refund_tokens = max(0.0, ctx.estimated_tokens - actual_tokens)
    if refund_cu == 0 and refund_tokens == 0:
        return
    await rate_limiter.refund(ctx.identity.key, cu=refund_cu, tokens=refund_tokens)
