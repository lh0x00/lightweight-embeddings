"""Token-bucket rate limiter behaviour."""

from __future__ import annotations

import pytest

from lightweight_embeddings.security.ratelimit import RateLimiter
from lightweight_embeddings.security.tier import Tier, TierName


def _tier(*, rpm: int = 60, burst: int = 200, rps: float = 200.0) -> Tier:
    return Tier(
        name=TierName.FREE,
        rps=rps,
        burst=burst,
        rpm=rpm,
        rph=rpm * 10,
        rpd=rpm * 100,
        daily_cu=100.0,
        daily_tokens=10_000,
        max_items=128,
        max_chars=500_000,
        concurrency=8,
    )


@pytest.mark.asyncio
async def test_consume_within_budget_allowed():
    rl = RateLimiter()
    tier = _tier()
    decision = await rl.consume("k1", tier, cost_req=1.0, cost_cu=1.0, cost_tokens=10)
    assert decision.allowed is True
    assert decision.snapshot["req_minute"][1] < tier.rpm


@pytest.mark.asyncio
async def test_consume_blocks_when_minute_exhausted():
    rl = RateLimiter()
    tier = _tier()
    # Spend the minute window quickly with cheap CU/tok.
    for _ in range(tier.rpm):
        d = await rl.consume("k2", tier, cost_req=1.0, cost_cu=0.0, cost_tokens=0)
        assert d.allowed
    blocked = await rl.consume("k2", tier, cost_req=1.0, cost_cu=0.0, cost_tokens=0)
    assert blocked.allowed is False
    assert blocked.limited_window is not None
    assert "Retry-After" in blocked.headers()


@pytest.mark.asyncio
async def test_refund_returns_budget():
    rl = RateLimiter()
    tier = _tier()
    await rl.consume("k3", tier, cost_req=1.0, cost_cu=10.0, cost_tokens=100)
    before = (await rl.consume("k3", tier, cost_req=0.0, cost_cu=0.0, cost_tokens=0)).snapshot[
        "cu_day"
    ][1]
    await rl.refund("k3", cu=5.0, tokens=50)
    after = (await rl.consume("k3", tier, cost_req=0.0, cost_cu=0.0, cost_tokens=0)).snapshot[
        "cu_day"
    ][1]
    assert after >= before
