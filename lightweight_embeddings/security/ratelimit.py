"""Multi-window token-bucket rate limiter (in-memory).

For each ``(identity, tier)`` we track several token buckets:

* ``req_second`` — RPS smoothing with burst.
* ``req_minute`` — short-term abuse cap.
* ``req_hour``   — medium horizon.
* ``req_day``    — daily request quota.
* ``cu_day``     — daily compute-unit quota.
* ``tok_day``    — daily token quota.

A request is allowed only if **all** windows have enough tokens.

The implementation uses an asyncio lock per identity, which is enough for a
single-process FastAPI worker. Multi-process deployments should switch to
the Redis backend (interface left in place for future addition).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from cachetools import TTLCache

from .tier import Tier, TierName

logger = logging.getLogger(__name__)


class LimiterError(Exception):
    """Base error for rate-limit operations."""


@dataclass(slots=True)
class RateLimitWindow:
    """A single token-bucket window."""

    name: str           # for headers & logs
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float
    last: float

    def refill(self, now: float) -> None:
        if now <= self.last:
            return
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_rate)
        self.last = now

    def try_consume(self, cost: float, now: float) -> tuple[bool, float]:
        self.refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        deficit = cost - self.tokens
        retry_after = deficit / self.refill_rate if self.refill_rate > 0 else float("inf")
        return False, retry_after

    def refund(self, amount: float, now: float) -> None:
        if amount <= 0:
            return
        self.refill(now)
        self.tokens = min(self.capacity, self.tokens + amount)


def _build_windows(tier: Tier) -> dict[str, RateLimitWindow]:
    now = time.monotonic()
    return {
        "req_second": RateLimitWindow(
            name="req-second",
            capacity=max(1.0, float(tier.burst)),
            refill_rate=max(0.001, tier.rps),
            tokens=float(tier.burst),
            last=now,
        ),
        "req_minute": RateLimitWindow(
            name="req-minute",
            capacity=float(tier.rpm),
            refill_rate=tier.rpm / 60.0,
            tokens=float(tier.rpm),
            last=now,
        ),
        "req_hour": RateLimitWindow(
            name="req-hour",
            capacity=float(tier.rph),
            refill_rate=tier.rph / 3600.0,
            tokens=float(tier.rph),
            last=now,
        ),
        "req_day": RateLimitWindow(
            name="req-day",
            capacity=float(tier.rpd),
            refill_rate=tier.rpd / 86400.0,
            tokens=float(tier.rpd),
            last=now,
        ),
        "cu_day": RateLimitWindow(
            name="cu-day",
            capacity=tier.daily_cu,
            refill_rate=tier.daily_cu / 86400.0,
            tokens=tier.daily_cu,
            last=now,
        ),
        "tok_day": RateLimitWindow(
            name="tok-day",
            capacity=float(tier.daily_tokens),
            refill_rate=tier.daily_tokens / 86400.0,
            tokens=float(tier.daily_tokens),
            last=now,
        ),
    }


@dataclass(slots=True)
class TierLimits:
    """Bucket bundle for a single identity at a given tier."""

    tier: Tier
    windows: dict[str, RateLimitWindow]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class Decision:
    """Outcome of a rate-limit check."""

    allowed: bool
    retry_after: float
    limited_window: str | None
    snapshot: dict[str, tuple[float, float]]
    tier: TierName

    def headers(self) -> dict[str, str]:
        """Return RFC-style ``RateLimit-*`` and informational headers."""
        out: dict[str, str] = {"X-RateLimit-Tier": self.tier.value}
        # Pick the most-restrictive window for the standard headers.
        if "req_minute" in self.snapshot:
            cap, rem = self.snapshot["req_minute"]
            out["X-RateLimit-Limit-Minute"] = str(int(cap))
            out["X-RateLimit-Remaining-Minute"] = str(max(0, int(rem)))
        if "req_day" in self.snapshot:
            cap, rem = self.snapshot["req_day"]
            out["X-RateLimit-Limit-Day"] = str(int(cap))
            out["X-RateLimit-Remaining-Day"] = str(max(0, int(rem)))
        if "cu_day" in self.snapshot:
            cap, rem = self.snapshot["cu_day"]
            out["X-RateLimit-Limit-CU-Day"] = str(int(cap))
            out["X-RateLimit-Remaining-CU-Day"] = str(max(0, int(rem)))
        if not self.allowed:
            out["Retry-After"] = str(max(1, int(self.retry_after) + 1))
        return out


class RateLimiter:
    """In-memory token-bucket limiter."""

    def __init__(
        self,
        *,
        identity_capacity: int = 100_000,
        identity_ttl_s: int = 86_400,
    ) -> None:
        # TTLCache evicts cold identities so we don't accumulate state forever.
        self._buckets: TTLCache[str, TierLimits] = TTLCache(
            maxsize=identity_capacity, ttl=identity_ttl_s
        )
        self._cache_lock = asyncio.Lock()

    async def _get_or_create(self, identity_key: str, tier: Tier) -> TierLimits:
        async with self._cache_lock:
            existing = self._buckets.get(identity_key)
            if existing is not None and existing.tier.name is tier.name:
                # Refresh TTL by re-set.
                self._buckets[identity_key] = existing
                return existing
            fresh = TierLimits(tier=tier, windows=_build_windows(tier))
            self._buckets[identity_key] = fresh
            return fresh

    async def consume(
        self,
        identity_key: str,
        tier: Tier,
        *,
        cost_req: float = 1.0,
        cost_cu: float = 0.0,
        cost_tokens: float = 0.0,
    ) -> Decision:
        bucket = await self._get_or_create(identity_key, tier)
        async with bucket.lock:
            now = time.monotonic()
            costs: list[tuple[str, float]] = [
                ("req_second", cost_req),
                ("req_minute", cost_req),
                ("req_hour", cost_req),
                ("req_day", cost_req),
                ("cu_day", cost_cu),
                ("tok_day", cost_tokens),
            ]
            # Refresh all first to compute snapshots.
            for w in bucket.windows.values():
                w.refill(now)

            # Pre-flight check (no mutation): all windows must allow.
            limited: str | None = None
            retry_after = 0.0
            for key, cost in costs:
                if cost <= 0:
                    continue
                w = bucket.windows[key]
                if w.tokens < cost:
                    deficit = cost - w.tokens
                    candidate = deficit / w.refill_rate if w.refill_rate > 0 else float("inf")
                    if limited is None or candidate > retry_after:
                        limited = key
                        retry_after = candidate
            allowed = limited is None
            if allowed:
                for key, cost in costs:
                    if cost > 0:
                        bucket.windows[key].tokens -= cost
            snapshot = {
                k: (w.capacity, w.tokens) for k, w in bucket.windows.items()
            }
            return Decision(
                allowed=allowed,
                retry_after=retry_after,
                limited_window=limited,
                snapshot=snapshot,
                tier=tier.name,
            )

    async def refund(
        self,
        identity_key: str,
        *,
        cu: float = 0.0,
        tokens: float = 0.0,
    ) -> None:
        bucket = self._buckets.get(identity_key)
        if bucket is None:
            return
        async with bucket.lock:
            now = time.monotonic()
            if cu > 0:
                bucket.windows["cu_day"].refund(cu, now)
            if tokens > 0:
                bucket.windows["tok_day"].refund(tokens, now)


def build_rate_limiter(backend: str = "memory") -> RateLimiter:
    """Factory; only ``memory`` is implemented at the moment."""
    if backend != "memory":
        logger.warning(
            "rate-limit backend %r is not implemented; falling back to memory", backend
        )
    return RateLimiter()
