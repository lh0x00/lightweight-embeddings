"""Service tiers and per-tier limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..settings import Settings
from .identity import Identity


class TierName(str, Enum):
    ANONYMOUS = "anonymous"
    FREE = "free"
    PRO = "pro"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class Tier:
    """Per-tier limits used by the rate limiter and concurrency guard."""

    name: TierName

    rps: float
    burst: int
    rpm: int
    rph: int
    rpd: int
    daily_cu: float
    daily_tokens: int

    max_items: int
    max_chars: int
    concurrency: int


def _anon_tier(s: Settings) -> Tier:
    return Tier(
        name=TierName.ANONYMOUS,
        rps=s.rl_anon_rps,
        burst=s.rl_anon_burst,
        rpm=s.rl_anon_rpm,
        rph=s.rl_anon_rph,
        rpd=s.rl_anon_rpd,
        daily_cu=s.rl_anon_daily_cu,
        daily_tokens=s.rl_anon_daily_tokens,
        max_items=s.rl_anon_max_items,
        max_chars=s.rl_anon_max_chars,
        concurrency=s.rl_anon_concurrency,
    )


def _free_tier(s: Settings) -> Tier:
    return Tier(
        name=TierName.FREE,
        rps=s.rl_free_rps,
        burst=s.rl_free_burst,
        rpm=s.rl_free_rpm,
        rph=s.rl_free_rph,
        rpd=s.rl_free_rpd,
        daily_cu=s.rl_free_daily_cu,
        daily_tokens=s.rl_free_daily_tokens,
        max_items=s.rl_free_max_items,
        max_chars=s.rl_free_max_chars,
        concurrency=s.rl_free_concurrency,
    )


def _pro_tier(s: Settings) -> Tier:
    return Tier(
        name=TierName.PRO,
        rps=s.rl_pro_rps,
        burst=s.rl_pro_burst,
        rpm=int(s.rl_pro_rps * 60),
        rph=int(s.rl_pro_rps * 3600),
        rpd=int(s.rl_pro_rps * 86400),
        daily_cu=s.rl_pro_daily_cu,
        daily_tokens=int(s.rl_pro_daily_cu * 1000),
        max_items=s.rl_free_max_items,
        max_chars=s.rl_free_max_chars,
        concurrency=s.rl_pro_concurrency,
    )


def resolve_tier(identity: Identity, settings: Settings) -> Tier:
    """Select the tier for an identity.

    For now this is simple: anonymous → ``ANONYMOUS``, authenticated →
    ``FREE``. The ``PRO`` tier is reserved for future scope-bearing tokens
    and can be plugged in by inspecting the identity in the future.
    """
    if not identity.is_authenticated:
        return _anon_tier(settings)
    return _free_tier(settings)


def all_tiers(settings: Settings) -> dict[TierName, Tier]:
    return {
        TierName.ANONYMOUS: _anon_tier(settings),
        TierName.FREE: _free_tier(settings),
        TierName.PRO: _pro_tier(settings),
    }
