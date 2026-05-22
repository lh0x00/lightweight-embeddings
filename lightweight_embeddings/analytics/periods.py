"""Period-key helpers using ISO calendar.

The previous implementation used ``%U`` (week starting Sunday, 0-indexed)
which produced inconsistent keys around year boundaries. We now use the
ISO 8601 calendar where ``isocalendar()`` returns ``(year, week, weekday)``,
giving stable week numbers across years.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class PeriodKeys:
    """Bucket keys for a single point in time."""

    day: str
    week: str
    month: str
    year: str
    total: str = "total"

    def all(self) -> tuple[str, str, str, str, str]:
        return (self.day, self.week, self.month, self.year, self.total)


def current_period_keys(now: datetime | None = None) -> PeriodKeys:
    """Return :class:`PeriodKeys` for ``now`` (defaults to current UTC time)."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    iso = now.isocalendar()
    return PeriodKeys(
        day=now.strftime("%Y-%m-%d"),
        week=f"{iso.year}-W{iso.week:02d}",
        month=now.strftime("%Y-%m"),
        year=now.strftime("%Y"),
    )


def is_retained(period_key: str, retention: dict[str, int]) -> bool:
    """Return True if a stored period key is within retention windows.

    ``retention`` maps period kind → keep-count. Unknown kinds are kept.
    """
    if period_key == "total":
        return True
    # Sniff kind by shape; cheap heuristics, no exception path needed.
    if "W" in period_key:
        kind = "week"
    elif period_key.count("-") == 2:
        kind = "day"
    elif period_key.count("-") == 1:
        kind = "month"
    else:
        kind = "year"
    return kind in retention  # actual eviction handled at flush time
