"""ISO calendar periodicity."""

from __future__ import annotations

from datetime import datetime, timezone

from lightweight_embeddings.analytics.periods import current_period_keys


def test_current_period_keys_format():
    pk = current_period_keys(datetime(2025, 3, 14, 12, 0, tzinfo=timezone.utc))
    assert pk.day == "2025-03-14"
    assert pk.month == "2025-03"
    assert pk.year == "2025"
    assert pk.week.startswith("2025-W")
    assert len(pk.week) == 8  # YYYY-WNN


def test_iso_week_year_boundary():
    # 2024-12-31 is ISO week 1 of 2025.
    pk = current_period_keys(datetime(2024, 12, 31, 0, 0, tzinfo=timezone.utc))
    assert pk.week == "2025-W01"


def test_naive_datetime_is_treated_as_utc():
    pk = current_period_keys(datetime(2025, 1, 1, 0, 0))
    assert pk.year == "2025"
