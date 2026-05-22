"""Adaptive load shedding driven by host metrics.

The shedder samples CPU/memory periodically (in a background task) and
decides whether the current request should be dropped, prioritising
internal/pro tiers. Decisions are cheap reads of cached values — no syscalls
on the hot path.
"""

from __future__ import annotations

import asyncio
import logging

import psutil

from ..settings import Settings
from .tier import TierName

logger = logging.getLogger(__name__)


class AdaptiveShedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cpu = 0.0
        self._rss = 0.0
        self._queue_depth = 0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def cpu_percent(self) -> float:
        return self._cpu

    @property
    def rss_percent(self) -> float:
        return self._rss

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def update_queue_depth(self, value: int) -> None:
        self._queue_depth = max(0, value)

    async def start(self) -> None:
        self._stop.clear()
        # Prime psutil cpu measurement (first call always returns 0).
        psutil.cpu_percent(interval=None)
        self._task = asyncio.create_task(self._loop(), name="shedder-sampler")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._cpu = psutil.cpu_percent(interval=None)
                    self._rss = psutil.virtual_memory().percent
                except Exception:  # pragma: no cover
                    pass
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self._settings.shed_sample_interval_s,
                    )
                    return
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    def should_shed(self, tier: TierName) -> bool:
        s = self._settings
        if self._queue_depth > s.shed_queue_max:
            return True
        if self._rss >= s.shed_rss_pct and tier in (TierName.ANONYMOUS, TierName.FREE):
            return True
        if self._cpu >= s.shed_cpu_global_pct:
            return True
        if tier is TierName.ANONYMOUS and self._cpu >= s.shed_cpu_anon_pct:
            return True
        return tier is TierName.FREE and self._cpu >= s.shed_cpu_free_pct
