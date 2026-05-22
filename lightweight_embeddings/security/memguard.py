"""Last-resort memory guard.

Two thresholds:

* **High water mark** — start dropping cold cache entries.
* **Panic water mark** — actively reject requests with 503.
"""

from __future__ import annotations

import gc
import logging

import psutil

from ..settings import Settings

logger = logging.getLogger(__name__)


class MemoryGuard:
    def __init__(self, settings: Settings, *, on_pressure=None, on_panic=None) -> None:
        self._settings = settings
        self._on_pressure = on_pressure
        self._on_panic = on_panic

    @staticmethod
    def percent() -> float:
        try:
            return psutil.virtual_memory().percent
        except Exception:  # pragma: no cover
            return 0.0

    def state(self) -> str:
        pct = self.percent()
        if pct >= self._settings.mem_panic_watermark_pct:
            return "panic"
        if pct >= self._settings.mem_high_watermark_pct:
            return "pressure"
        return "ok"

    def maybe_relieve(self) -> None:
        """Trigger callbacks based on current memory state."""
        state = self.state()
        if state == "panic":
            logger.warning("memory PANIC at %.1f%%", self.percent())
            if self._on_panic is not None:
                try:
                    self._on_panic()
                except Exception:  # pragma: no cover
                    logger.exception("on_panic callback failed")
            gc.collect()
        elif state == "pressure":
            logger.info("memory pressure at %.1f%%", self.percent())
            if self._on_pressure is not None:
                try:
                    self._on_pressure()
                except Exception:  # pragma: no cover
                    logger.exception("on_pressure callback failed")
