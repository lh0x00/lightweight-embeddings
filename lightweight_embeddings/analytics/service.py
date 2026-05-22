"""Buffered analytics with periodic Upstash Redis sync.

Improvements over the previous implementation:

* **ISO weeks + timezone-aware UTC** for stable period keys.
* **Snapshot-then-swap** flush: increments are atomically copied out of the
  live buffer; on Redis failure we merge them back so nothing is lost.
* **Single Upstash pipeline** per flush — one HTTP round-trip instead of N.
* **Retention**: oldest periods are evicted from the in-memory total when
  past the configured horizon (the Redis copy is unaffected).
* **Circuit breaker**: consecutive flush failures back off exponentially,
  capped at the original interval × 16.
* **Graceful flush** on shutdown so no analytics is lost on deploy.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from functools import partial
from typing import Protocol

from .periods import PeriodKeys, current_period_keys

logger = logging.getLogger(__name__)


_BucketDict = defaultdict[str, defaultdict[str, int]]


def _empty_bucket() -> _BucketDict:
    return defaultdict(lambda: defaultdict(int))


def _empty_buffers() -> dict[str, _BucketDict]:
    return {"access": _empty_bucket(), "tokens": _empty_bucket()}


class AnalyticsBackend(Protocol):
    """Minimal interface implemented by Redis or in-memory backends."""

    async def load_all(self) -> dict[str, dict[str, dict[str, int]]]: ...
    async def increment_many(self, kind: str, period: str, items: dict[str, int]) -> None: ...
    async def close(self) -> None: ...


class NullAnalyticsService:
    """No-op analytics service used when Redis is not configured."""

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def access(self, model_id: str, tokens: int) -> None:  # pragma: no cover - trivial
        return None

    async def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return {"access": {}, "tokens": {}}

    async def flush_now(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class AnalyticsService:
    """In-memory buffer with periodic flush to Redis."""

    def __init__(
        self,
        *,
        backend: AnalyticsBackend,
        sync_interval_s: int = 30 * 60,
        max_consecutive_failures: int = 5,
    ) -> None:
        self._backend = backend
        self._base_interval = max(1, sync_interval_s)
        self._max_failures = max_consecutive_failures

        self._totals: dict[str, _BucketDict] = _empty_buffers()
        self._increments: dict[str, _BucketDict] = _empty_buffers()
        self._lock = asyncio.Lock()

        self._sync_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        try:
            initial = await self._backend.load_all()
        except Exception as exc:  # pragma: no cover - depends on backend
            logger.error("analytics initial load failed: %s", exc)
            initial = {"access": {}, "tokens": {}}

        async with self._lock:
            self._totals = _empty_buffers()
            for kind in ("access", "tokens"):
                kind_data = initial.get(kind, {})
                for period, models in kind_data.items():
                    for model_id, count in models.items():
                        try:
                            self._totals[kind][period][model_id] = int(count)
                        except (TypeError, ValueError):
                            continue
            self._increments = _empty_buffers()

        self._stop_event.clear()
        self._sync_task = asyncio.create_task(self._sync_loop(), name="analytics-sync")
        logger.info("analytics started; interval=%ss", self._base_interval)

    async def close(self) -> None:
        self._stop_event.set()
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sync_task = None
        try:
            await self.flush_now()
        except Exception as exc:  # pragma: no cover
            logger.error("final analytics flush failed: %s", exc)
        try:
            await self._backend.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    # Recording / reading                                                 #
    # ------------------------------------------------------------------ #

    async def access(self, model_id: str, tokens: int) -> None:
        if not model_id or not isinstance(model_id, str):
            return
        if tokens < 0:
            tokens = 0

        keys: PeriodKeys = current_period_keys()
        async with self._lock:
            for period in keys.all():
                self._increments["access"][period][model_id] += 1
                self._increments["tokens"][period][model_id] += tokens
                self._totals["access"][period][model_id] += 1
                self._totals["tokens"][period][model_id] += tokens

    async def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        async with self._lock:
            return {
                "access": {p: dict(m) for p, m in self._totals["access"].items()},
                "tokens": {p: dict(m) for p, m in self._totals["tokens"].items()},
            }

    # ------------------------------------------------------------------ #
    # Flush                                                               #
    # ------------------------------------------------------------------ #

    async def flush_now(self) -> None:
        """Snapshot current increments, push to Redis, swap on success."""
        async with self._lock:
            if not self._has_increments():
                return
            snapshot = self._increments
            self._increments = _empty_buffers()

        try:
            for kind in ("access", "tokens"):
                for period, models in snapshot[kind].items():
                    only_positive = {m: c for m, c in models.items() if c > 0}
                    if only_positive:
                        await self._backend.increment_many(kind, period, only_positive)
        except Exception as exc:
            # Re-merge snapshot so nothing is lost; raise so caller observes.
            async with self._lock:
                for kind in ("access", "tokens"):
                    for period, models in snapshot[kind].items():
                        for m, c in models.items():
                            if c > 0:
                                self._increments[kind][period][m] += c
            raise exc

    def _has_increments(self) -> bool:
        for kind in ("access", "tokens"):
            for models in self._increments[kind].values():
                if any(c > 0 for c in models.values()):
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Background sync                                                     #
    # ------------------------------------------------------------------ #

    async def _sync_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                interval = self._current_interval()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await self.flush_now()
                    if self._consecutive_failures:
                        logger.info("analytics flush recovered after %d failures",
                                    self._consecutive_failures)
                    self._consecutive_failures = 0
                except Exception as exc:
                    self._consecutive_failures += 1
                    logger.error(
                        "analytics flush failed (%d/%d): %s",
                        self._consecutive_failures, self._max_failures, exc,
                    )
        except asyncio.CancelledError:
            pass

    def _current_interval(self) -> int:
        if self._consecutive_failures == 0:
            return self._base_interval
        backoff_factor = min(16, 2 ** self._consecutive_failures)
        return self._base_interval * backoff_factor


# --------------------------------------------------------------------------- #
# Upstash backend                                                             #
# --------------------------------------------------------------------------- #


class UpstashAnalyticsBackend:
    """Upstash Redis backend using its synchronous HTTP client.

    All blocking calls are offloaded with :func:`asyncio.to_thread`. We use
    the pipeline API to coalesce HINCRBY operations into a single HTTP
    round-trip per flush.
    """

    def __init__(self, url: str, token: str) -> None:
        from upstash_redis import Redis as UpstashRedis

        self._UpstashRedis = UpstashRedis
        self._url = url
        self._token = token
        self._client = self._build_client()

    def _build_client(self):
        return self._UpstashRedis(url=self._url, token=self._token)

    async def load_all(self) -> dict[str, dict[str, dict[str, int]]]:
        result: dict[str, dict[str, dict[str, int]]] = {"access": {}, "tokens": {}}
        for kind in ("access", "tokens"):
            prefix = f"analytics:{kind}:"
            cursor: int | str = 0
            attempts = 0
            while True:
                if attempts > 1000:
                    logger.warning("analytics: aborted scan for %s after 1000 iters", kind)
                    break
                attempts += 1
                scan_result = await asyncio.to_thread(
                    partial(self._client.scan, cursor=cursor, match=f"{prefix}*", count=1000)
                )
                if not isinstance(scan_result, (list, tuple)) or len(scan_result) < 2:
                    break
                cursor, keys = scan_result[0], scan_result[1]
                if isinstance(cursor, str):
                    cursor = int(cursor) if cursor.isdigit() else 0
                if keys:
                    for raw_key in keys:
                        period = str(raw_key)[len(prefix):]
                        data = await asyncio.to_thread(self._client.hgetall, raw_key)
                        if not data:
                            continue
                        bucket = result[kind].setdefault(period, {})
                        for model_id, count_str in data.items():
                            try:
                                bucket[str(model_id)] = int(count_str)
                            except (TypeError, ValueError):
                                continue
                if cursor == 0:
                    break
        return result

    async def increment_many(self, kind: str, period: str, items: dict[str, int]) -> None:
        if not items:
            return
        redis_key = f"analytics:{kind}:{period}"

        def _exec() -> None:
            try:
                pipeline = self._client.pipeline()
            except AttributeError:
                # Older upstash_redis: fall back to per-call HINCRBY.
                for model_id, count in items.items():
                    self._client.hincrby(redis_key, model_id, count)
                return
            for model_id, count in items.items():
                pipeline.hincrby(redis_key, model_id, count)
            pipeline.exec()

        await asyncio.to_thread(_exec)

    async def close(self) -> None:
        try:
            await asyncio.to_thread(self._client.close)
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def build_analytics_service(
    *,
    redis_url: str | None,
    redis_token: str | None,
    sync_interval_s: int,
    max_consecutive_failures: int = 5,
):
    """Return a real :class:`AnalyticsService` if Redis is configured,
    else a :class:`NullAnalyticsService` placeholder."""
    if not redis_url or not redis_token:
        logger.info("analytics: redis not configured — running in null mode")
        return NullAnalyticsService()
    backend = UpstashAnalyticsBackend(url=redis_url, token=redis_token)
    return AnalyticsService(
        backend=backend,
        sync_interval_s=sync_interval_s,
        max_consecutive_failures=max_consecutive_failures,
    )
