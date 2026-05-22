"""Concurrency caps: global, per-identity and per-model."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from cachetools import TTLCache


class ConcurrencyError(Exception):
    """Raised when a concurrency cap is exceeded."""

    def __init__(self, scope: str) -> None:
        super().__init__(f"too many concurrent requests ({scope})")
        self.scope = scope


class _NonBlockingSemaphore:
    """Semaphore wrapper that raises immediately when full."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._sem = asyncio.Semaphore(self._capacity)

    @asynccontextmanager
    async def acquire_or_fail(self, scope: str):
        if self._sem.locked():
            raise ConcurrencyError(scope)
        # ``Semaphore.acquire()`` does not have a non-blocking flavour
        # before Python 3.13; use the internal counter check.
        if self._sem._value <= 0:  # type: ignore[attr-defined]
            raise ConcurrencyError(scope)
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


class ConcurrencyLimiter:
    """Track three layers of concurrency: global, per-identity, per-model.

    Per-identity buckets are stored in a TTLCache so cold identities are
    evicted automatically.
    """

    def __init__(
        self,
        *,
        global_capacity: int,
        per_model_capacity: int,
        identity_ttl_s: int = 600,
        identity_capacity: int = 100_000,
    ) -> None:
        self._global = _NonBlockingSemaphore(global_capacity)
        self._per_model_capacity = per_model_capacity
        self._models: dict[str, _NonBlockingSemaphore] = {}
        self._identities: TTLCache[str, _NonBlockingSemaphore] = TTLCache(
            maxsize=identity_capacity, ttl=identity_ttl_s
        )

    def _model_sem(self, name: str) -> _NonBlockingSemaphore:
        sem = self._models.get(name)
        if sem is None:
            sem = _NonBlockingSemaphore(self._per_model_capacity)
            self._models[name] = sem
        return sem

    def _identity_sem(self, identity_key: str, capacity: int) -> _NonBlockingSemaphore:
        sem = self._identities.get(identity_key)
        if sem is None or getattr(sem, "_capacity", capacity) != capacity:
            sem = _NonBlockingSemaphore(capacity)
            self._identities[identity_key] = sem
        return sem

    @asynccontextmanager
    async def acquire(self, *, identity_key: str, identity_capacity: int, model: str):
        identity_sem = self._identity_sem(identity_key, identity_capacity)
        model_sem = self._model_sem(model)
        async with (
            self._global.acquire_or_fail("global"),
            identity_sem.acquire_or_fail("identity"),
            model_sem.acquire_or_fail("model"),
        ):
            yield
