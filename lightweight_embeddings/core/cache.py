"""Thread-safe LRU cache for per-item text embeddings.

Keys are 128-bit BLAKE2b digests of ``"<model>\\x00<text>"`` to avoid
collisions even at very large cache sizes (the previous 32-bit MD5 prefix
caused frequent collisions and silently returned wrong vectors).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable

import numpy as np
from cachetools import LRUCache

_KEY_DIGEST_BYTES = 16


def make_cache_key(model: str, text: str) -> str:
    """Return a stable 128-bit hex key for ``(model, text)``."""
    digest = hashlib.blake2b(
        f"{model}\x00{text}".encode(),
        digest_size=_KEY_DIGEST_BYTES,
    )
    return digest.hexdigest()


class EmbeddingCache:
    """Thread-safe wrapper around ``cachetools.LRUCache``.

    Stores 1-D ``np.ndarray`` embeddings (already normalized when applicable).
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            self._cache: LRUCache | None = None
        else:
            self._cache = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    def get(self, key: str) -> np.ndarray | None:
        if self._cache is None:
            return None
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            # return a view; consumers should not mutate
            return value

    def get_many(self, keys: Iterable[str]) -> list[np.ndarray | None]:
        if self._cache is None:
            return [None for _ in keys]
        with self._lock:
            results: list[np.ndarray | None] = []
            for k in keys:
                v = self._cache.get(k)
                if v is None:
                    self._misses += 1
                else:
                    self._hits += 1
                results.append(v)
            return results

    def set(self, key: str, value: np.ndarray) -> None:
        if self._cache is None:
            return
        # Ensure 1-D shape and contiguous memory for predictable storage.
        if value.ndim != 1:
            value = np.asarray(value).reshape(-1)
        with self._lock:
            self._cache[key] = value

    def set_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        if self._cache is None:
            return
        with self._lock:
            for key, value in items:
                if value.ndim != 1:
                    value = np.asarray(value).reshape(-1)
                self._cache[key] = value

    def clear(self) -> None:
        if self._cache is None:
            return
        with self._lock:
            self._cache.clear()

    def shrink(self, factor: float) -> None:
        """Keep approximately ``factor`` of current entries (most-recent)."""
        if self._cache is None or not 0 < factor < 1:
            return
        with self._lock:
            keep = int(len(self._cache) * factor)
            while len(self._cache) > keep:
                self._cache.popitem()

    def stats(self) -> dict[str, int]:
        with self._lock:
            size = len(self._cache) if self._cache is not None else 0
            return {"size": size, "hits": self._hits, "misses": self._misses}
