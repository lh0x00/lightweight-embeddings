"""Embeddings service: lazy-loading, thread-safe, async-friendly.

Design goals over the previous implementation:

* **Lazy loading** of models — the previous version eagerly preloaded all 10
  models on import, costing 8-12 GB of RAM per worker.
* **No spurious locking** around inference — sentence-transformers is
  thread-safe under the GIL; the only lock we keep is for the *load* step.
* **Item-level caching** that benefits batched requests: misses are batched
  and encoded together, hits are returned from the cache.
* **Embedding-side normalization** — we always store/return L2-normalized
  vectors so cosine similarity reduces to a dot product on the hot path.
* **Safe image fetching** with SSRF, content-type, size and decompression
  bomb guards. Pre-processing and inference happen in a single thread offload.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import math
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image

from ..settings import Settings
from . import math_ops
from .cache import EmbeddingCache, make_cache_key
from .device import build_ort_session_options, ort_providers, select_device
from .registry import REGISTRY, ModelKind, ModelRegistry, ModelSpec
from .tokens import count_tokens_exact, estimate_tokens_fast

logger = logging.getLogger(__name__)


_BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
]


class ImageFetchError(ValueError):
    """Raised when an image cannot be fetched/decoded safely."""


class EmbeddingsService:
    """Async-friendly embedding generator with lazy model loading."""

    def __init__(self, settings: Settings, registry: ModelRegistry = REGISTRY) -> None:
        self._settings = settings
        self._registry = registry

        self._device = select_device(settings.device)
        # Pre-compute exp(logit_scale)=100 once; matches the historical default
        # used for text-only ranking without per-request recomputation.
        self._rank_scale: float = math.exp(4.60517)

        # Apply PIL bomb guard from settings (kept module-level for safety).
        Image.MAX_IMAGE_PIXELS = settings.image_max_pixels

        # Lazily-populated model containers.
        self._text_models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._image_models: dict[str, Any] = {}
        self._image_processors: dict[str, Any] = {}

        # Per-model load locks (asyncio): the load itself runs in a thread, but
        # we serialize concurrent loads of the *same* model to avoid double
        # work and racing weight downloads.
        self._load_locks: dict[str, asyncio.Lock] = {
            spec.name: asyncio.Lock() for spec in registry
        }

        # In-process embedding cache.
        self.cache = EmbeddingCache(maxsize=settings.embedding_cache_size)

        # Concurrency limit for image downloads.
        self._image_semaphore = asyncio.Semaphore(settings.image_fetch_concurrency)

        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Initialize HTTP client and preload requested models."""
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self._settings.http_connect_timeout_s,
                read=self._settings.http_read_timeout_s,
                write=self._settings.http_write_timeout_s,
                pool=self._settings.http_pool_timeout_s,
            ),
            limits=httpx.Limits(
                max_connections=self._settings.http_max_connections,
                max_keepalive_connections=self._settings.http_max_keepalive,
            ),
            http2=self._settings.http_http2,
            follow_redirects=False,
            headers={"User-Agent": "lightweight-embeddings/1.1"},
        )

        targets = self._resolve_preload_targets()
        if targets:
            logger.info("preloading models: %s", ", ".join(targets))
            await asyncio.gather(*(self._ensure_loaded(name) for name in targets))

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _resolve_preload_targets(self) -> list[str]:
        items = self._settings.models_preload_list
        if not items:
            return []
        if items == ["*"]:
            return [s.name for s in self._registry]
        valid = []
        for name in items:
            if name in self._registry:
                valid.append(name)
            else:
                logger.warning("models_preload: unknown model %r ignored", name)
        return valid

    # ------------------------------------------------------------------ #
    # Inspection                                                          #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        return self._device

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def is_loaded(self, name: str) -> bool:
        spec = self._registry.get(name)
        if spec.kind is ModelKind.TEXT:
            return name in self._text_models
        return name in self._image_models

    def loaded_models(self) -> list[str]:
        return [*self._text_models.keys(), *self._image_models.keys()]

    # ------------------------------------------------------------------ #
    # Lazy model loading                                                  #
    # ------------------------------------------------------------------ #

    async def _ensure_loaded(self, name: str) -> ModelSpec:
        spec = self._registry.get(name)
        if spec.kind is ModelKind.TEXT and name in self._text_models:
            return spec
        if spec.kind is ModelKind.IMAGE and name in self._image_models:
            return spec
        async with self._load_locks[name]:
            # Double-checked locking.
            if spec.kind is ModelKind.TEXT and name in self._text_models:
                return spec
            if spec.kind is ModelKind.IMAGE and name in self._image_models:
                return spec
            await asyncio.to_thread(self._load_blocking, spec)
        return spec

    def _load_blocking(self, spec: ModelSpec) -> None:
        if spec.kind is ModelKind.TEXT:
            self._load_text_model(spec)
        else:
            self._load_image_model(spec)

    def _load_text_model(self, spec: ModelSpec) -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs: dict[str, Any] = {}
        if spec.onnx_file is not None:
            session_options = build_ort_session_options(
                intra_threads=self._settings.onnx_intra_threads or 0,
                inter_threads=self._settings.onnx_inter_threads,
            )
            providers = ort_providers(self._device)
            model_kwargs.update(
                providers=providers,
                file_name=spec.onnx_file,
            )
            if session_options is not None:
                model_kwargs["session_options"] = session_options

        backend = "onnx" if spec.onnx_file else "torch"
        logger.info(
            "loading text model %s (backend=%s, device=%s)",
            spec.name, backend, self._device,
        )
        model = SentenceTransformer(
            spec.hf_id,
            device=self._device,
            backend=backend,
            model_kwargs=model_kwargs or None,
            trust_remote_code=True,
        )
        model.max_seq_length = spec.max_seq_length

        # Resolve the underlying tokenizer once for token counting.
        tokenizer = None
        try:
            first_module = model._first_module()
            tokenizer = getattr(first_module, "tokenizer", None)
        except Exception:
            tokenizer = None

        self._text_models[spec.name] = model
        if tokenizer is not None:
            self._tokenizers[spec.name] = tokenizer

    def _load_image_model(self, spec: ModelSpec) -> None:
        from transformers import AutoModel, AutoProcessor

        logger.info("loading image model %s (device=%s)", spec.name, self._device)
        model = AutoModel.from_pretrained(spec.hf_id)
        model = model.to(self._device)
        model.eval()
        if self._device == "cuda":
            try:
                model = model.half()
            except Exception:  # pragma: no cover
                logger.debug("model.half() not supported for %s", spec.name)
        processor = AutoProcessor.from_pretrained(spec.hf_id)

        self._image_models[spec.name] = model
        self._image_processors[spec.name] = processor

    # ------------------------------------------------------------------ #
    # Public embedding/rank API                                           #
    # ------------------------------------------------------------------ #

    async def generate_embeddings(
        self,
        model: str,
        inputs: str | list[str],
        *,
        normalize: bool = True,
        dimensions: int | None = None,
    ) -> tuple[np.ndarray, ModelSpec]:
        """Generate embeddings for ``inputs`` using ``model``.

        Returns the matrix and the resolved :class:`ModelSpec`. The matrix is
        L2-normalized when ``normalize=True`` (default).
        """
        spec = await self._ensure_loaded(model)
        items = _coerce_to_list(inputs)
        truncate_to = spec.truncate_dim(dimensions)

        if spec.kind is ModelKind.TEXT:
            matrix = await self._embed_text(spec, items, normalize=normalize)
        elif spec.kind is ModelKind.IMAGE:
            matrix = await self._embed_image(spec, items, normalize=normalize)
        else:  # pragma: no cover - exhaustive
            raise RuntimeError(f"unsupported modality: {spec.kind}")

        if truncate_to is not None:
            matrix = matrix[:, :truncate_to]
            if normalize:
                matrix = math_ops.normalize(matrix, axis=1)
        return matrix, spec

    async def rank(
        self,
        model: str,
        queries: str | list[str],
        candidates: str | list[str],
    ) -> dict[str, Any]:
        """Rank ``candidates`` against ``queries`` using cosine + softmax."""
        q_items = _coerce_to_list(queries)
        c_items = _coerce_to_list(candidates)
        spec = await self._ensure_loaded(model)

        q_task = asyncio.create_task(self.generate_embeddings(model, q_items, normalize=True))
        c_task = asyncio.create_task(self.generate_embeddings(model, c_items, normalize=True))
        (q_emb, _), (c_emb, _) = await asyncio.gather(q_task, c_task)

        sim = math_ops.cosine_similarity_normalized(q_emb, c_emb)
        probs = math_ops.softmax(self._rank_scale * sim, axis=-1)

        usage_tokens = 0
        if spec.kind is ModelKind.TEXT:
            usage_tokens = self.count_tokens(model, q_items) + self.count_tokens(model, c_items)

        return {
            "probabilities": probs.tolist(),
            "cosine_similarities": sim.tolist(),
            "usage": {"prompt_tokens": usage_tokens, "total_tokens": usage_tokens},
        }

    # ------------------------------------------------------------------ #
    # Token accounting                                                    #
    # ------------------------------------------------------------------ #

    def count_tokens(self, model: str, inputs: str | list[str]) -> int:
        """Exact token count for *the requested model*."""
        items = _coerce_to_list(inputs)
        spec = self._registry.get(model)
        if spec.kind is not ModelKind.TEXT:
            return 0
        tokenizer = self._tokenizers.get(model)
        if tokenizer is None:
            return estimate_tokens_fast(items)
        return count_tokens_exact(tokenizer, items)

    # ------------------------------------------------------------------ #
    # Text embedding internals                                            #
    # ------------------------------------------------------------------ #

    async def _embed_text(
        self,
        spec: ModelSpec,
        texts: list[str],
        *,
        normalize: bool,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("text input list cannot be empty")
        for t in texts:
            if not isinstance(t, str):
                raise ValueError("text inputs must be strings")

        # Version cache by normalization flag (different vectors).
        key_prefix = f"{spec.name}|n={int(normalize)}"
        keys = [make_cache_key(key_prefix, t) for t in texts]
        cached = self.cache.get_many(keys)

        miss_indices = [i for i, c in enumerate(cached) if c is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            new_matrix = await asyncio.to_thread(
                self._encode_text_blocking, spec.name, miss_texts, normalize
            )
            for offset, idx in enumerate(miss_indices):
                vec = new_matrix[offset]
                cached[idx] = vec
                self.cache.set(keys[idx], vec)

        return np.stack(cached, axis=0)

    def _encode_text_blocking(
        self, name: str, texts: list[str], normalize: bool
    ) -> np.ndarray:
        model = self._text_models[name]
        # sentence-transformers handles truncation internally via max_seq_length.
        embeddings = model.encode(
            texts,
            batch_size=self._settings.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        # Ensure float32 contiguous output.
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        return embeddings

    # ------------------------------------------------------------------ #
    # Image embedding internals                                           #
    # ------------------------------------------------------------------ #

    async def _embed_image(
        self,
        spec: ModelSpec,
        urls: list[str],
        *,
        normalize: bool,
    ) -> np.ndarray:
        if not urls:
            raise ValueError("image input list cannot be empty")

        async def _fetch(idx_url: tuple[int, str]):
            idx, url = idx_url
            async with self._image_semaphore:
                return idx, await self._fetch_image(url)

        tasks = [asyncio.create_task(_fetch(iu)) for iu in enumerate(urls)]
        pil_images: list[Image.Image | None] = [None] * len(urls)
        try:
            for coro in asyncio.as_completed(tasks):
                idx, img = await coro
                pil_images[idx] = img
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

        # Type narrow.
        ordered = [img for img in pil_images if img is not None]
        if len(ordered) != len(urls):  # pragma: no cover - defensive
            raise ImageFetchError("internal: missing image after fetch")

        return await asyncio.to_thread(
            self._encode_image_blocking, spec.name, ordered, normalize
        )

    def _encode_image_blocking(
        self, name: str, images: list[Image.Image], normalize: bool
    ) -> np.ndarray:
        import torch

        processor = self._image_processors[name]
        model = self._image_models[name]

        inputs = processor(images=images, return_tensors="pt")
        # ``inputs`` is a BatchFeature; supports .to(device).
        inputs = inputs.to(self._device)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out = features.detach().to("cpu", dtype=torch.float32).numpy()
        return out

    # ------------------------------------------------------------------ #
    # Image fetching with SSRF + size + bomb protection                   #
    # ------------------------------------------------------------------ #

    async def _fetch_image(self, target: str) -> Image.Image:
        """Fetch an image from URL or local path with safety guards."""
        if target.startswith(("http://", "https://")):
            return await self._fetch_image_http(target)
        if not self._settings.image_allow_local_paths:
            raise ImageFetchError("local image paths are disabled")
        # Both ``is_file`` and ``Image.open`` are blocking; do them in a
        # single thread offload so we never touch the file system from the
        # event loop.
        return await asyncio.to_thread(_open_image_from_path_str, target)

    async def _fetch_image_http(self, url: str) -> Image.Image:
        if self._http is None:
            raise RuntimeError("EmbeddingsService.start() must be called first")

        max_redirects = 3
        current_url = url
        max_bytes = self._settings.image_max_bytes

        for _ in range(max_redirects + 1):
            parsed = urlparse(current_url)
            if parsed.scheme not in ("http", "https"):
                raise ImageFetchError(f"scheme not allowed: {parsed.scheme}")
            host = parsed.hostname
            if not host:
                raise ImageFetchError("url missing host")
            await asyncio.to_thread(_assert_host_public, host)

            try:
                async with self._http.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        loc = response.headers.get("location")
                        if not loc:
                            raise ImageFetchError("redirect without location")
                        current_url = loc
                        continue
                    response.raise_for_status()
                    ctype = response.headers.get("content-type", "").split(";")[0].strip()
                    if ctype and not ctype.startswith("image/"):
                        raise ImageFetchError(f"non-image content-type: {ctype}")
                    declared = int(response.headers.get("content-length", "0") or 0)
                    if declared and declared > max_bytes:
                        raise ImageFetchError("image exceeds max bytes")
                    buffer = bytearray()
                    async for chunk in response.aiter_bytes(64 * 1024):
                        buffer.extend(chunk)
                        if len(buffer) > max_bytes:
                            raise ImageFetchError("image exceeds max bytes (stream)")
                    return await asyncio.to_thread(_open_image_from_bytes, bytes(buffer))
            except httpx.HTTPError as exc:
                raise ImageFetchError(f"image fetch failed: {exc}") from exc

        raise ImageFetchError("too many redirects")


# --------------------------------------------------------------------------- #
# Module-level helpers                                                        #
# --------------------------------------------------------------------------- #


def _coerce_to_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("input cannot be empty")
        return [value]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError("input must be a string or list of strings")
    if not value:
        raise ValueError("input list cannot be empty")
    return value


def _assert_host_public(host: str) -> None:
    """Resolve ``host`` and reject if any address is in a blocked network."""
    try:
        addr = ipaddress.ip_address(host)
        addrs = [addr]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise ImageFetchError(f"dns resolution failed: {exc}") from exc
        addrs = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr and sockaddr[0]:
                try:
                    addrs.append(ipaddress.ip_address(sockaddr[0]))
                except ValueError:
                    continue
    for ip in addrs:
        for net in _BLOCKED_IP_NETWORKS:
            if ip in net:
                raise ImageFetchError(f"blocked private/loopback host: {host}")


def _open_image_from_path(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.load()
        return img.convert("RGB")


def _open_image_from_path_str(target: str) -> Image.Image:
    path = Path(target)
    if not path.is_file():
        raise ImageFetchError(f"local image not found: {target}")
    return _open_image_from_path(path)


def _open_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return img.convert("RGB")
