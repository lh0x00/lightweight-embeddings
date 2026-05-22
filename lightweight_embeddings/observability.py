"""Prometheus metrics: registration + HTTP middleware.

Uses the default global registry so metrics survive across reloads in dev.
The endpoint is exposed by :func:`install_metrics` on a FastAPI app.
"""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

from .settings import Settings

REQUEST_COUNT = Counter(
    "lwe_http_requests_total",
    "HTTP requests by route, method, and status.",
    ["route", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "lwe_http_request_duration_seconds",
    "Request duration in seconds.",
    ["route", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
EMBED_TOKENS = Counter(
    "lwe_embedding_tokens_total",
    "Tokens consumed by embeddings/rank operations.",
    ["model", "kind"],
)
EMBED_REQUESTS = Counter(
    "lwe_embedding_requests_total",
    "Embedding/rank requests by model and outcome.",
    ["model", "operation", "outcome"],
)
MODELS_LOADED = Gauge(
    "lwe_models_loaded",
    "Whether a registered model is currently loaded.",
    ["model"],
)
CACHE_HITS = Counter("lwe_embedding_cache_hits_total", "Embedding cache hits.")
CACHE_MISSES = Counter("lwe_embedding_cache_misses_total", "Embedding cache misses.")
RATE_LIMIT_REJECTS = Counter(
    "lwe_rate_limit_rejects_total",
    "Requests rejected by the rate limiter.",
    ["window", "tier"],
)
SHED_REJECTS = Counter(
    "lwe_shed_rejects_total",
    "Requests dropped by the adaptive shedder.",
    ["tier"],
)
SYSTEM_CPU = Gauge("lwe_cpu_percent", "Recent CPU utilisation (0-100).")
SYSTEM_RSS = Gauge("lwe_memory_percent", "Recent RSS utilisation (0-100).")


async def metrics_endpoint(_: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """Lightweight middleware to record request count + latency."""

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == self.settings.metrics_path:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        # Compute the route template if Starlette has matched it; else use path.
        route = path
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code, route
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Try to read matched route template for grouping.
                rt = scope.get("route")
                if rt is not None and getattr(rt, "path", None):
                    route = rt.path
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_COUNT.labels(route=route, method=method, status=str(status_code)).inc()
            REQUEST_LATENCY.labels(route=route, method=method).observe(elapsed)
