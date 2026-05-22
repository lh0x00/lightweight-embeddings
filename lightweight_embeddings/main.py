"""FastAPI application factory and lifespan.

This module is the single entry point used by ``app.py`` and tests. It is
intentionally free of import-time side effects: model loading and Redis
connections happen inside the ``lifespan`` block.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from . import __version__
from .analytics.service import build_analytics_service
from .api.errors import install_handlers
from .api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitHeadersMiddleware,
    RequestIDMiddleware,
)
from .api.routes import embeddings as embeddings_route
from .api.routes import health as health_route
from .api.routes import models as models_route
from .api.routes import quota as quota_route
from .api.routes import rank as rank_route
from .api.routes import stats as stats_route
from .core.device import configure_threading
from .core.service import EmbeddingsService
from .logging_config import configure_logging
from .observability import (
    MODELS_LOADED,
    SYSTEM_CPU,
    SYSTEM_RSS,
    MetricsMiddleware,
    metrics_endpoint,
)
from .security.concurrency import ConcurrencyLimiter
from .security.memguard import MemoryGuard
from .security.ratelimit import build_rate_limiter
from .security.shedder import AdaptiveShedder
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a fully-wired FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings)

    threading_info = configure_threading(
        onnx_intra_threads=settings.onnx_intra_threads,
        torch_num_threads=settings.torch_num_threads,
    )
    logger.info("threading: %s", threading_info)

    app = FastAPI(
        title="Lightweight Embeddings API",
        description=(
            "Fast, free, multilingual embeddings & reranking service. "
            "Supports text and image inputs."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=ORJSONResponse,
        lifespan=_lifespan,
    )

    app.state.settings = settings

    # ---------------- middleware ----------------
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(MetricsMiddleware, settings=settings)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RateLimitHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware, header_name=settings.request_id_header)

    # ---------------- error handlers ----------------
    install_handlers(app)

    # ---------------- routes ----------------
    app.include_router(embeddings_route.router, prefix="/v1")
    app.include_router(rank_route.router, prefix="/v1")
    app.include_router(stats_route.router, prefix="/v1")
    app.include_router(models_route.router, prefix="/v1")
    app.include_router(quota_route.router, prefix="/v1")
    app.include_router(health_route.router)
    if settings.enable_metrics:
        app.add_api_route(
            settings.metrics_path,
            metrics_endpoint,
            methods=["GET"],
            include_in_schema=False,
        )

    # ---------------- UI ----------------
    if settings.enable_ui:
        try:
            from .web.ui import mount_demo

            mount_demo(app, path="/")
        except Exception as exc:  # pragma: no cover - depends on optional dep
            logger.warning("UI disabled (gradio not available): %s", exc)

    return app


# --------------------------------------------------------------------------- #
# Lifespan                                                                    #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # ---------- service ----------
    service = EmbeddingsService(settings=settings)
    app.state.service = service
    await service.start()
    for name in service.loaded_models():
        MODELS_LOADED.labels(model=name).set(1.0)

    # ---------- analytics ----------
    redis_token = (
        settings.redis_token.get_secret_value() if settings.redis_token else None
    )
    analytics = build_analytics_service(
        redis_url=settings.redis_url,
        redis_token=redis_token,
        sync_interval_s=settings.analytics_sync_interval_s,
        max_consecutive_failures=settings.analytics_max_flush_failures,
    )
    app.state.analytics = analytics
    await analytics.start()

    # ---------- rate limit / concurrency / shedding / memory guard ----------
    app.state.rate_limiter = build_rate_limiter(settings.rate_limit_backend)
    app.state.concurrency = ConcurrencyLimiter(
        global_capacity=settings.concurrency_global,
        per_model_capacity=settings.concurrency_per_model,
    )
    shedder = AdaptiveShedder(settings)
    await shedder.start()
    app.state.shedder = shedder
    app.state.memguard = MemoryGuard(
        settings,
        on_pressure=lambda: service.cache.shrink(0.5),
        on_panic=lambda: service.cache.clear(),
    )

    # Mirror system metrics.
    SYSTEM_CPU.set(0.0)
    SYSTEM_RSS.set(0.0)

    logger.info("application ready (device=%s)", service.device)
    try:
        yield
    finally:
        logger.info("shutting down…")
        try:
            await shedder.close()
        except Exception:
            logger.exception("shedder shutdown failed")
        try:
            await analytics.close()
        except Exception:
            logger.exception("analytics shutdown failed")
        try:
            await service.close()
        except Exception:
            logger.exception("service shutdown failed")
        logger.info("shutdown complete")
