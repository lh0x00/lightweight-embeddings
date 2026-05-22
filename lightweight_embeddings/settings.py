"""Application settings loaded from environment variables.

All configuration is centralized here using pydantic-settings.
Environment variables are prefixed with ``LWE_`` (case-insensitive).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DeviceMode = Literal["auto", "cpu", "cuda"]
RateLimitBackend = Literal["memory", "redis"]
QuantLevel = Literal["int8", "fp16", "fp32"]


class Settings(BaseSettings):
    """Runtime settings; values can be overridden by environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="LWE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- bootstrap ----------
    log_level: LogLevel = "INFO"
    log_json: bool = False
    enable_ui: bool = True
    debug: bool = False

    # ---------- security ----------
    access_token: SecretStr | None = None
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False
    trusted_proxies: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ---------- models ----------
    models_preload: str = "multilingual-e5-small"
    default_text_model: str = "multilingual-e5-small"
    default_image_model: str = "siglip-base-patch16-256-multilingual"

    # ---------- device / runtime ----------
    device: DeviceMode = "auto"
    onnx_intra_threads: int | None = None
    onnx_inter_threads: int = 1
    torch_compile: bool = False
    torch_num_threads: int | None = None
    quant_level: QuantLevel = "int8"

    # ---------- batching ----------
    encode_batch_size: int = 64
    micro_batch_max: int = 64
    micro_batch_delay_ms: int = 4

    # ---------- cache ----------
    embedding_cache_size: int = 50_000

    # ---------- image ----------
    image_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    image_max_pixels: int = 50_000_000  # 50 MP
    image_fetch_concurrency: int = 8
    image_allow_local_paths: bool = False
    image_max_per_request: int = 16

    # ---------- request limits ----------
    max_body_bytes: int = 2 * 1024 * 1024  # 2 MB
    max_total_chars: int = 500_000
    max_items_per_request: int = 128

    # ---------- rate limit / quota ----------
    rate_limit_backend: RateLimitBackend = "memory"

    # anonymous tier
    rl_anon_rps: float = 1.0
    rl_anon_burst: int = 3
    rl_anon_rpm: int = 30
    rl_anon_rph: int = 300
    rl_anon_rpd: int = 2_000
    rl_anon_daily_cu: float = 2_000.0
    rl_anon_daily_tokens: int = 200_000
    rl_anon_max_items: int = 16
    rl_anon_max_chars: int = 50_000
    rl_anon_concurrency: int = 2

    # free authenticated tier
    rl_free_rps: float = 5.0
    rl_free_burst: int = 20
    rl_free_rpm: int = 200
    rl_free_rph: int = 5_000
    rl_free_rpd: int = 50_000
    rl_free_daily_cu: float = 50_000.0
    rl_free_daily_tokens: int = 5_000_000
    rl_free_max_items: int = 128
    rl_free_max_chars: int = 500_000
    rl_free_concurrency: int = 8

    # pro / internal (configurable, defaults very generous)
    rl_pro_rps: float = 30.0
    rl_pro_burst: int = 100
    rl_pro_daily_cu: float = 1_000_000.0
    rl_pro_concurrency: int = 32

    # ---------- adaptive shedding ----------
    shed_cpu_anon_pct: float = 92.0
    shed_cpu_free_pct: float = 96.0
    shed_cpu_global_pct: float = 99.0
    shed_queue_max: int = 200
    shed_rss_pct: float = 88.0
    shed_sample_interval_s: float = 1.0

    # ---------- memory guard ----------
    mem_high_watermark_pct: float = 90.0
    mem_panic_watermark_pct: float = 95.0

    # ---------- concurrency ----------
    concurrency_global: int = 64
    concurrency_per_model: int = 16

    # ---------- analytics / redis ----------
    redis_url: str | None = None
    redis_token: SecretStr | None = None
    analytics_sync_interval_s: int = 30 * 60
    analytics_retention_days: int = 90
    analytics_retention_weeks: int = 16
    analytics_retention_months: int = 24
    analytics_retention_years: int = 5
    analytics_max_flush_failures: int = 5

    # ---------- observability ----------
    enable_metrics: bool = True
    metrics_path: str = "/metrics"
    request_id_header: str = "X-Request-ID"

    # ---------- HTTP client ----------
    http_connect_timeout_s: float = 3.0
    http_read_timeout_s: float = 10.0
    http_write_timeout_s: float = 5.0
    http_pool_timeout_s: float = 5.0
    http_max_connections: int = 64
    http_max_keepalive: int = 32
    http_http2: bool = True

    # ---------- validators ----------
    @field_validator("cors_origins", "trusted_proxies", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("models_preload")
    @classmethod
    def _normalize_preload(cls, v: str) -> str:
        v = (v or "").strip()
        return v or "none"

    @property
    def models_preload_list(self) -> list[str]:
        if self.models_preload in ("none", ""):
            return []
        if self.models_preload == "*":
            return ["*"]
        return [m.strip() for m in self.models_preload.split(",") if m.strip()]


_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The cache avoids repeated env parsing while still allowing tests to
    override by calling :func:`reset_settings`.
    """
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
    return _cached_settings


def reset_settings() -> None:
    """Clear the cached settings (test helper)."""
    global _cached_settings
    _cached_settings = None
