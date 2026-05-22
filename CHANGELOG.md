# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0]

### Added
- Settings module (`pydantic-settings`) — single source of truth for env vars.
- App factory `create_app()` with FastAPI `lifespan` for clean startup/shutdown.
- `ModelRegistry` consolidating per-model metadata (HF id, ONNX, dim, max
  tokens, cost weight, Matryoshka dims).
- Lazy model loading with double-checked async locks.
- Multi-tier quota & rate limiting (anonymous / free / pro):
  per-second / per-minute / per-hour / per-day request and Compute-Unit
  buckets; refund on actual cost.
- Concurrency limiter (global / per-identity / per-model).
- Adaptive shedder (CPU/RAM/queue thresholds; tier-aware).
- Memory guard with cache shrink + panic.
- Image fetch SSRF guard, content-type/size streaming caps, decompression
  bomb cap (`PIL.MAX_IMAGE_PIXELS`).
- Endpoints: `GET /v1/models`, `GET /v1/quota`, `GET /healthz`,
  `GET /readyz`, `GET /metrics`.
- Prometheus metrics + `RequestIDMiddleware`.
- OpenAI-compatible `encoding_format=base64` and Matryoshka `dimensions`.
- Tests: cache key, math ops, periods, auth, log filter, cost, rate limiter,
  settings, registry, identity, tokens.
- Multi-stage `Dockerfile` (slim) with `jemalloc` + `HEALTHCHECK`.
- CI workflow with ruff + pytest + docker build smoke.
- Comprehensive `README.md` rewrite.

### Changed
- `requirements.txt` and `pyproject.toml` synchronised; dependencies pinned.
- Default response class is `ORJSONResponse` (3-10× faster JSON encoding).
- Cosine similarity uses normalized embeddings for the rank hot path.
- Logging routed through `dictConfig`; access log filter actually drops
  `/v1` lines (previously the logic was inverted).
- Authentication uses `hmac.compare_digest` (constant-time).
- `datetime.utcnow()` replaced by `datetime.now(timezone.utc)` everywhere.
- Analytics period keys use ISO calendar (`%G-W%V`) and survive year
  boundaries; flush is snapshot-then-swap and pipelined to Upstash.

### Fixed
- Cache collisions caused by 32-bit MD5 prefix — now 128-bit BLAKE2b.
- Cache only benefited single-text requests — now batched hits/misses.
- Truncation tokenized text 2-3× — now relies on
  `SentenceTransformer.max_seq_length`.
- `estimate_tokens` ignored the requested model — now uses the right
  tokenizer per model.
- Inference was serialised by an unnecessary lock — removed.
- Image processor ran in the event loop — now offloaded to a thread, single
  batch call.
- HTTP client missed pool/HTTP2/UA configuration — now configured.
- Rate limit was per-process, leaked memory, not thread-safe — now in
  TTLCache with proper eviction.
- 500 errors echoed `str(exception)` to clients — now generic with
  `request_id`.
- Module-level side effects (model loading, Redis connect) on import —
  removed in favour of lifespan.

### Removed
- Legacy `service.py`, `router.py`, `analytics.py` flat modules.
- `requests` and `pandas` runtime dependencies.

## [1.0.0]
- Initial public version.
