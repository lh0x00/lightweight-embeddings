---
title: Lightweight Embeddings API
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Lightweight Embeddings

A multilingual **text + image** embedding & reranking API.
Production-grade, OpenAI-compatible, single Docker image.

```bash
docker run --rm -p 7860:7860 ghcr.io/lh0x00/lightweight-embeddings
```

→ API: <http://localhost:7860/docs> · Playground: <http://localhost:7860/>

---

## Why

- **One service, ten models** — switch via the `model` field; only the models you actually use are loaded.
- **OpenAI-compatible** — `/v1/embeddings`, `/v1/rank`, `/v1/models`; supports `encoding_format=base64` and Matryoshka `dimensions`.
- **Crash-resistant** — body size limits, request validation, multi-tier rate limits, adaptive shedding, memory guard.
- **Observable** — Prometheus `/metrics`, structured JSON logs, `X-Request-ID` propagation.
- **Slim** — multi-stage `python:3.10-slim` image with `jemalloc` + `HEALTHCHECK`.

## Models

| Name | Kind | Dim | Max tokens | Cost |
|---|---|---|---|---|
| `multilingual-e5-small` *(default)* | text | 384 | 512 | 1.0 |
| `multilingual-e5-base` | text | 768 | 512 | 2.0 |
| `multilingual-e5-large` | text | 1024 | 512 | 4.0 |
| `paraphrase-multilingual-MiniLM-L12-v2` | text | 384 | 128 | 0.8 |
| `paraphrase-multilingual-mpnet-base-v2` | text | 768 | 128 | 2.0 |
| `gte-multilingual-base` | text | 768 | 8192 | 2.5 |
| `bge-m3` | text | 1024 | 8192 | 5.0 |
| `snowflake-arctic-embed-l-v2.0` | text | 1024 *(Matryoshka 256/512/1024)* | 8192 | 5.0 |
| `embeddinggemma-300m` | text | 768 *(Matryoshka 128/256/512/768)* | 2048 | 4.0 |
| `siglip-base-patch16-256-multilingual` | image | 768 | — | 6.0 |

## Usage

```bash
# Embed
curl -X POST http://localhost:7860/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "multilingual-e5-small",
    "input": ["Xin chào", "Hello"],
    "encoding_format": "float"
  }'

# Rerank
curl -X POST http://localhost:7860/v1/rank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "multilingual-e5-small",
    "queries": "happy person",
    "candidates": ["happy dog", "sunny day", "very happy person"]
  }'

# Inspect
curl http://localhost:7860/v1/models
curl http://localhost:7860/healthz
curl http://localhost:7860/v1/quota \
  -H 'Authorization: Bearer $LWE_ACCESS_TOKEN'
```

Use `encoding_format=base64` for ~45% smaller payloads, or `dimensions` to truncate Matryoshka models.

## Quota tiers

| Tier | RPS | Per minute | Per day | CU/day | Concurrency |
|---|---|---|---|---|---|
| **anonymous** | 1 (burst 3) | 30 | 2 000 | 2 000 | 2 |
| **free** *(any valid token)* | 5 (burst 20) | 200 | 50 000 | 50 000 | 8 |
| **pro** *(reserved)* | 30 (burst 100) | configurable | configurable | 1 000 000 | 32 |

CU = `model_cost_weight × tokens / 1000` for text, or `× n_images` for images.
Limits surface in `X-RateLimit-*` headers; `429` and `503` always include `Retry-After`.

## Configuration

All variables are prefixed `LWE_`. Selected highlights — see [`settings.py`](lightweight_embeddings/settings.py) for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `LWE_ACCESS_TOKEN` | unset | Bearer token enabling the *free* tier |
| `LWE_MODELS_PRELOAD` | `multilingual-e5-small` | CSV / `*` / `none` — which models to load on boot |
| `LWE_DEVICE` | `auto` | `auto` · `cpu` · `cuda` |
| `LWE_LOG_JSON` | `false` | Switch logs to structured JSON |
| `LWE_CORS_ORIGINS` | `*` | CSV of allowed origins |
| `LWE_MAX_BODY_BYTES` | `2097152` | Hard request body cap |
| `LWE_CONCURRENCY_GLOBAL` | `64` | Total concurrent requests |
| `LWE_CONCURRENCY_PER_MODEL` | `16` | Per-model concurrency |
| `LWE_REDIS_URL` / `LWE_REDIS_TOKEN` | unset | Upstash Redis for analytics persistence |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/embeddings` | Generate text or image embeddings |
| `POST` | `/v1/rank` | Cosine + softmax reranking |
| `GET`  | `/v1/models` | List registered models |
| `GET`  | `/v1/stats` | Usage analytics *(token-gated)* |
| `GET`  | `/v1/quota` | Caller's current rate-limit state |
| `GET`  | `/healthz` | Liveness |
| `GET`  | `/readyz` | Readiness (model loaded, memory OK) |
| `GET`  | `/metrics` | Prometheus metrics |

## Develop

```bash
pip install -e ".[dev]"
pytest tests/unit -q
ruff check lightweight_embeddings tests
```

Heavy integration tests are gated behind the `integration` pytest marker.

## License

MIT — see [LICENSE](LICENSE).
