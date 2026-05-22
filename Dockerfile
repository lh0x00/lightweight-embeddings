# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Lightweight Embeddings — multi-stage image (CPU-only by default).
#
#   Stage 1  builder   compiles wheels from requirements.txt into /install
#   Stage 2  runtime   slim image with jemalloc + healthcheck + non-root user
#
# Build:
#   docker build -t lightweight-embeddings .
# Run:
#   docker run --rm -p 7860:7860 lightweight-embeddings
# -----------------------------------------------------------------------------

ARG PYTHON_VERSION=3.10

# ============================================================================
# Stage 1: builder
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build tools required by torch/transformers wheels with no prebuilt arch.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates

WORKDIR /build
COPY requirements.txt ./

# CPU torch wheels are an order of magnitude smaller than the CUDA ones.
# Override at build time with: --build-arg TORCH_INDEX_URL=...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --prefix=/install \
        --extra-index-url "${TORCH_INDEX_URL}" \
        -r requirements.txt

# ============================================================================
# Stage 2: runtime
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="lightweight-embeddings" \
      org.opencontainers.image.description="Multilingual text+image embeddings & reranking API" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/lh0x00/lightweight-embeddings"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

# jemalloc keeps RSS predictable for workloads with frequent (de)allocations.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libjemalloc2 ca-certificates curl

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Non-root.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Pull the prebuilt site-packages from stage 1.
COPY --from=builder /install /usr/local

# Application source.
COPY --chown=user . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/healthz >/dev/null || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
