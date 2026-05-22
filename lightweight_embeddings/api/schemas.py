"""Pydantic v2 request/response schemas with strict input validation."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.registry import REGISTRY

# Build a list of valid model names from the registry; used for OpenAPI examples
# but NOT for Literal typing because that would be too strict for users still
# adding custom models via env in the future.
_DEFAULT_TEXT = "multilingual-e5-small"

EncodingFormat = Literal["float", "base64"]


class EmbeddingRequest(BaseModel):
    """Request body for ``POST /v1/embeddings``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(
        default=_DEFAULT_TEXT,
        min_length=1,
        max_length=128,
        description="Registered model name. See `GET /v1/models`.",
    )
    input: Annotated[
        str | list[str],
        Field(
            description="A single text/image URL or a list of them.",
        ),
    ]
    encoding_format: EncodingFormat = Field(
        default="float",
        description="`float` returns a JSON array; `base64` returns the raw "
        "float32 little-endian bytes encoded as base64 (OpenAI-compatible).",
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Optional Matryoshka truncation. Only supported by some models.",
    )
    user: str | None = Field(
        default=None, max_length=128, description="Optional caller hint, OpenAI-compat."
    )


class RankRequest(BaseModel):
    """Request body for ``POST /v1/rank``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(
        default=_DEFAULT_TEXT, min_length=1, max_length=128,
    )
    queries: Annotated[
        str | list[str],
        Field(description="Query text(s) or image URL(s) for image models."),
    ]
    candidates: Annotated[
        list[str],
        Field(min_length=1, description="Candidate texts to rank."),
    ]


class EmbeddingDataItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingDataItem]
    model: str
    usage: Usage


class RankResponse(BaseModel):
    probabilities: list[list[float]]
    cosine_similarities: list[list[float]]
    usage: Usage


class StatsBucket(BaseModel):
    total: dict[str, int] = Field(default_factory=dict)
    daily: dict[str, int] = Field(default_factory=dict)
    weekly: dict[str, int] = Field(default_factory=dict)
    monthly: dict[str, int] = Field(default_factory=dict)
    yearly: dict[str, int] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    access: StatsBucket
    tokens: StatsBucket


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    kind: Literal["text", "image"]
    family: str = ""
    embedding_dim: int
    max_seq_length: int
    matryoshka_dims: list[int] = Field(default_factory=list)
    cost_weight: float
    loaded: bool = False


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ok", "loading", "degraded"]
    version: str
    models_loaded: list[str]
    device: str
    cpu_percent: float
    memory_percent: float


class QuotaResponse(BaseModel):
    tier: str
    daily_request_limit: int
    daily_request_remaining: int
    daily_cu_limit: float
    daily_cu_remaining: float
    minute_request_limit: int
    minute_request_remaining: int
    concurrency_limit: int


class ErrorResponse(BaseModel):
    error: dict[str, Any]


def all_model_names() -> list[str]:
    return [s.name for s in REGISTRY]
