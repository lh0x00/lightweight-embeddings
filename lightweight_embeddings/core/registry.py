"""Single source of truth for supported models.

Each :class:`ModelSpec` captures everything needed at runtime to load and serve
a model: HuggingFace ID, ONNX file (optional), max sequence length, modality,
cost weight (for quota accounting) and Matryoshka-allowed dimensions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class ModelKind(str, Enum):
    """Modality of a registered model."""

    TEXT = "text"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static metadata for a registered model."""

    name: str
    kind: ModelKind
    hf_id: str
    onnx_file: str | None = None
    max_seq_length: int = 512
    embedding_dim: int = 0
    cost_weight: float = 1.0
    matryoshka_dims: tuple[int, ...] = field(default_factory=tuple)
    family: str = ""

    def supports_dimensions(self, dim: int) -> bool:
        """Return True if the model can serve embeddings of the given size."""
        if not self.matryoshka_dims:
            return dim == self.embedding_dim
        return dim in self.matryoshka_dims

    def truncate_dim(self, requested: int | None) -> int | None:
        """Resolve the effective output dimension for a request.

        Returns ``None`` when no truncation is requested. Raises ``ValueError``
        for unsupported dimensions.
        """
        if requested is None:
            return None
        if requested <= 0:
            raise ValueError("dimensions must be positive")
        if not self.matryoshka_dims:
            if requested != self.embedding_dim:
                raise ValueError(
                    f"model '{self.name}' does not support custom dimensions; "
                    f"native dim={self.embedding_dim}"
                )
            return None
        if requested not in self.matryoshka_dims:
            allowed = ", ".join(str(d) for d in self.matryoshka_dims)
            raise ValueError(
                f"model '{self.name}' supports dimensions: [{allowed}]; got {requested}"
            )
        if requested == self.embedding_dim:
            return None
        return requested


class ModelRegistry:
    """Read-only registry of supported models indexed by short name."""

    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        self._by_name: dict[str, ModelSpec] = {}
        for spec in specs:
            if spec.name in self._by_name:
                raise ValueError(f"duplicate model name: {spec.name}")
            self._by_name[spec.name] = spec

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    def get(self, name: str) -> ModelSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise UnknownModelError(name) from exc

    def names(self, kind: ModelKind | None = None) -> list[str]:
        if kind is None:
            return list(self._by_name.keys())
        return [s.name for s in self._by_name.values() if s.kind is kind]

    def specs(self, kind: ModelKind | None = None) -> list[ModelSpec]:
        if kind is None:
            return list(self._by_name.values())
        return [s for s in self._by_name.values() if s.kind is kind]


class UnknownModelError(ValueError):
    """Raised when a request references a model that is not registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown model: {name!r}")


# ---------------------------------------------------------------------------
# Built-in model definitions
# ---------------------------------------------------------------------------

_TEXT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="multilingual-e5-small",
        kind=ModelKind.TEXT,
        hf_id="Xenova/multilingual-e5-small",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=512,
        embedding_dim=384,
        cost_weight=1.0,
        family="e5",
    ),
    ModelSpec(
        name="multilingual-e5-base",
        kind=ModelKind.TEXT,
        hf_id="Xenova/multilingual-e5-base",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=512,
        embedding_dim=768,
        cost_weight=2.0,
        family="e5",
    ),
    ModelSpec(
        name="multilingual-e5-large",
        kind=ModelKind.TEXT,
        hf_id="Xenova/multilingual-e5-large",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=512,
        embedding_dim=1024,
        cost_weight=4.0,
        family="e5",
    ),
    ModelSpec(
        name="snowflake-arctic-embed-l-v2.0",
        kind=ModelKind.TEXT,
        hf_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=8192,
        embedding_dim=1024,
        cost_weight=5.0,
        matryoshka_dims=(256, 512, 1024),
        family="snowflake",
    ),
    ModelSpec(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        kind=ModelKind.TEXT,
        hf_id="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=128,
        embedding_dim=384,
        cost_weight=0.8,
        family="minilm",
    ),
    ModelSpec(
        name="paraphrase-multilingual-mpnet-base-v2",
        kind=ModelKind.TEXT,
        hf_id="Xenova/paraphrase-multilingual-mpnet-base-v2",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=128,
        embedding_dim=768,
        cost_weight=2.0,
        family="mpnet",
    ),
    ModelSpec(
        name="bge-m3",
        kind=ModelKind.TEXT,
        hf_id="Xenova/bge-m3",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=8192,
        embedding_dim=1024,
        cost_weight=5.0,
        family="bge",
    ),
    ModelSpec(
        name="gte-multilingual-base",
        kind=ModelKind.TEXT,
        hf_id="onnx-community/gte-multilingual-base",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=8192,
        embedding_dim=768,
        cost_weight=2.5,
        family="gte",
    ),
    ModelSpec(
        name="embeddinggemma-300m",
        kind=ModelKind.TEXT,
        hf_id="onnx-community/embeddinggemma-300m-ONNX",
        onnx_file="onnx/model_quantized.onnx",
        max_seq_length=2048,
        embedding_dim=768,
        cost_weight=4.0,
        matryoshka_dims=(128, 256, 512, 768),
        family="gemma",
    ),
)

_IMAGE_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="siglip-base-patch16-256-multilingual",
        kind=ModelKind.IMAGE,
        hf_id="google/siglip-base-patch16-256-multilingual",
        onnx_file=None,
        max_seq_length=0,
        embedding_dim=768,
        cost_weight=6.0,
        family="siglip",
    ),
)

REGISTRY: ModelRegistry = ModelRegistry((*_TEXT_SPECS, *_IMAGE_SPECS))


def detect_model_kind(name: str) -> ModelKind:
    """Return the modality of a registered model.

    Raises:
        UnknownModelError: when the name is not registered.
    """
    return REGISTRY.get(name).kind
