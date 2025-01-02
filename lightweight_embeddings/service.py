"""
Lightweight Embeddings Service Module (Revised & Simplified)

This module provides a service for generating and comparing embeddings from text and images
using state-of-the-art transformer models. It supports both CPU and GPU inference.

Features:
- Text and image embedding generation
- Cross-modal similarity ranking
- Batch processing support
- Asynchronous API support

Supported Text Model IDs:
- "multilingual-e5-small"
- "multilingual-e5-base"
- "multilingual-e5-large"
- "snowflake-arctic-embed-l-v2.0"
- "paraphrase-multilingual-MiniLM-L12-v2"
- "paraphrase-multilingual-mpnet-base-v2"
- "bge-m3"
- "gte-multilingual-base"

Supported Image Model IDs:
- "google/siglip-base-patch16-256-multilingual" (default, but extensible)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Union, Literal, Dict, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

import requests
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextModelType(str, Enum):
    """
    Enumeration of supported text models.
    Adjust as needed for your environment.
    """

    MULTILINGUAL_E5_SMALL = "multilingual-e5-small"
    MULTILINGUAL_E5_BASE = "multilingual-e5-base"
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"
    SNOWFLAKE_ARCTIC_EMBED_L_V2 = "snowflake-arctic-embed-l-v2.0"
    PARAPHRASE_MULTILINGUAL_MINILM_L12_V2 = "paraphrase-multilingual-MiniLM-L12-v2"
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2 = "paraphrase-multilingual-mpnet-base-v2"
    BGE_M3 = "bge-m3"
    GTE_MULTILINGUAL_BASE = "gte-multilingual-base"


class ImageModelType(str, Enum):
    """
    Enumeration of supported image models.
    """

    SIGLIP_BASE_PATCH16_256_MULTILINGUAL = "siglip-base-patch16-256-multilingual"


class ModelInfo(NamedTuple):
    """
    Simple container that maps an enum to:
      - model_id: Hugging Face model ID (or local path)
      - onnx_file: Path to ONNX file (if available)
    """

    model_id: str
    onnx_file: Optional[str] = None


@dataclass
class ModelConfig:
    """
    Configuration for text and image models.
    """

    text_model_type: TextModelType = TextModelType.MULTILINGUAL_E5_SMALL
    image_model_type: ImageModelType = (
        ImageModelType.SIGLIP_BASE_PATCH16_256_MULTILINGUAL
    )

    # If you need extra parameters like `logit_scale`, etc., keep them here
    logit_scale: float = 4.60517

    @property
    def text_model_info(self) -> ModelInfo:
        """
        Return ModelInfo for the configured text_model_type.
        """
        text_configs = {
            TextModelType.MULTILINGUAL_E5_SMALL: ModelInfo(
                model_id="Xenova/multilingual-e5-small",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.MULTILINGUAL_E5_BASE: ModelInfo(
                model_id="Xenova/multilingual-e5-base",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.MULTILINGUAL_E5_LARGE: ModelInfo(
                model_id="Xenova/multilingual-e5-large",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.SNOWFLAKE_ARCTIC_EMBED_L_V2: ModelInfo(
                model_id="Snowflake/snowflake-arctic-embed-l-v2.0",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.PARAPHRASE_MULTILINGUAL_MINILM_L12_V2: ModelInfo(
                model_id="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2: ModelInfo(
                model_id="Xenova/paraphrase-multilingual-mpnet-base-v2",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.BGE_M3: ModelInfo(
                model_id="Xenova/bge-m3",
                onnx_file="onnx/model_quantized.onnx",
            ),
            TextModelType.GTE_MULTILINGUAL_BASE: ModelInfo(
                model_id="onnx-community/gte-multilingual-base",
                onnx_file="onnx/model_quantized.onnx",
            ),
        }
        return text_configs[self.text_model_type]

    @property
    def image_model_info(self) -> ModelInfo:
        """
        Return ModelInfo for the configured image_model_type.
        """
        image_configs = {
            ImageModelType.SIGLIP_BASE_PATCH16_256_MULTILINGUAL: ModelInfo(
                model_id="google/siglip-base-patch16-256-multilingual"
            ),
        }
        return image_configs[self.image_model_type]


class EmbeddingsService:
    """
    Service for generating text/image embeddings and performing ranking.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or ModelConfig()

        # Preloaded text & image models
        self.text_models: Dict[TextModelType, SentenceTransformer] = {}
        self.image_models: Dict[ImageModelType, AutoModel] = {}
        self.image_processors: Dict[ImageModelType, AutoProcessor] = {}

        # Load all models
        self._load_all_models()

    def _load_all_models(self) -> None:
        """
        Pre-load all known text and image models for quick switching.
        """
        try:
            for t_model_type in TextModelType:
                info = ModelConfig(text_model_type=t_model_type).text_model_info
                logger.info("Loading text model: %s", info.model_id)

                # If you have an ONNX file AND your SentenceTransformer supports ONNX
                if info.onnx_file:
                    logger.info("Using ONNX file: %s", info.onnx_file)
                    # The following 'backend' & 'model_kwargs' parameters
                    # are recognized only in special/certain distributions of SentenceTransformer
                    self.text_models[t_model_type] = SentenceTransformer(
                        info.model_id,
                        device=self.device,
                        backend="onnx",  # or "ort" in some custom forks
                        model_kwargs={
                            "provider": "CPUExecutionProvider",  # or "CUDAExecutionProvider"
                            "file_name": info.onnx_file,
                        },
                        trust_remote_code=True,
                    )
                else:
                    # Fallback: standard HF loading
                    self.text_models[t_model_type] = SentenceTransformer(
                        info.model_id, device=self.device, trust_remote_code=True,
                    )

            for i_model_type in ImageModelType:
                model_id = ModelConfig(
                    image_model_type=i_model_type
                ).image_model_info.model_id
                logger.info("Loading image model: %s", model_id)

                # Typically, for CLIP-like models:
                model = AutoModel.from_pretrained(model_id).to(self.device)
                processor = AutoProcessor.from_pretrained(model_id)

                self.image_models[i_model_type] = model
                self.image_processors[i_model_type] = processor

            logger.info("All models loaded successfully.")
        except Exception as e:
            msg = f"Error loading models: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def _validate_text_input(input_text: Union[str, List[str]]) -> List[str]:
        """
        Ensure input_text is a non-empty string or list of strings.
        """
        if isinstance(input_text, str):
            if not input_text.strip():
                raise ValueError("Text input cannot be empty.")
            return [input_text]

        if not isinstance(input_text, list) or not all(
            isinstance(x, str) for x in input_text
        ):
            raise ValueError("Text input must be a string or a list of strings.")

        if len(input_text) == 0:
            raise ValueError("Text input list cannot be empty.")

        return input_text

    @staticmethod
    def _validate_modality(modality: str) -> None:
        if modality not in ("text", "image"):
            raise ValueError("Unsupported modality. Must be 'text' or 'image'.")

    def _process_image(self, path_or_url: Union[str, Path]) -> torch.Tensor:
        """
        Download/Load image from path/URL and apply transformations.
        """
        try:
            if isinstance(path_or_url, Path) or not path_or_url.startswith("http"):
                # Local file path
                img = Image.open(path_or_url).convert("RGB")
            else:
                # URL
                resp = requests.get(path_or_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")

            proc = self.image_processors[self.config.image_model_type]
            data = proc(images=img, return_tensors="pt").to(self.device)
            return data
        except Exception as e:
            raise ValueError(f"Error processing image '{path_or_url}': {str(e)}") from e

    def _generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate text embeddings using the currently configured text model.
        """
        try:
            model = self.text_models[self.config.text_model_type]
            embeddings = model.encode(texts)  # shape: (num_items, emb_dim)
            return embeddings
        except Exception as e:
            raise RuntimeError(
                f"Error generating text embeddings for model '{self.config.text_model_type}': {e}"
            ) from e

    def _generate_image_embeddings(
        self,
        images: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate image embeddings using the currently configured image model.
        If `batch_size` is None, all images are processed at once.
        """
        try:
            model = self.image_models[self.config.image_model_type]

            # Single image
            if isinstance(images, str):
                processed = self._process_image(images)
                with torch.no_grad():
                    emb = model.get_image_features(**processed)
                return emb.cpu().numpy()

            # Multiple images
            if batch_size is None:
                # Process them all in one batch
                tensors = []
                for img_path in images:
                    tensors.append(self._process_image(img_path))
                # Concatenate
                keys = tensors[0].keys()
                combined = {k: torch.cat([t[k] for t in tensors], dim=0) for k in keys}
                with torch.no_grad():
                    emb = model.get_image_features(**combined)
                return emb.cpu().numpy()

            # Process in smaller batches
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                # Process each sub-batch
                tensors = []
                for img_path in batch_images:
                    tensors.append(self._process_image(img_path))
                keys = tensors[0].keys()
                combined = {k: torch.cat([t[k] for t in tensors], dim=0) for k in keys}

                with torch.no_grad():
                    emb = model.get_image_features(**combined)
                all_embeddings.append(emb.cpu().numpy())

            return np.vstack(all_embeddings)

        except Exception as e:
            raise RuntimeError(
                f"Error generating image embeddings for model '{self.config.image_model_type}': {e}"
            ) from e

    async def generate_embeddings(
        self,
        input_data: Union[str, List[str]],
        modality: Literal["text", "image"],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Asynchronously generate embeddings for text or image.
        """
        self._validate_modality(modality)
        if modality == "text":
            text_list = self._validate_text_input(input_data)
            return self._generate_text_embeddings(text_list)
        else:
            return self._generate_image_embeddings(input_data, batch_size=batch_size)

    async def rank(
        self,
        queries: Union[str, List[str]],
        candidates: List[str],
        modality: Literal["text", "image"],
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[List[float]]]:
        """
        Rank candidates (always text) against the queries, which may be text or image.
        Returns dict of { probabilities, cosine_similarities }.
        """
        # 1) Generate embeddings for queries
        query_embeds = await self.generate_embeddings(queries, modality, batch_size)
        # 2) Generate embeddings for text candidates
        candidate_embeds = await self.generate_embeddings(candidates, "text")

        # 3) Compute cosine sim
        sim_matrix = self.cosine_similarity(query_embeds, candidate_embeds)
        # 4) Apply logit scale + softmax
        scaled = np.exp(self.config.logit_scale) * sim_matrix
        probs = self.softmax(scaled)

        return {
            "probabilities": probs.tolist(),
            "cosine_similarities": sim_matrix.tolist(),
        }

    def estimate_tokens(self, input_data: Union[str, List[str]]) -> int:
        """
        Very rough heuristic: ~4 chars per token.
        """
        texts = self._validate_text_input(input_data)
        total_chars = sum(len(t) for t in texts)
        return max(1, round(total_chars / 4))

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Standard softmax along the last dimension.
        """
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        a: (N, D)
        b: (M, D)
        Return: (N, M) of cos sim
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)
