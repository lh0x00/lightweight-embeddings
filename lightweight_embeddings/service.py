from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import List, Union, Dict, Optional, NamedTuple, Any
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from hashlib import md5
from cachetools import LRUCache

import httpx
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
    Container mapping a model type to its model identifier and optional ONNX file.
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
    logit_scale: float = 4.60517  # Example scale used in cross-modal similarity

    @property
    def text_model_info(self) -> ModelInfo:
        """
        Return model information for the configured text model.
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
        Return model information for the configured image model.
        """
        image_configs = {
            ImageModelType.SIGLIP_BASE_PATCH16_256_MULTILINGUAL: ModelInfo(
                model_id="google/siglip-base-patch16-256-multilingual"
            ),
        }
        return image_configs[self.image_model_type]


class ModelKind(str, Enum):
    """
    Indicates the type of model: text or image.
    """

    TEXT = "text"
    IMAGE = "image"


def detect_model_kind(model_id: str) -> ModelKind:
    """
    Detect whether the model identifier corresponds to a text or image model.

    Raises:
        ValueError: If the model identifier is unrecognized.
    """
    if model_id in [m.value for m in TextModelType]:
        return ModelKind.TEXT
    elif model_id in [m.value for m in ImageModelType]:
        return ModelKind.IMAGE
    else:
        raise ValueError(
            f"Unrecognized model ID: {model_id}.\n"
            f"Valid text: {[m.value for m in TextModelType]}\n"
            f"Valid image: {[m.value for m in ImageModelType]}"
        )


class EmbeddingsService:
    """
    Service for generating text/image embeddings and performing similarity ranking.
    Asynchronous methods are used to maximize throughput and avoid blocking the event loop.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the service by setting up model caches, device configuration,
        and asynchronous HTTP client.
        """
        self.lru_cache = LRUCache(maxsize=10_000)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or ModelConfig()

        # Dictionaries to hold preloaded models.
        self.text_models: Dict[TextModelType, SentenceTransformer] = {}
        self.image_models: Dict[ImageModelType, AutoModel] = {}
        self.image_processors: Dict[ImageModelType, AutoProcessor] = {}

        # Create a persistent asynchronous HTTP client.
        self.async_http_client = httpx.AsyncClient(timeout=10)

        # Preload all models.
        self._load_all_models()

    def _load_all_models(self) -> None:
        """
        Pre-load all text and image models to minimize latency at request time.
        """
        try:
            # Preload text models.
            for t_model_type in TextModelType:
                info = ModelConfig(text_model_type=t_model_type).text_model_info
                logger.info("Loading text model: %s", info.model_id)
                if info.onnx_file:
                    logger.info("Using ONNX file: %s", info.onnx_file)
                    self.text_models[t_model_type] = SentenceTransformer(
                        info.model_id,
                        device=self.device,
                        backend="onnx",
                        model_kwargs={
                            "provider": "CPUExecutionProvider",
                            "file_name": info.onnx_file,
                        },
                        trust_remote_code=True,
                    )
                else:
                    self.text_models[t_model_type] = SentenceTransformer(
                        info.model_id,
                        device=self.device,
                        trust_remote_code=True,
                    )

            # Preload image models.
            for i_model_type in ImageModelType:
                model_id = ModelConfig(
                    image_model_type=i_model_type
                ).image_model_info.model_id
                logger.info("Loading image model: %s", model_id)
                model = AutoModel.from_pretrained(model_id).to(self.device)
                model.eval()  # Set the model to evaluation mode.
                processor = AutoProcessor.from_pretrained(model_id)
                self.image_models[i_model_type] = model
                self.image_processors[i_model_type] = processor

            logger.info("All models loaded successfully.")
        except Exception as e:
            msg = f"Error loading models: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def _validate_text_list(input_text: Union[str, List[str]]) -> List[str]:
        """
        Validate and convert text input into a non-empty list of strings.

        Raises:
            ValueError: If the input is invalid.
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
    def _validate_image_list(input_images: Union[str, List[str]]) -> List[str]:
        """
        Validate and convert image input into a non-empty list of image paths/URLs.

        Raises:
            ValueError: If the input is invalid.
        """
        if isinstance(input_images, str):
            if not input_images.strip():
                raise ValueError("Image input cannot be empty.")
            return [input_images]

        if not isinstance(input_images, list) or not all(
            isinstance(x, str) for x in input_images
        ):
            raise ValueError("Image input must be a string or a list of strings.")

        if len(input_images) == 0:
            raise ValueError("Image input list cannot be empty.")

        return input_images

    async def _fetch_image(self, path_or_url: str) -> Image.Image:
        """
        Asynchronously fetch an image from a URL or load from a local path.

        Args:
            path_or_url: The URL or file path of the image.

        Returns:
            A PIL Image in RGB mode.

        Raises:
            ValueError: If image fetching or processing fails.
        """
        try:
            if path_or_url.startswith("http"):
                # Asynchronously fetch the image bytes.
                response = await self.async_http_client.get(path_or_url)
                response.raise_for_status()
                # Offload the blocking I/O (PIL image opening) to a thread.
                img = await asyncio.to_thread(Image.open, BytesIO(response.content))
            else:
                # Offload file I/O to a thread.
                img = await asyncio.to_thread(Image.open, Path(path_or_url))
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error fetching image '{path_or_url}': {str(e)}") from e

    async def _process_image(self, path_or_url: str) -> Dict[str, torch.Tensor]:
        """
        Asynchronously load and process a single image.

        Args:
            path_or_url: The image URL or local path.

        Returns:
            A dictionary of processed tensors ready for model input.

        Raises:
            ValueError: If image processing fails.
        """
        img = await self._fetch_image(path_or_url)
        processor = self.image_processors[self.config.image_model_type]
        # Note: Processor may perform CPU-intensive work; if needed, offload to thread.
        processed_data = processor(images=img, return_tensors="pt").to(self.device)
        return processed_data

    def _generate_text_embeddings(
        self,
        model_id: TextModelType,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generate text embeddings using the SentenceTransformer model.
        Single-text requests are cached using an LRU cache.

        Returns:
            A NumPy array of text embeddings.

        Raises:
            RuntimeError: If text embedding generation fails.
        """
        try:
            if len(texts) == 1:
                single_text = texts[0]
                key = md5(f"{model_id}:{single_text}".encode("utf-8")).hexdigest()[:8]
                if key in self.lru_cache:
                    return self.lru_cache[key]
                model = self.text_models[model_id]
                emb = model.encode([single_text])
                self.lru_cache[key] = emb
                return emb

            model = self.text_models[model_id]
            return model.encode(texts)
        except Exception as e:
            raise RuntimeError(
                f"Error generating text embeddings with model '{model_id}': {e}"
            ) from e

    async def _async_generate_image_embeddings(
        self,
        model_id: ImageModelType,
        images: List[str],
    ) -> np.ndarray:
        """
        Asynchronously generate image embeddings.

        This method concurrently processes multiple images and offloads
        the blocking model inference to a separate thread.

        Returns:
            A NumPy array of image embeddings.

        Raises:
            RuntimeError: If image embedding generation fails.
        """
        try:
            # Concurrently process all images.
            processed_tensors = await asyncio.gather(
                *[self._process_image(img_path) for img_path in images]
            )
            # Assume all processed outputs have the same keys.
            keys = processed_tensors[0].keys()
            combined = {
                k: torch.cat([pt[k] for pt in processed_tensors], dim=0) for k in keys
            }

            def infer():
                with torch.no_grad():
                    embeddings = self.image_models[model_id].get_image_features(
                        **combined
                    )
                return embeddings.cpu().numpy()

            return await asyncio.to_thread(infer)
        except Exception as e:
            raise RuntimeError(
                f"Error generating image embeddings with model '{model_id}': {e}"
            ) from e

    async def generate_embeddings(
        self,
        model: str,
        inputs: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Asynchronously generate embeddings for text or image inputs based on model type.

        Args:
            model: The model identifier.
            inputs: The text or image input(s).

        Returns:
            A NumPy array of embeddings.
        """
        modality = detect_model_kind(model)
        if modality == ModelKind.TEXT:
            text_model_id = TextModelType(model)
            text_list = self._validate_text_list(inputs)
            return await asyncio.to_thread(
                self._generate_text_embeddings, text_model_id, text_list
            )
        elif modality == ModelKind.IMAGE:
            image_model_id = ImageModelType(model)
            image_list = self._validate_image_list(inputs)
            return await self._async_generate_image_embeddings(
                image_model_id, image_list
            )

    async def rank(
        self,
        model: str,
        queries: Union[str, List[str]],
        candidates: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Asynchronously rank candidate texts/images against the provided queries.
        Embeddings for queries and candidates are generated concurrently.

        Returns:
            A dictionary containing probabilities, cosine similarities, and usage statistics.
        """
        modality = detect_model_kind(model)
        if modality == ModelKind.TEXT:
            model_enum = TextModelType(model)
        else:
            model_enum = ImageModelType(model)

        # Concurrently generate embeddings.
        query_task = asyncio.create_task(self.generate_embeddings(model, queries))
        candidate_task = asyncio.create_task(
            self.generate_embeddings(model, candidates)
        )
        query_embeds, candidate_embeds = await asyncio.gather(
            query_task, candidate_task
        )

        # Compute cosine similarity.
        sim_matrix = self.cosine_similarity(query_embeds, candidate_embeds)
        scaled = np.exp(self.config.logit_scale) * sim_matrix
        probs = self.softmax(scaled)

        if modality == ModelKind.TEXT:
            query_tokens = self.estimate_tokens(queries)
            candidate_tokens = self.estimate_tokens(candidates)
            total_tokens = query_tokens + candidate_tokens
        else:
            total_tokens = 0

        usage = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        return {
            "probabilities": probs.tolist(),
            "cosine_similarities": sim_matrix.tolist(),
            "usage": usage,
        }

    def estimate_tokens(self, input_data: Union[str, List[str]]) -> int:
        """
        Estimate the token count for the given text input using the SentenceTransformer tokenizer.

        Returns:
            The total number of tokens.
        """
        texts = self._validate_text_list(input_data)
        model = self.text_models[self.config.text_model_type]
        tokenized = model.tokenize(texts)
        return sum(len(ids) for ids in tokenized["input_ids"])

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Compute the softmax over the last dimension of the input array.

        Returns:
            The softmax probabilities.
        """
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise cosine similarity between all rows of arrays a and b.

        Returns:
            A (N x M) matrix of cosine similarities.
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)

    async def close(self) -> None:
        """
        Close the asynchronous HTTP client.
        """
        await self.async_http_client.aclose()
