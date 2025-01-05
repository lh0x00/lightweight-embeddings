from __future__ import annotations

import logging
from enum import Enum
from typing import List, Union, Dict, Optional, NamedTuple, Any
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from hashlib import md5
from cachetools import LRUCache

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
    This container maps an enum to:
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
    logit_scale: float = 4.60517  # Example scale used in cross-modal similarity

    @property
    def text_model_info(self) -> ModelInfo:
        """
        Returns ModelInfo for the configured text_model_type.
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
        Returns ModelInfo for the configured image_model_type.
        """
        image_configs = {
            ImageModelType.SIGLIP_BASE_PATCH16_256_MULTILINGUAL: ModelInfo(
                model_id="google/siglip-base-patch16-256-multilingual"
            ),
        }
        return image_configs[self.image_model_type]


class ModelKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"


def detect_model_kind(model_id: str) -> ModelKind:
    """
    Detect whether model_id belongs to a text or an image model.
    Raises ValueError if the model is not recognized.
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
    Batch size has been removed. Single or multiple inputs are handled uniformly.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.lru_cache = LRUCache(maxsize=10_000)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or ModelConfig()

        # Dictionaries to hold preloaded models
        self.text_models: Dict[TextModelType, SentenceTransformer] = {}
        self.image_models: Dict[ImageModelType, AutoModel] = {}
        self.image_processors: Dict[ImageModelType, AutoProcessor] = {}

        # Load all relevant models on init
        self._load_all_models()

    def _load_all_models(self) -> None:
        """
        Pre-load all known text and image models for quick switching.
        """
        try:
            # Preload text models
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

            # Preload image models
            for i_model_type in ImageModelType:
                model_id = ModelConfig(
                    image_model_type=i_model_type
                ).image_model_info.model_id
                logger.info("Loading image model: %s", model_id)

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
    def _validate_text_list(input_text: Union[str, List[str]]) -> List[str]:
        """
        Convert text input into a non-empty list of strings.
        Raises ValueError if the input is invalid.
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
        Convert image input into a non-empty list of image paths/URLs.
        Raises ValueError if the input is invalid.
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

    def _process_image(self, path_or_url: str) -> Dict[str, torch.Tensor]:
        """
        Loads and processes a single image from local path or URL.
        Returns a dictionary of tensors ready for the model.
        """
        try:
            if path_or_url.startswith("http"):
                resp = requests.get(path_or_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(Path(path_or_url)).convert("RGB")

            processor = self.image_processors[self.config.image_model_type]
            processed_data = processor(images=img, return_tensors="pt").to(self.device)
            return processed_data
        except Exception as e:
            raise ValueError(f"Error processing image '{path_or_url}': {str(e)}") from e

    def _generate_text_embeddings(
        self,
        model_id: TextModelType,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generates text embeddings using the SentenceTransformer-based model.
        Utilizes an LRU cache for single-input scenarios.
        """
        try:
            if len(texts) == 1:
                single_text = texts[0]
                key = md5(single_text.encode("utf-8")).hexdigest()
                if key in self.lru_cache:
                    return self.lru_cache[key]

                model = self.text_models[model_id]
                emb = model.encode([single_text])
                self.lru_cache[key] = emb
                return emb

            # For multiple texts, no LRU cache is used
            model = self.text_models[model_id]
            return model.encode(texts)

        except Exception as e:
            raise RuntimeError(
                f"Error generating text embeddings with model '{model_id}': {e}"
            ) from e

    def _generate_image_embeddings(
        self,
        model_id: ImageModelType,
        images: List[str],
    ) -> np.ndarray:
        """
        Generates image embeddings using the CLIP-like transformer model.
        Handles single or multiple images uniformly (no batch size parameter).
        """
        try:
            model = self.image_models[model_id]
            # Collect processed inputs in a single batch
            processed_tensors = []
            for img_path in images:
                processed_tensors.append(self._process_image(img_path))

            # Keys should be the same for all processed outputs
            keys = processed_tensors[0].keys()
            # Concatenate along the batch dimension
            combined = {
                k: torch.cat([pt[k] for pt in processed_tensors], dim=0) for k in keys
            }

            with torch.no_grad():
                embeddings = model.get_image_features(**combined)
            return embeddings.cpu().numpy()

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
        Asynchronously generates embeddings for either text or image based on the model type.
        """
        modality = detect_model_kind(model)

        if modality == ModelKind.TEXT:
            text_model_id = TextModelType(model)
            text_list = self._validate_text_list(inputs)
            return self._generate_text_embeddings(text_model_id, text_list)

        elif modality == ModelKind.IMAGE:
            image_model_id = ImageModelType(model)
            image_list = self._validate_image_list(inputs)
            return self._generate_image_embeddings(image_model_id, image_list)

    async def rank(
        self,
        model: str,
        queries: Union[str, List[str]],
        candidates: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Ranks text `candidates` given `queries`, which can be text or images.
        Always returns a dictionary of { probabilities, cosine_similarities, usage }.

        Note: This implementation uses the same model for both queries and candidates.
              For true cross-modal ranking, you might need separate models or a shared model.
        """
        modality = detect_model_kind(model)

        # Convert the string model to the appropriate enum
        if modality == ModelKind.TEXT:
            model_enum = TextModelType(model)
        else:
            model_enum = ImageModelType(model)

        # 1) Generate embeddings for queries
        query_embeds = await self.generate_embeddings(model_enum.value, queries)

        # 2) Generate embeddings for candidates (assumed text if queries are text;
        #    or if queries are images, also use the image model for candidates).
        candidate_embeds = await self.generate_embeddings(model_enum.value, candidates)

        # 3) Compute cosine similarity
        sim_matrix = self.cosine_similarity(query_embeds, candidate_embeds)

        # 4) Apply logit scale + softmax to obtain probabilities
        scaled = np.exp(self.config.logit_scale) * sim_matrix
        probs = self.softmax(scaled)

        # 5) Estimate token usage if we're dealing with text
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
        Estimates token count using the SentenceTransformer tokenizer.
        Only applicable if the current configured model is a text model.
        """
        texts = self._validate_text_list(input_data)
        model = self.text_models[self.config.text_model_type]
        tokenized = model.tokenize(texts)
        # Summing over the lengths of input_ids for each example
        return sum(len(ids) for ids in tokenized["input_ids"])

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Applies the standard softmax function along the last dimension.
        """
        # Stabilize scores by subtracting max
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise cosine similarity between all rows of a and b.
        a: (N, D)
        b: (M, D)
        Return: (N, M) matrix of cosine similarities
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)
