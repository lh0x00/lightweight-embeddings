# filename: service.py

"""
Lightweight Embeddings Service Module

This module provides a service for generating and comparing embeddings from text and images
using state-of-the-art transformer models. It supports both CPU and GPU inference.

Key Features:
- Text and image embedding generation
- Cross-modal similarity ranking
- Batch processing support
- Asynchronous API support

Supported Text Model IDs:
- "multilingual-e5-small"
- "paraphrase-multilingual-MiniLM-L12-v2"
- "bge-m3"

Supported Image Model ID (default):
- "google/siglip-base-patch16-256-multilingual"
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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default Model IDs
TEXT_MODEL_ID = "Xenova/multilingual-e5-small"
IMAGE_MODEL_ID = "google/siglip-base-patch16-256-multilingual"


class TextModelType(str, Enum):
    """
    Enumeration of supported text models.
    Please ensure the ONNX files and Hugging Face model IDs are consistent
    with your local or remote environment.
    """

    MULTILINGUAL_E5_SMALL = "multilingual-e5-small"
    PARAPHRASE_MULTILINGUAL_MINILM_L12_V2 = "paraphrase-multilingual-MiniLM-L12-v2"
    BGE_M3 = "bge-m3"


class ModelInfo(NamedTuple):
    """
    Simple container for mapping a given text model type
    to its Hugging Face model repository and the local ONNX file path.
    """

    model_id: str
    onnx_file: str


@dataclass
class ModelConfig:
    """
    Configuration settings for model providers, backends, and defaults.
    """

    provider: str = "CPUExecutionProvider"
    backend: str = "onnx"
    logit_scale: float = 4.60517
    text_model_type: TextModelType = TextModelType.MULTILINGUAL_E5_SMALL
    image_model_id: str = IMAGE_MODEL_ID

    @property
    def text_model_info(self) -> ModelInfo:
        """
        Retrieves the ModelInfo for the currently selected text_model_type.
        """
        model_configs = {
            TextModelType.MULTILINGUAL_E5_SMALL: ModelInfo(
                "Xenova/multilingual-e5-small",
                "onnx/model_quantized.onnx",
            ),
            TextModelType.PARAPHRASE_MULTILINGUAL_MINILM_L12_V2: ModelInfo(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "onnx/model_quint8_avx2.onnx",
            ),
            TextModelType.BGE_M3: ModelInfo(
                "BAAI/bge-m3",
                "model.onnx",
            ),
        }
        return model_configs[self.text_model_type]


class EmbeddingsService:
    """
    Service for generating and comparing text/image embeddings.

    This service supports multiple text models and a single image model.
    It provides methods for:
        - Generating text embeddings
        - Generating image embeddings
        - Ranking candidates by similarity
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize the EmbeddingsService.

        Args:
            config: Optional ModelConfig object to override default settings.
        """
        # Determine whether GPU (CUDA) is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use the provided config or fall back to defaults
        self.config = config or ModelConfig()

        # Dictionary to hold multiple text models
        self.text_models: Dict[TextModelType, SentenceTransformer] = {}

        # Load all models (text + image) into memory
        self._load_models()

    def _load_models(self) -> None:
        """
        Load text and image models into memory.

        This pre-loads all text models defined in the TextModelType enum
        and a single image model, enabling quick switching at runtime.
        """
        try:
            # Load all text models
            for model_type in TextModelType:
                model_info = ModelConfig(text_model_type=model_type).text_model_info
                logger.info(f"Loading text model: {model_info.model_id}")

                self.text_models[model_type] = SentenceTransformer(
                    model_info.model_id,
                    device=self.device,
                    backend=self.config.backend,
                    model_kwargs={
                        "provider": self.config.provider,
                        "file_name": model_info.onnx_file,
                    },
                )

            logger.info(f"Loading image model: {self.config.image_model_id}")
            self.image_model = AutoModel.from_pretrained(self.config.image_model_id).to(
                self.device
            )
            self.image_processor = AutoProcessor.from_pretrained(
                self.config.image_model_id
            )

            logger.info(f"All models loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(
                "Model loading failed. Please ensure you have valid model IDs and local files.\n"
                f"Error details: {str(e)}"
            )
            raise RuntimeError(f"Failed to load models: {str(e)}") from e

    @staticmethod
    def _validate_text_input(input_text: Union[str, List[str]]) -> List[str]:
        """
        Validate and standardize the input for text embeddings.

        Args:
            input_text: Either a single string or a list of strings.

        Returns:
            A list of strings to process.

        Raises:
            ValueError: If input_text is empty or not string-based.
        """
        if isinstance(input_text, str):
            return [input_text]
        if not isinstance(input_text, list) or not all(
            isinstance(x, str) for x in input_text
        ):
            raise ValueError(
                "Text input must be a single string or a list of strings. "
                "Found a different data type instead."
            )
        if not input_text:
            raise ValueError("Text input list cannot be empty.")
        return input_text

    @staticmethod
    def _validate_modality(modality: str) -> None:
        """
        Validate the input modality.

        Args:
            modality: Must be either 'text' or 'image'.

        Raises:
            ValueError: If modality is neither 'text' nor 'image'.
        """
        if modality not in ["text", "image"]:
            raise ValueError(
                "Invalid modality. Please specify 'text' or 'image' for embeddings."
            )

    def _process_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess an image from either a local path or a URL.

        Args:
            image_path: Path to the local image file or a URL.

        Returns:
            Torch Tensor suitable for model input.

        Raises:
            ValueError: If the image file or URL cannot be loaded.
        """
        try:
            if str(image_path).startswith("http"):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image_content = BytesIO(response.content)
            else:
                image_content = image_path

            image = Image.open(image_content).convert("RGB")
            processed = self.image_processor(images=image, return_tensors="pt").to(
                self.device
            )
            return processed

        except Exception as e:
            raise ValueError(
                f"Failed to process image at '{image_path}'. Check the path/URL and file format.\n"
                f"Details: {str(e)}"
            ) from e

    def _generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Helper method to generate text embeddings for a list of texts
        using the currently configured text model.

        Args:
            texts: A list of text strings.

        Returns:
            Numpy array of shape (num_texts, embedding_dim).

        Raises:
            RuntimeError: If the text model fails to generate embeddings.
        """
        try:
            logger.info(
                f"Generating embeddings for {len(texts)} text items using model: "
                f"{self.config.text_model_type}"
            )
            # Select the preloaded text model based on the current config
            model = self.text_models[self.config.text_model_type]
            embeddings = model.encode(texts)
            return embeddings
        except Exception as e:
            error_msg = (
                f"Error generating text embeddings with model: {self.config.text_model_type}. "
                f"Details: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _generate_image_embeddings(
        self, input_data: Union[str, List[str]], batch_size: Optional[int]
    ) -> np.ndarray:
        """
        Helper method to generate image embeddings.

        Args:
            input_data: Either a single image path/URL or a list of them.
            batch_size: Batch size for processing images in chunks.
                        If None, process all at once.

        Returns:
            Numpy array of shape (num_images, embedding_dim).

        Raises:
            RuntimeError: If the image model fails to generate embeddings.
        """
        try:
            if isinstance(input_data, str):
                # Single image scenario
                processed = self._process_image(input_data)
                with torch.no_grad():
                    embedding = self.image_model.get_image_features(**processed)
                return embedding.cpu().numpy()

            # Multiple images scenario
            logger.info(f"Generating embeddings for {len(input_data)} images.")
            if batch_size is None:
                # Process all images at once
                processed_batches = [
                    self._process_image(img_path) for img_path in input_data
                ]
                with torch.no_grad():
                    # Concatenate all images along the batch dimension
                    batch_keys = processed_batches[0].keys()
                    concatenated = {
                        k: torch.cat([pb[k] for pb in processed_batches], dim=0)
                        for k in batch_keys
                    }
                    embedding = self.image_model.get_image_features(**concatenated)
                return embedding.cpu().numpy()

            # Process images in smaller batches
            embeddings_list = []
            for i, img_path in enumerate(input_data):
                if i % batch_size == 0:
                    logger.debug(
                        f"Processing image batch {i // batch_size + 1} with size up to {batch_size}."
                    )
                processed = self._process_image(img_path)
                with torch.no_grad():
                    embedding = self.image_model.get_image_features(**processed)
                embeddings_list.append(embedding.cpu().numpy())

            return np.vstack(embeddings_list)

        except Exception as e:
            error_msg = (
                f"Error generating image embeddings with model: {self.config.image_model_id}. "
                f"Details: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def generate_embeddings(
        self,
        input_data: Union[str, List[str]],
        modality: Literal["text", "image"] = "text",
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Asynchronously generate embeddings for text or image inputs.

        Args:
            input_data: A string or list of strings (text/image paths/URLs).
            modality: "text" for text data or "image" for image data.
            batch_size: Optional batch size for processing images in chunks.

        Returns:
            Numpy array of embeddings.

        Raises:
            ValueError: If the modality is invalid.
        """
        self._validate_modality(modality)

        if modality == "text":
            texts = self._validate_text_input(input_data)
            return self._generate_text_embeddings(texts)
        else:
            return self._generate_image_embeddings(input_data, batch_size)

    async def rank(
        self,
        queries: Union[str, List[str]],
        candidates: List[str],
        modality: Literal["text", "image"] = "text",
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[List[float]]]:
        """
        Rank a set of candidate texts against one or more queries using cosine similarity
        and a softmax to produce probability-like scores.

        Args:
            queries: Query text(s) or image path(s)/URL(s).
            candidates: Candidate texts to be ranked.
                        (Note: This implementation always treats candidates as text.)
            modality: "text" for text queries or "image" for image queries.
            batch_size: Batch size if images are processed in chunks.

        Returns:
            Dictionary containing:
                - "probabilities": 2D list of softmax-normalized scores.
                - "cosine_similarities": 2D list of raw cosine similarity values.

        Raises:
            RuntimeError: If the query or candidate embeddings fail to generate.
        """
        logger.info(
            f"Ranking {len(candidates)} candidates against "
            f"{len(queries) if isinstance(queries, list) else 1} query item(s)."
        )

        # Generate embeddings for queries
        query_embeds = await self.generate_embeddings(
            queries, modality=modality, batch_size=batch_size
        )

        # Generate embeddings for candidates (always text)
        candidate_embeds = await self.generate_embeddings(
            candidates, modality="text", batch_size=batch_size
        )

        # Compute cosine similarity and scaled probabilities
        cosine_sims = self.cosine_similarity(query_embeds, candidate_embeds)
        logit_scale = np.exp(self.config.logit_scale)
        probabilities = self.softmax(logit_scale * cosine_sims)

        return {
            "probabilities": probabilities.tolist(),
            "cosine_similarities": cosine_sims.tolist(),
        }

    def estimate_tokens(self, input_data: Union[str, List[str]]) -> int:
        """
        Roughly estimate the total number of tokens in the given text(s).

        Args:
            input_data: A string or list of strings representing text input.

        Returns:
            Estimated token count (int).

        Raises:
            ValueError: If the input is not valid text data.
        """
        texts = self._validate_text_input(input_data)
        # Very rough approximation: assume ~4 characters per token
        total_chars = sum(len(t) for t in texts)
        return max(1, round(total_chars / 4))

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Apply softmax along the last dimension of the scores array.

        Args:
            scores: Numpy array of shape (..., num_candidates).

        Returns:
            Numpy array of softmax-normalized values, same shape as scores.
        """
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    @staticmethod
    def cosine_similarity(
        query_embeds: np.ndarray, candidate_embeds: np.ndarray
    ) -> np.ndarray:
        """
        Compute the cosine similarity between two sets of vectors.

        Args:
            query_embeds: Numpy array of shape (num_queries, embed_dim).
            candidate_embeds: Numpy array of shape (num_candidates, embed_dim).

        Returns:
            2D Numpy array of shape (num_queries, num_candidates)
            containing cosine similarity scores.
        """
        # Normalize embeddings
        query_norm = query_embeds / np.linalg.norm(query_embeds, axis=1, keepdims=True)
        candidate_norm = candidate_embeds / np.linalg.norm(
            candidate_embeds, axis=1, keepdims=True
        )
        return np.dot(query_norm, candidate_norm.T)
