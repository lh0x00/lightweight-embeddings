# filename: router.py

"""
FastAPI Router for Embeddings Service

This file exposes the EmbeddingsService functionality via a RESTful API
to generate embeddings and rank candidates.

Supported Text Model IDs:
- "multilingual-e5-small"
- "paraphrase-multilingual-MiniLM-L12-v2"
- "bge-m3"

Supported Image Model ID:
- "google/siglip-base-patch16-256-multilingual"
"""

from __future__ import annotations

import logging
from typing import List, Union
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import ModelConfig, TextModelType, EmbeddingsService

logger = logging.getLogger(__name__)

# Initialize FastAPI router
router = APIRouter(
    tags=["v1"],
    responses={404: {"description": "Not found"}},
)


class ModelType(str, Enum):
    """
    High-level distinction for text vs. image models.
    """

    TEXT = "text"
    IMAGE = "image"


def detect_model_type(model_id: str) -> ModelType:
    """
    Detect whether the provided model ID is for text or image.

    Supported text model IDs:
      - "multilingual-e5-small"
      - "paraphrase-multilingual-MiniLM-L12-v2"
      - "bge-m3"

    Supported image model ID:
      - "google/siglip-base-patch16-256-multilingual"
      (or any model containing "siglip" in its identifier).

    Args:
        model_id: String identifier of the model.

    Returns:
        ModelType.TEXT if it matches one of the recognized text model IDs,
        ModelType.IMAGE if it matches (or contains "siglip").

    Raises:
        ValueError: If the model_id is not recognized as either text or image.
    """
    # Gather all known text model IDs (from TextModelType enum)
    text_model_ids = {m.value for m in TextModelType}

    # Simple check: if it's in text_model_ids, it's text;
    # if 'siglip' is in the model ID, it's recognized as an image model.
    if model_id in text_model_ids:
        return ModelType.TEXT
    elif "siglip" in model_id.lower():
        return ModelType.IMAGE

    error_msg = (
        f"Unsupported model ID: '{model_id}'.\n"
        "Valid text model IDs are: "
        "'multilingual-e5-small', 'paraphrase-multilingual-MiniLM-L12-v2', 'bge-m3'.\n"
        "Valid image model ID contains 'siglip', for example: 'google/siglip-base-patch16-256-multilingual'."
    )
    raise ValueError(error_msg)


# Pydantic Models for request/response
class EmbeddingRequest(BaseModel):
    """
    Request body for embedding creation.

    Model IDs (text):
      - "multilingual-e5-small"
      - "paraphrase-multilingual-MiniLM-L12-v2"
      - "bge-m3"

    Model ID (image):
      - "google/siglip-base-patch16-256-multilingual"
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Model ID to use. Possible text models include: 'multilingual-e5-small', "
            "'paraphrase-multilingual-MiniLM-L12-v2', 'bge-m3'. "
            "For images, you can use: 'google/siglip-base-patch16-256-multilingual' "
            "or any ID containing 'siglip'."
        ),
    )
    input: Union[str, List[str]] = Field(
        ...,
        description=(
            "Input text(s) or image path(s)/URL(s). "
            "Accepts a single string or a list of strings."
        ),
    )


class RankRequest(BaseModel):
    """
    Request body for ranking candidates against queries.

    Model IDs (text):
      - "multilingual-e5-small"
      - "paraphrase-multilingual-MiniLM-L12-v2"
      - "bge-m3"

    Model ID (image):
      - "google/siglip-base-patch16-256-multilingual"
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Model ID to use for the queries. Supported text models: "
            "'multilingual-e5-small', 'paraphrase-multilingual-MiniLM-L12-v2', 'bge-m3'. "
            "For image queries, use an ID containing 'siglip' such as 'google/siglip-base-patch16-256-multilingual'."
        ),
    )
    queries: Union[str, List[str]] = Field(
        ...,
        description=(
            "Query input(s): can be text(s) or image path(s)/URL(s). "
            "If using an image model, ensure your inputs reference valid image paths or URLs."
        ),
    )
    candidates: List[str] = Field(
        ...,
        description=(
            "List of candidate texts to rank against the given queries. "
            "Currently, all candidates must be text."
        ),
    )


class EmbeddingResponse(BaseModel):
    """
    Response structure for embedding creation.
    """

    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


class RankResponse(BaseModel):
    """
    Response structure for ranking results.
    """

    probabilities: List[List[float]]
    cosine_similarities: List[List[float]]


# Initialize the service with default configuration
service_config = ModelConfig()
embeddings_service = EmbeddingsService(config=service_config)


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for the provided input text(s) or image(s).

    Supported Model IDs for text:
      - "multilingual-e5-small"
      - "paraphrase-multilingual-MiniLM-L12-v2"
      - "bge-m3"

    Supported Model ID for image:
      - "google/siglip-base-patch16-256-multilingual"

    Steps:
      1. Detects model type (text or image) based on the model ID.
      2. Adjusts the service configuration accordingly.
      3. Produces embeddings via the EmbeddingsService.
      4. Returns embedding vectors along with usage information.

    Raises:
      HTTPException: For any errors during model detection or embedding generation.
    """
    try:
        modality = detect_model_type(request.model)

        # Adjust global config based on the detected modality
        if modality == ModelType.TEXT:
            service_config.text_model_type = TextModelType(request.model)
        else:
            service_config.image_model_id = request.model

        # Generate embeddings asynchronously
        embeddings = await embeddings_service.generate_embeddings(
            input_data=request.input, modality=modality.value
        )

        # Estimate tokens only if it's text
        total_tokens = 0
        if modality == ModelType.TEXT:
            total_tokens = embeddings_service.estimate_tokens(request.input)

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": emb.tolist(),
                }
                for idx, emb in enumerate(embeddings)
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    except Exception as e:
        error_msg = (
            "Failed to generate embeddings. Please verify your model ID, input data, and server logs.\n"
            f"Error Details: {str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/rank", response_model=RankResponse, tags=["rank"])
async def rank_candidates(request: RankRequest):
    """
    Rank the given candidate texts against the provided queries.

    Supported Model IDs for text queries:
      - "multilingual-e5-small"
      - "paraphrase-multilingual-MiniLM-L12-v2"
      - "bge-m3"

    Supported Model ID for image queries:
      - "google/siglip-base-patch16-256-multilingual"

    Steps:
      1. Detects model type (text or image) based on the query model ID.
      2. Adjusts the service configuration accordingly.
      3. Generates embeddings for the queries (text or image).
      4. Generates embeddings for the candidates (always text).
      5. Computes cosine similarities and returns softmax-normalized probabilities.

    Raises:
      HTTPException: For any errors during model detection or ranking.
    """
    try:
        modality = detect_model_type(request.model)

        # Adjust global config based on the detected modality
        if modality == ModelType.TEXT:
            service_config.text_model_type = TextModelType(request.model)
        else:
            service_config.image_model_id = request.model

        # Perform the ranking
        results = await embeddings_service.rank(
            queries=request.queries,
            candidates=request.candidates,
            modality=modality.value,
        )
        return results

    except Exception as e:
        error_msg = (
            "Failed to rank candidates. Please verify your model ID, input data, and server logs.\n"
            f"Error Details: {str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
