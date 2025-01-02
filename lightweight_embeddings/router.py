"""
FastAPI Router for Embeddings Service (Revised & Simplified)

Exposes the EmbeddingsService methods via a RESTful API.

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
- "siglip-base-patch16-256-multilingual"
"""

from __future__ import annotations

import logging
from typing import List, Union
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import (
    ModelConfig,
    TextModelType,
    ImageModelType,
    EmbeddingsService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["v1"],
    responses={404: {"description": "Not found"}},
)


class ModelKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"


def detect_model_kind(model_id: str) -> ModelKind:
    """
    Detect whether model_id is for a text or an image model.
    Raises ValueError if unrecognized.
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


class EmbeddingRequest(BaseModel):
    """
    Input to /v1/embeddings
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Which model ID to use? "
            "Text: ['multilingual-e5-small', 'multilingual-e5-base', 'multilingual-e5-large', 'snowflake-arctic-embed-l-v2.0', 'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-multilingual-mpnet-base-v2', 'bge-m3']. "
            "Image: ['siglip-base-patch16-256-multilingual']."
        ),
    )
    input: Union[str, List[str]] = Field(
        ..., description="Text(s) or Image URL(s)/path(s)."
    )


class RankRequest(BaseModel):
    """
    Input to /v1/rank
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Model ID for the queries. "
            "Text or Image model, e.g. 'siglip-base-patch16-256-multilingual' for images."
        ),
    )
    queries: Union[str, List[str]] = Field(
        ..., description="Query text or image(s) depending on the model type."
    )
    candidates: List[str] = Field(
        ..., description="Candidate texts to rank. Must be text."
    )


class EmbeddingResponse(BaseModel):
    """
    Response of /v1/embeddings
    """

    object: str
    data: List[dict]
    model: str
    usage: dict


class RankResponse(BaseModel):
    """
    Response of /v1/rank
    """

    probabilities: List[List[float]]
    cosine_similarities: List[List[float]]

service_config = ModelConfig()
embeddings_service = EmbeddingsService(config=service_config)


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for the given input (text or image).
    """
    try:
        # 1) Determine if it's text or image
        mkind = detect_model_kind(request.model)

        # 2) Update global service config so it uses the correct model
        if mkind == ModelKind.TEXT:
            service_config.text_model_type = TextModelType(request.model)
        else:
            service_config.image_model_type = ImageModelType(request.model)

        # 3) Generate
        embeddings = await embeddings_service.generate_embeddings(
            input_data=request.input, modality=mkind.value
        )

        # 4) Estimate tokens for text only
        total_tokens = 0
        if mkind == ModelKind.TEXT:
            total_tokens = embeddings_service.estimate_tokens(request.input)

        resp = {
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
        for idx, emb in enumerate(embeddings):
            resp["data"].append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": emb.tolist(),
                }
            )

        return resp

    except Exception as e:
        msg = (
            "Failed to generate embeddings. Check model ID, inputs, etc.\n"
            f"Details: {str(e)}"
        )
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)


@router.post("/rank", response_model=RankResponse, tags=["rank"])
async def rank_candidates(request: RankRequest):
    """
    Ranks candidate texts against the given queries (which can be text or image).
    """
    try:
        mkind = detect_model_kind(request.model)

        if mkind == ModelKind.TEXT:
            service_config.text_model_type = TextModelType(request.model)
        else:
            service_config.image_model_type = ImageModelType(request.model)

        results = await embeddings_service.rank(
            queries=request.queries,
            candidates=request.candidates,
            modality=mkind.value,
        )
        return results

    except Exception as e:
        msg = (
            "Failed to rank candidates. Check model ID, inputs, etc.\n"
            f"Details: {str(e)}"
        )
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)
