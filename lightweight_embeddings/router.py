from __future__ import annotations

import logging
import os
from typing import Dict, List, Union
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .analytics import Analytics
from .service import (
    ModelConfig,
    TextModelType,
    EmbeddingsService,
    ModelKind,
    detect_model_kind,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["v1"],
    responses={404: {"description": "Not found"}},
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


class StatsBucket(BaseModel):
    """Helper model for daily/weekly/monthly/yearly stats"""

    total: Dict[str, int]
    daily: Dict[str, int]
    weekly: Dict[str, int]
    monthly: Dict[str, int]
    yearly: Dict[str, int]


class StatsResponse(BaseModel):
    """Analytics stats response model, including both access and token counts"""

    access: StatsBucket
    tokens: StatsBucket


service_config = ModelConfig()
embeddings_service = EmbeddingsService(config=service_config)

analytics = Analytics(
    url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    token=os.environ.get("REDIS_TOKEN", "***"),
    sync_interval=30 * 60,  # 30 minutes
)


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(
    request: EmbeddingRequest, background_tasks: BackgroundTasks
):
    """
    Generates embeddings for the given input (text or image).
    """
    try:
        modality = detect_model_kind(request.model)
        embeddings = await embeddings_service.generate_embeddings(
            inputs=request.input,
            model=request.model,
        )

        # Estimate tokens for text only
        total_tokens = 0
        if modality == ModelKind.TEXT:
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

        background_tasks.add_task(
            analytics.access, request.model, resp["usage"]["total_tokens"]
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
async def rank_candidates(request: RankRequest, background_tasks: BackgroundTasks):
    """
    Ranks candidate texts against the given queries (which can be text or image).
    """
    try:
        results = await embeddings_service.rank(
            model=request.model,
            queries=request.queries,
            candidates=request.candidates,
        )

        background_tasks.add_task(
            analytics.access, request.model, results["usage"]["total_tokens"]
        )

        return results

    except Exception as e:
        msg = (
            "Failed to rank candidates. Check model ID, inputs, etc.\n"
            f"Details: {str(e)}"
        )
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)


@router.get("/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats():
    """Get usage statistics for all models, including access and tokens."""
    try:
        day_key = datetime.utcnow().strftime("%Y-%m-%d")
        week_key = f"{datetime.utcnow().year}-W{datetime.utcnow().strftime('%U')}"
        month_key = datetime.utcnow().strftime("%Y-%m")
        year_key = datetime.utcnow().strftime("%Y")

        stats_data = await analytics.stats()  # { "access": {...}, "tokens": {...} }

        return {
            "access": {
                "total": stats_data["access"].get("total", {}),
                "daily": stats_data["access"].get(day_key, {}),
                "weekly": stats_data["access"].get(week_key, {}),
                "monthly": stats_data["access"].get(month_key, {}),
                "yearly": stats_data["access"].get(year_key, {}),
            },
            "tokens": {
                "total": stats_data["tokens"].get("total", {}),
                "daily": stats_data["tokens"].get(day_key, {}),
                "weekly": stats_data["tokens"].get(week_key, {}),
                "monthly": stats_data["tokens"].get(month_key, {}),
                "yearly": stats_data["tokens"].get(year_key, {}),
            },
        }
    except Exception as e:
        msg = f"Failed to fetch analytics stats: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)
