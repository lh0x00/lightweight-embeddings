from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Header, Request
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
    Request model for generating embeddings.
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Which model ID to use? "
            "Text options: ['multilingual-e5-small', 'multilingual-e5-base', 'multilingual-e5-large', "
            "'snowflake-arctic-embed-l-v2.0', 'paraphrase-multilingual-MiniLM-L12-v2', "
            "'paraphrase-multilingual-mpnet-base-v2', 'bge-m3', 'gte-multilingual-base', 'embeddinggemma-300m']. "
            "Image option: ['siglip-base-patch16-256-multilingual']."
        ),
    )
    input: Union[str, List[str]] = Field(
        ..., description="Text(s) or image URL(s)/path(s)."
    )


class RankRequest(BaseModel):
    """
    Request model for ranking candidates.
    """

    model: str = Field(
        default=TextModelType.MULTILINGUAL_E5_SMALL.value,
        description=(
            "Model ID for the queries. "
            "Can be a text or image model (e.g. 'siglip-base-patch16-256-multilingual' for images)."
        ),
    )
    queries: Union[str, List[str]] = Field(
        ..., description="Query text(s) or image(s) depending on the model type."
    )
    candidates: List[str] = Field(..., description="Candidate texts to rank.")


class EmbeddingResponse(BaseModel):
    """
    Response model for embeddings.
    """

    object: str
    data: List[dict]
    model: str
    usage: dict


class RankResponse(BaseModel):
    """
    Response model for ranking results.
    """

    probabilities: List[List[float]]
    cosine_similarities: List[List[float]]


class StatsBucket(BaseModel):
    """
    Model for daily/weekly/monthly/yearly stats.
    """

    total: Dict[str, int]
    daily: Dict[str, int]
    weekly: Dict[str, int]
    monthly: Dict[str, int]
    yearly: Dict[str, int]


class StatsResponse(BaseModel):
    """
    Analytics stats response model, including both access and token counts.
    """

    access: StatsBucket
    tokens: StatsBucket


# Initialize the embeddings service and analytics.
service_config = ModelConfig()
embeddings_service = EmbeddingsService(config=service_config)

analytics = Analytics(
    url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    token=os.environ.get("REDIS_TOKEN", "***"),
    sync_interval=30 * 60,  # 30 minutes
)

# Rate limiting cache: {ip: [timestamp1, timestamp2, ...]}
rate_limit_cache: Dict[str, List[float]] = {}


def check_rate_limit(
    client_ip: str, max_requests: int = 10, window_seconds: int = 60
) -> bool:
    """
    Check if the client IP has exceeded the rate limit.
    Returns True if request is allowed, False if rate limited.
    """
    current_time = time.time()

    # Clean up old entries and get current requests
    if client_ip in rate_limit_cache:
        # Remove requests older than the window
        rate_limit_cache[client_ip] = [
            timestamp
            for timestamp in rate_limit_cache[client_ip]
            if current_time - timestamp < window_seconds
        ]
    else:
        rate_limit_cache[client_ip] = []

    # Check if under limit
    if len(rate_limit_cache[client_ip]) < max_requests:
        # Add current request timestamp
        rate_limit_cache[client_ip].append(current_time)
        return True

    return False


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    authorization: str = Header(None),
):
    """
    Generate embeddings for the given text or image inputs.
    """
    # Check authorization
    expected_token = os.environ.get("ACCESS_TOKEN")
    is_authenticated = False

    if expected_token and authorization:
        # Support both "Bearer <token>" and plain token formats
        token = authorization
        if authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix

        if token == expected_token:
            is_authenticated = True

    # If not authenticated (no token, empty token, or wrong token), apply rate limit
    if not is_authenticated:
        # Get client IP
        client_ip = fastapi_request.client.host
        if hasattr(fastapi_request.headers, "get"):
            # Check for forwarded IP (in case of proxy)
            forwarded_for = fastapi_request.headers.get("X-Forwarded-For")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()

            real_ip = fastapi_request.headers.get("X-Real-IP")
            if real_ip:
                client_ip = real_ip.strip()

        # Check rate limit (10 requests per minute for unauthenticated users)
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 10 requests per minute for unauthenticated users.",
            )

    try:
        modality = detect_model_kind(request.model)
        embeddings = await embeddings_service.generate_embeddings(
            model=request.model,
            inputs=request.input,
        )

        # Estimate tokens if using a text model.
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

        # Record analytics in the background.
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
async def rank_candidates(
    request: RankRequest, 
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    authorization: str = Header(None),
):
    """
    Rank candidate texts against the given queries.
    """
    # Check authorization
    expected_token = os.environ.get("ACCESS_TOKEN")
    is_authenticated = False

    if expected_token and authorization:
        # Support both "Bearer <token>" and plain token formats
        token = authorization
        if authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix

        if token == expected_token:
            is_authenticated = True

    # If not authenticated (no token, empty token, or wrong token), apply rate limit
    if not is_authenticated:
        # Get client IP
        client_ip = fastapi_request.client.host
        if hasattr(fastapi_request.headers, "get"):
            # Check for forwarded IP (in case of proxy)
            forwarded_for = fastapi_request.headers.get("X-Forwarded-For")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()

            real_ip = fastapi_request.headers.get("X-Real-IP")
            if real_ip:
                client_ip = real_ip.strip()

        # Check rate limit (10 requests per minute for unauthenticated users)
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 10 requests per minute for unauthenticated users.",
            )

    try:
        results = await embeddings_service.rank(
            model=request.model,
            queries=request.queries,
            candidates=request.candidates,
        )

        # Record analytics in the background.
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
    """
    Retrieve usage statistics for all models, including access counts and token usage.
    """
    try:
        day_key = datetime.utcnow().strftime("%Y-%m-%d")
        week_key = f"{datetime.utcnow().year}-W{datetime.utcnow().strftime('%U')}"
        month_key = datetime.utcnow().strftime("%Y-%m")
        year_key = datetime.utcnow().strftime("%Y")

        stats_data = (
            await analytics.stats()
        )  # Expected to return a dict with 'access' and 'tokens' keys

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
