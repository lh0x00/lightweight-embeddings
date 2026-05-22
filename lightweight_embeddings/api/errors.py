"""Centralized exception handlers.

Goals:

* Never leak ``str(exception)`` directly to clients (the previous code
  echoed full Python tracebacks via the FastAPI 500 detail).
* Always include a ``request_id`` in the response body so support can
  correlate with logs.
* Return shape is OpenAI-compatible so existing client SDKs work:
  ``{"error": {"message": ..., "type": ..., "code": ...}}``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse

from ..core.registry import UnknownModelError
from ..core.service import ImageFetchError
from ..security.concurrency import ConcurrencyError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None) or request.headers.get("x-request-id")


def _error_body(
    *,
    message: str,
    code: str,
    type_: str,
    request_id: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"message": message, "type": type_, "code": code}}
    if request_id:
        body["error"]["request_id"] = request_id
    if extra:
        body["error"].update(extra)
    return body


def install_handlers(app: FastAPI) -> None:
    """Register all error handlers on ``app``."""

    @app.exception_handler(HTTPException)
    async def _http_handler(request: Request, exc: HTTPException) -> ORJSONResponse:
        rid = _request_id(request)
        # Preserve any caller-provided headers (e.g. Retry-After from the
        # rate limiter) but always set Content-Type via ORJSONResponse.
        return ORJSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                message=str(exc.detail) if not isinstance(exc.detail, dict) else "request failed",
                code=str(exc.status_code),
                type_=_status_to_type(exc.status_code),
                request_id=rid,
                extra=exc.detail if isinstance(exc.detail, dict) else None,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> ORJSONResponse:
        rid = _request_id(request)
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                message="invalid request",
                code="invalid_request",
                type_="invalid_request_error",
                request_id=rid,
                extra={"details": exc.errors()},
            ),
        )

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(request: Request, exc: UnknownModelError) -> ORJSONResponse:
        rid = _request_id(request)
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(
                message=f"unknown model: {exc.name!r}",
                code="unknown_model",
                type_="invalid_request_error",
                request_id=rid,
            ),
        )

    @app.exception_handler(ImageFetchError)
    async def _image_error(request: Request, exc: ImageFetchError) -> ORJSONResponse:
        rid = _request_id(request)
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                message=str(exc),
                code="image_fetch_error",
                type_="invalid_request_error",
                request_id=rid,
            ),
        )

    @app.exception_handler(ConcurrencyError)
    async def _concurrency_error(request: Request, exc: ConcurrencyError) -> ORJSONResponse:
        rid = _request_id(request)
        return ORJSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=_error_body(
                message=str(exc),
                code="too_many_concurrent_requests",
                type_="rate_limit_error",
                request_id=rid,
            ),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> ORJSONResponse:
        rid = _request_id(request)
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(
                message=str(exc),
                code="invalid_request",
                type_="invalid_request_error",
                request_id=rid,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> ORJSONResponse:
        rid = _request_id(request)
        # Log the full traceback server-side; keep the public response generic.
        logger.exception("unhandled error on %s %s [request_id=%s]",
                         request.method, request.url.path, rid)
        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                message="internal server error",
                code="internal_error",
                type_="server_error",
                request_id=rid,
            ),
        )


def _status_to_type(code: int) -> str:
    if code == 429:
        return "rate_limit_error"
    if 400 <= code < 500:
        return "invalid_request_error"
    if 500 <= code < 600:
        return "server_error"
    return "api_error"
