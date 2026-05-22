"""Generic HTTP middleware: request-id propagation and body size limits."""

from __future__ import annotations

import uuid

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Read or generate ``X-Request-ID`` and stash it on ``request.state``."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        rid = request.headers.get(self.header_name) or uuid.uuid4().hex
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers[self.header_name] = rid
        return response


class BodySizeLimitMiddleware:
    """Reject requests whose body exceeds ``max_bytes`` early.

    Inspects the ``Content-Length`` header (if present) and also enforces a
    streaming cap. Returns ``413`` with a JSON error body that matches the
    rest of the API.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max(1024, max_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_headers = MutableHeaders(scope=scope)
        cl = request_headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self._max_bytes:
                    await _send_413(send, self._max_bytes)
                    return
            except ValueError:
                pass

        received = 0
        too_large = False

        async def receive_capped() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message.get("type") == "http.request":
                body: bytes = message.get("body", b"") or b""
                received += len(body)
                if received > self._max_bytes:
                    too_large = True
            return message

        sent_413 = False

        async def send_wrapper(message: Message) -> None:
            nonlocal sent_413
            if too_large and not sent_413:
                sent_413 = True
                await _send_413(send, self._max_bytes)
                return
            if not too_large:
                await send(message)

        try:
            await self._app(scope, receive_capped, send_wrapper)
        except Exception:
            if too_large and not sent_413:
                await _send_413(send, self._max_bytes)
                return
            raise


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Copy ``request.state.rate_limit_headers`` (if set by a route) onto
    the outgoing response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response: Response = await call_next(request)
        headers = getattr(request.state, "rate_limit_headers", None)
        if headers:
            for name, value in headers.items():
                response.headers[name] = value
        return response


async def _send_413(send: Send, max_bytes: int) -> None:
    body = (
        b'{"error":{"message":"request body too large","type":"invalid_request_error",'
        b'"code":"payload_too_large","limit_bytes":' + str(max_bytes).encode() + b"}}"
    )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
